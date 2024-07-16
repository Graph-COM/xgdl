import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np
from torch_sparse import transpose
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from ..base import BaseRandom


def relabel(x, edge_index, batch, pos=None):

    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def split_batch(g, edge_index):
    split = degree(g.batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


class DIR(BaseRandom):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.name = 'dir'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device
        self.epoch = None
        self.causal_loss_coef = config['causal_loss_coef']
        self.conf_loss_coef = config['conf_loss_coef']
        self.reg_loss_coef = config['reg_loss_coef']
        self.causal_ratio = config['causal_ratio']
        self.split_data = 'old'
        self.test_flag = True

    def __loss__(self, causal_pred, conf_pred, clf_labels, epoch):
        detach_conf_out = conf_pred.clone().detach()

        causal_loss = self.criterion(causal_pred, clf_labels.float())
        conf_loss = self.criterion(conf_pred, clf_labels.float())

        env_loss = torch.tensor([]).to(conf_pred.device)
        for single_conf_pred in detach_conf_out:
            combine_pred = torch.sigmoid(single_conf_pred) * causal_pred
            tmp = self.criterion(combine_pred, clf_labels.float())
            env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])
        reg_loss = env_loss.mean() + torch.var(env_loss * causal_pred.size(0))
        # all_loss = [self.criterion(conf_pred[i], clf_labels[i].float()) for i in range(conf_pred.shape[0])]

        causal_loss = self.causal_loss_coef * causal_loss
        conf_loss = self.conf_loss_coef * conf_loss
        reg_loss = self.reg_loss_coef * (epoch ** 1.6) * reg_loss

        loss = causal_loss + conf_loss + reg_loss
        loss_dict = {'loss': loss.item(), 'pred': causal_loss.item(), 'conf': conf_loss.item(), 'reg': reg_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, do_sampling, **kwargs):
        emb, edge_index = self.clf.get_emb(data)
        # original_clf_logits = self.clf(data, edge_attr=data.edge_attr)
        node_weight = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        pred_edge_weight = self.node_attn_to_edge_attn(node_weight, edge_index)

        if self.split_data == 'continuous':
            pred_edge_weight = pred_edge_weight.view(-1)
            device = data.x.device
            causal_edge_weight = pred_edge_weight.clone()
            spu_edge_weight = -pred_edge_weight.clone()

            edge_indices, _, _, num_edges, cum_edges = split_batch(data)
            for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
                n_reserve = int(self.causal_ratio * N)
                # single_mask = pred_edge_weight[C:C + N]
                single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
                # rank = np.argpartition(-single_mask_detach, n_reserve)
                # idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

                idx_reserve = single_mask_detach.argsort()[-n_reserve:]
                idx_drop = single_mask_detach.argsort()[:-n_reserve]

                if self.test_flag:
                    causal_edge_weight[C:C + N][idx_reserve] = 1
                causal_edge_weight[C:C + N][idx_drop] = 0
                    # = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
                spu_edge_weight[C:C + N][idx_reserve] = 0

                # causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
                # spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

        else:
            assert self.split_data == 'old'
            pred_edge_weight = pred_edge_weight.view(-1)
            device = data.x.device
            causal_edge_index = torch.LongTensor([[], []]).to(device)
            causal_edge_weight = torch.tensor([]).to(device)
            # causal_edge_attr = torch.tensor([]).to(device)
            spu_edge_index = torch.LongTensor([[], []]).to(device)
            spu_edge_weight = torch.tensor([]).to(device)
            # spu_edge_attr = torch.tensor([]).to(device)

            edge_indices, _, _, num_edges, cum_edges = split_batch(data, edge_index)
            for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
                n_reserve = int(self.causal_ratio * N)

                # edge_attr = data.edge_attr[C:C + N]
                single_mask = pred_edge_weight[C:C + N]
                single_mask_detach = pred_edge_weight[C:C + N].detach()
                rank = np.argpartition(-single_mask_detach.cpu().numpy(), n_reserve)
                idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
                # if debug:
                #     print(n_reserve)
                #     print(idx_reserve)
                #     print(idx_drop)
                causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
                spu_edge_index = torch.cat([spu_edge_index, edge_index[:, idx_drop]], dim=1)

                causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
                spu_edge_weight = torch.cat([spu_edge_weight, -1 * single_mask[idx_drop]])

                # causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
                # spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

            causal_x, causal_edge_index, causal_batch, causal_pos = relabel(data.x, causal_edge_index, data.batch, data.pos)
            spu_x, spu_edge_index, spu_batch, spu_pos = relabel(data.x, spu_edge_index, data.batch, data.pos)

            causal_data = Batch(batch=causal_batch, edge_index=causal_edge_index,
                                x=causal_x, pos=causal_pos)
            conf_data = Batch(batch=spu_batch, edge_index=spu_edge_index,
                                x=spu_x, pos=spu_pos)

        if self.split_data == 'continuous':
            causal_node_emb = self.clf.get_emb(data, edge_attn=causal_edge_weight.reshape(-1, 1))[0]
            conf_node_emb = self.clf.get_emb(data, edge_attn=spu_edge_weight.reshape(-1, 1))[0].detach()
            causal_out = self.clf.get_pred_from_emb(causal_node_emb, data.batch)
            conf_out = self.clf.get_pred_from_spu_emb(conf_node_emb, data.batch)
        else:
            causal_node_emb = self.clf.get_emb(causal_data, edge_attn=causal_data.edge_weight)[0]
            conf_node_emb = self.clf.get_emb(conf_data, edge_attn=conf_data.edge_weight)[0].detach()
            # causal_graph_emb = self.clf.pool(causal_node_emb, causal_data.batch)
            # conf_graph_emb = self.clf.pool(conf_node_emb, conf_data.batch)
            causal_out = self.clf.get_pred_from_emb(causal_node_emb, causal_data.batch)
            conf_out = self.clf.get_pred_from_spu_emb(conf_node_emb, conf_data.batch)

        loss, loss_dict = self.__loss__(causal_out, conf_out, data.y, epoch)
        combine_out = torch.sigmoid(conf_out) * causal_out
        # res_attn = pred_edge_weight
        res_attn = pred_edge_weight if hasattr(data, 'edge_label') else node_weight

        return loss, loss_dict, combine_out, res_attn.reshape(-1)

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        # edge_attn = 1 - (1 - src_attn) * (1 - dst_attn)
        edge_attn = src_attn * dst_attn
        # edge_attn = (src_attn + dst_attn) / 2
        return edge_attn
