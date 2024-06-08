import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
from ..base import BaseRandom

def split_batch(g, edge_index):
    split = degree(g.batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

def relabel(x, edge_index, batch, pos=None):

    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    # try:
    #     assert batch[sub_nodes].unique().shape[0] == 128
    batch = batch[sub_nodes]

    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


class CIGA(BaseRandom):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.name = 'ciga'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.contrast_loss_coef = config['contrast_loss_coef']
        self.hinge_loss_coef = config['hinge_loss_coef']
        # self.reg_loss_coef = config.reg_loss_coef
        self.causal_ratio = config['causal_ratio']
        # self.contrast_method = config['contrast_method']



    def __loss__(self, causal_logits, conf_logits, clf_labels, contrast_loss):
        # pred_loss = self.criterion()
        pred_loss = self.criterion(causal_logits, clf_labels.float())
        causal_loss = self.criterion(causal_logits, clf_labels.float(), reduction='none').reshape(-1)
        conf_loss = self.criterion(conf_logits, clf_labels.float(), reduction='none').reshape(-1)
        # causal_loss = [self.criterion(causal_logits[i], clf_labels[i].float()) for i in range(conf_logits.shape[0])]
        # conf_loss = [self.criterion(conf_logits[i], clf_labels[i].float()) for i in range(conf_logits.shape[0])]

        spu_loss_weight = torch.zeros(conf_loss.size()).to(clf_labels.device)
        spu_loss_weight[conf_loss > causal_loss] = 1.0
        hinge_loss = conf_loss.dot(spu_loss_weight) / ((conf_loss > causal_loss).sum() + 1e-6)

        hinge_loss = self.hinge_loss_coef * hinge_loss
        contrast_loss = self.contrast_loss_coef * contrast_loss

        loss = pred_loss + hinge_loss + contrast_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'contrast': contrast_loss.item(), 'hinge': hinge_loss.item()}
        return loss, loss_dict

    def forward(self, data, edge_attn=None):
        emb, edge_index = self.clf.get_emb(data, edge_attn=edge_attn)
        node_weight = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        pred_edge_weight = self.node_attn_to_edge_attn(node_weight, edge_index)
        # pred_edge_weight = self.linear(edge_rep).view(-1)
        pred_edge_weight = pred_edge_weight.view(-1)

        causal_edge_index = torch.LongTensor([[], []]).to(self.device)
        causal_edge_weight = torch.tensor([]).to(self.device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data, edge_index)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.causal_ratio * N)
            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach()
            rank = np.argpartition(-single_mask_detach.cpu().numpy(), n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
            # the edges are quite a few so that we directly use the original graph
            if n_reserve == 0:
                causal_edge_index = torch.cat([causal_edge_index, edge_index], dim=1)
                causal_edge_weight = torch.cat([causal_edge_weight, single_mask])
            else:
                causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
                causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])

        causal_x, causal_edge_index, causal_batch, causal_pos = relabel(data.x, causal_edge_index, data.batch, data.pos)

        if hasattr(data, "x_lig"):
            causal_data = Batch(batch=causal_batch, edge_index=causal_edge_index, x=causal_x, pos=causal_pos,
                            x_lig=data.x_lig, pos_lig=data.pos_lig, x_lig_batch=data.x_lig_batch)
        else:
            causal_data = Batch(batch=causal_batch, edge_index=causal_edge_index, x=causal_x, pos=causal_pos)

        causal_node_emb = self.clf.get_emb(causal_data, edge_attn=causal_data.edge_weight)[0]
        causal_logits = self.clf.get_pred_from_emb(causal_node_emb, causal_data.batch)
        return causal_logits

    def forward_pass(self, data, epoch, do_sampling, **kwargs):
        # x = self.clf.get_emb(data, edge_attr=data.edge_attr)
        emb, edge_index = self.clf.get_emb(data)
        # original_clf_logits = self.clf(data, edge_attr=data.edge_attr)
        node_weight = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        pred_edge_weight = self.node_attn_to_edge_attn(node_weight, edge_index)
        # pred_edge_weight = self.linear(edge_rep).view(-1)

        pred_edge_weight = pred_edge_weight.view(-1)
        device = data.x.device
        causal_edge_index = torch.LongTensor([[], []]).to(self.device)
        causal_edge_weight = torch.tensor([]).to(self.device)
        # causal_edge_attr = torch.tensor([]).to(device)
        spu_edge_index = torch.LongTensor([[], []]).to(self.device)
        spu_edge_weight = torch.tensor([]).to(self.device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data, edge_index)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.causal_ratio * N)

            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach()
            rank = np.argpartition(-single_mask_detach.cpu().numpy(), n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            spu_edge_index = torch.cat([spu_edge_index, edge_index[:, idx_drop]], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            spu_edge_weight = torch.cat([spu_edge_weight, -1 * single_mask[idx_drop]])

            # causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            # spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

        causal_x, causal_edge_index, causal_batch, causal_pos = relabel(data.x, causal_edge_index, data.batch, data.pos)
        spu_x, spu_edge_index, spu_batch, spu_pos = relabel(data.x, spu_edge_index, data.batch, data.pos)

        if hasattr(data, "x_lig"):
            causal_data = Batch(batch=causal_batch, edge_index=causal_edge_index, x=causal_x, pos=causal_pos,
                            x_lig=data.x_lig, pos_lig=data.pos_lig, x_lig_batch=data.x_lig_batch)
            conf_data = Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, pos=spu_pos,
                              x_lig=data.x_lig, pos_lig=data.pos_lig, x_lig_batch=data.x_lig_batch)
        else:
            causal_data = Batch(batch=causal_batch, edge_index=causal_edge_index, x=causal_x, pos=causal_pos)
            conf_data = Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, pos=spu_pos)

        causal_node_emb = self.clf.get_emb(causal_data, edge_attn=causal_data.edge_weight)[0]
        conf_node_emb = self.clf.get_emb(conf_data, edge_attn=conf_data.edge_weight)[0]
        causal_graph_emb = self.clf.pool(causal_node_emb, causal_data.batch)

        causal_logits = self.clf.get_pred_from_emb(causal_node_emb, causal_data.batch)
        # this is not the same as the origin dir
        conf_logits = self.clf.get_pred_from_emb(conf_node_emb, conf_data.batch)


        contrast_loss = self.get_contrast_loss(causal_graph_emb, data.y.reshape(-1), norm=1)
        loss, loss_dict = self.__loss__(causal_logits, conf_logits, data.y, contrast_loss)

        res_weights = self.node_attn_to_edge_attn(node_weight, data.edge_index) if hasattr(data, 'edge_label') else node_weight

        return loss, loss_dict, causal_logits, res_weights.reshape(-1)


    def create_new_data(self, graph, idx, edge_weight):
        x = graph.x
        edge_index = graph.edge_index[:, idx]
        edge_attr = graph.edge_attr[idx] if graph.edge_attr is not None else None

        # node relabel
        num_nodes = x.size(0)
        sub_nodes = torch.unique(edge_index)

        x = x[sub_nodes]

        pos = graph.pos[sub_nodes] if graph.pos is not None else None
        # pos = pos

        row, col = edge_index
        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]

        return Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight)

    def get_contrast_loss(self, causal_rep, labels, norm=None, contrast_t=1.0, sampling='mul', y_pred=None):

        if norm != None:
            causal_rep = F.normalize(causal_rep)
        if sampling.lower() in ['mul', 'var']:
            # modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
            device = causal_rep.device
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            # tile mask: no need
            # mask = mask.repeat(anchor_count, contrast_count)
            batch_size = labels.size(0)
            anchor_count = 1
            # mask-out self-contrast cases (set the diagonal to 0)
            logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
            mask = mask * logits_mask
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            # min_value = torch.min(log_prob)
            # log_prob = torch.where(log_prob == torch.inf, 2 * min_value, log_prob)

            # print(log_prob)
            # print(mask.sum(1))
            # compute mean of log-likelihood over positive
            is_valid = mask.sum(1) != 0
            mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
            # some classes may not be sampled by more than 2
            # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

            # loss
            # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
            # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
            contrast_loss = -mean_log_prob_pos.mean()
            if sampling.lower() == 'var':
                contrast_loss += mean_log_prob_pos.var()
        elif sampling.lower() == 'single':
            N = causal_rep.size(0)
            pos_idx = torch.arange(N)
            neg_idx = torch.randperm(N)
            for i in range(N):
                for j in range(N):
                    if labels[i] == labels[j]:
                        pos_idx[i] = j
                    else:
                        neg_idx[i] = j
            contrast_loss = -torch.mean(
                torch.bmm(causal_rep.unsqueeze(1), causal_rep[pos_idx].unsqueeze(1).transpose(1, 2)) -
                torch.matmul(causal_rep.unsqueeze(1), causal_rep[neg_idx].unsqueeze(1).transpose(1, 2)))
        elif sampling.lower() == 'cncp':
            # correct & contrast with hard postive only https://arxiv.org/abs/2203.01517
            device = causal_rep.device
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            # tile mask: no need
            # mask = mask.repeat(anchor_count, contrast_count)
            batch_size = labels.size(0)
            anchor_count = 1
            # mask-out self-contrast cases
            logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
            mask = mask * logits_mask
            # find hard postive & negative
            pos_mask = y_pred != labels
            neg_mask = y_pred == labels

            # hard negative: diff label && correct pred
            neg_mask = torch.logical_not(mask)  # * neg_mask
            # hard positive: same label && incorrect pred
            pos_mask = mask * pos_mask

            # compute log_prob
            neg_exp_logits = torch.exp(logits) * neg_mask
            pos_exp_logits = torch.exp(logits) * pos_mask
            log_prob = logits - \
                       torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                                 neg_exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            is_valid = pos_mask.sum(1) != 0
            mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
            # some classes may not be sampled by more than 2
            # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

            # loss
            # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
            # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
            contrast_loss = -mean_log_prob_pos.mean()
        elif sampling.lower() == 'cnc':
            # correct & contrast https://arxiv.org/abs/2203.01517
            device = causal_rep.device
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            # tile mask: no need
            # mask = mask.repeat(anchor_count, contrast_count)
            batch_size = labels.size(0)
            anchor_count = 1
            # mask-out self-contrast cases
            logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                        torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
            mask = mask * logits_mask
            # find hard postive & negative
            pos_mask = y_pred != labels
            neg_mask = y_pred == labels
            # hard negative: diff label && correct pred
            neg_mask = torch.logical_not(mask) * neg_mask * logits_mask
            # hard positive: same label && incorrect pred
            pos_mask = mask * pos_mask
            if neg_mask.sum() == 0:
                neg_mask = torch.logical_not(mask)
            if pos_mask.sum() == 0:
                pos_mask = mask
            # compute log_prob
            neg_exp_logits = torch.exp(logits) * neg_mask
            pos_exp_logits = torch.exp(logits) * pos_mask
            log_prob = logits - \
                       torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                                 neg_exp_logits.sum(1, keepdim=True) + 1e-12)
            # compute mean of log-likelihood over positive
            is_valid = pos_mask.sum(1) != 0
            mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
            contrast_loss = -mean_log_prob_pos.mean()
        else:
            raise Exception("Not implmented contrasting method")
        return contrast_loss

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        # edge_attn = 1 - (1 - src_attn) * (1 - dst_attn)
        edge_attn = src_attn * dst_attn
        # edge_attn = (src_attn + dst_attn) / 2
        return edge_attn
