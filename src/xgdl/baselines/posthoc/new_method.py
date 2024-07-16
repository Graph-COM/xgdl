import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree

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

def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

class Test(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.name = 'test'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device
        self.epoch = 0
        self.causal_loss_coef = 1
        self.conf_loss_coef = 0
        self.reg_loss_coef = 0
        self.causal_ratio = 1

    def __loss__(self, causal_pred, conf_pred, clf_labels):
        causal_loss = self.criterion(causal_pred, clf_labels.float())

        loss = causal_loss
        loss_dict = {'loss': loss.item(), 'causal': causal_loss.item()}
        return loss, loss_dict


    def warming(self, data):
        clf_logits = self.clf(data)
        pred_loss = self.criterion(clf_logits, data.y.float())
        return pred_loss, clf_logits

    def select_epoch(self):
        imp = 1
        return imp

    def explaining(self, data, x_level, **kwargs):

        emb = self.clf.get_emb(data, edge_attr=data.edge_attr)
        detach_emb = emb.clone().detach()
        org_logits = self.clf.get_pred_from_emb(emb, data.batch)
        # original_clf_logits = self.clf(data, edge_attr=data.edge_attr)
        attach_node_weight = self.extractor(emb, data, att_type='node')
        detach_node_weight = self.extractor(detach_emb, data, att_type='node')
        attach_edge_weight = self.node_attn_to_edge_attn(attach_node_weight, data.edge_index)
        detach_edge_weight = self.node_attn_to_edge_attn(detach_node_weight, data.edge_index)
        # pred_edge_weight[:] = 1
        # pred_edge_weight = self.linear(edge_rep).view(-1)
        causal_edge_weight = detach_edge_weight.clone()
        conf_edge_weight = detach_edge_weight.clone()

        causal_edge_weight[:] = 1 if self.epoch % 2 == 1 else -1
        # causal_edge_weight[:] = self.select_epoch()
        conf_edge_weight[:] = 1
        for i in range(4):
            causal_out = self.clf(data, edge_attn=causal_edge_weight.reshape(-1, 1))
        # causal_out = self.clf.get_pred_from_emb(emb, data.batch)
        # causal_out = self.clf(data, edge_attn=causal_edge_weight.reshape(-1, 1))
        conf_out = self.clf(data, edge_attn=conf_edge_weight.reshape(-1, 1))

        loss, loss_dict = self.__loss__(causal_out, 0, data.y)

        # res_attn = pred_edge_weight
        res_attn = attach_edge_weight if hasattr(data, 'edge_label') else attach_node_weight

        return loss, loss_dict, org_logits, res_attn.reshape(-1)


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

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        # edge_attn = 1 - (1 - src_attn) * (1 - dst_attn)
        edge_attn = src_attn * dst_attn
        # edge_attn = (src_attn + dst_attn) / 2
        return edge_attn
