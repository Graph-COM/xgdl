import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from math import sqrt


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

    def __init__(self, clf, extractor, criterion):
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

    def forward_pass(self, data, epoch, do_sampling, **kwargs):
        data.pos.requires_grad = True
        data.edge_attn = torch.ones((data.num_edges, 1), device=self.device)
        data.edge_attn.requires_grad = True

        emb, _ = self.clf.get_emb(data)
        org_logits = self.clf.get_pred_from_emb(emb, data.batch)
        # original_clf_logits = self.clf(data, edge_attr=data.edge_attr)

        if not hasattr(data, 'node_grads') and not hasattr(data, 'node_grads'):
            data.node_grads, data.edge_grads = torch.zeros((data.num_nodes,), device=self.device), torch.zeros((data.num_edges, 1), device=self.device)

        logits = self.clf(data, edge_attn=data.edge_attn)
        # causal_out = self.clf.get_pred_from_emb(emb, data.batch)

        loss, loss_dict = self.__loss__(logits, 0, data.y)

        # res_attn = pred_edge_weight
        res_attn = data.node_grads if not hasattr(data, 'edge_label') else data.edge_grads
        res_attn = (res_attn - res_attn.min()) / (res_attn.max() - res_attn.min())

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
