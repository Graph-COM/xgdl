import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np
from ..base import BaseRandom
from torch_geometric.nn import ASAPooling as BasePooling
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops, softmax


class ASAPooling(BasePooling):
    def __init__(self, *args, **kargs):
        super(ASAPooling, self).__init__(*args, **kargs)

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_attn=None):
        """"""
        N = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1., num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')
        x_q = self.lin(x_q)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add')

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening.
        row, col = edge_index
        A = SparseTensor(row=row, col=col, value=edge_weight,
                         sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))
        S = S[:, perm]

        A = S.t() @ A @ S

        if self.add_self_loops:
            A = A.fill_diag(1.)
        else:
            A = A.remove_diag()

        row, col, edge_weight = A.coo()
        edge_index = torch.stack([row, col], dim=0)
        if return_attn == 'fitness':
            return x, edge_index, edge_weight, batch, fitness
        else:
            assert return_attn == 'none'
            return x, edge_index, edge_weight, batch, perm


class ASAP(BaseRandom):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()

        self.name = 'asap'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.return_attn = config['return_attn']

    def __loss__(self, clf_logits, clf_labels):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        loss_dict = {'loss': pred_loss.item(), 'pred': pred_loss.item()}
        return pred_loss, loss_dict

    def forward(self, data, edge_attn=None):
        emb, edge_index = self.clf.get_emb(data, edge_attn=edge_attn)
        # extractor is ASAPooling
        casual_emb, edge_index, causal_edge_weight, batch, perm = self.extractor(emb, edge_index, batch=data.batch, return_attn=self.return_attn)
        masked_clf_logits = self.clf.get_pred_from_emb(casual_emb, batch)
        return masked_clf_logits

    def forward_pass(self, data, epoch, do_sampling):

        emb, edge_index = self.clf.get_emb(data)
        # node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        casual_emb, edge_index, causal_edge_weight, batch, attn = self.extractor(emb, edge_index, batch=data.batch, return_attn=self.return_attn)

        masked_clf_logits = self.clf.get_pred_from_emb(casual_emb, batch)
        if self.return_attn == 'none':
            node_attn = torch.zeros(data.x.size(0), device=self.device)
            node_attn[attn] = 1
        else:
            node_attn = attn
        # node_attn = self.sampling(node_attn_log_logits, do_sampling)
        # node_attn = node_attn * (data.num_nodes / node_attn.sum())
        # edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        # masked_clf_logits = self.clf.get_pred_from_emb(node_attn * emb, batch=data.batch)
        # masked_clf_logits = self.clf(data, edge_attn=edge_attn)
        # original_clf_logits = self.clf(data)

        loss, loss_dict = self.__loss__(masked_clf_logits, data.y)
        return loss, loss_dict, masked_clf_logits, node_attn.reshape(-1)

