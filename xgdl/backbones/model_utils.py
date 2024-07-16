import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, InstanceNorm, knn_graph, radius_graph, VGAE
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from typing import Callable, Union, Tuple
from torch import Tensor
from torch_sparse import SparseTensor, fill_diag
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size


class GCNConv(gnn.GCNConv):
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GINEConv(gnn.GINEConv):

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_attn: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_attn=edge_attn, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_attn: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_attn is not None:
            return m * edge_attn
        else:
            return m


class GINConv(gnn.GINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, edge_attn: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attn=edge_attn, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attn: OptTensor = None) -> Tensor:
        if edge_attn is not None:
            return x_j * edge_attn
        else:
            return x_j

class MySequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            elif isinstance(module, InstanceNorm):
                inputs = module(inputs, batch=kwargs['batch'])
            else:
                inputs = module(inputs)
        return inputs
# class GINConv(gnn.GINConv):
#
#     def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
#                  **kwargs):
#         super().__init__(nn, eps, train_eps, **kwargs)
#         self.edge_weight = None
#         self.fc_steps = None
#         self.reweight = None
#
#     # def children(self):
#     #     if
#     #     return iter([])
#
#
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_weight: OptTensor = None, **kwargs) -> Tensor:
#         """"""
#         self.num_nodes = x.shape[0]
#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)
#
#         # propagate_type: (x: OptPairTensor)
#         if edge_weight is not None:
#             self.edge_weight = edge_weight
#             assert edge_weight.shape[0] == edge_index.shape[1]
#             self.reweight = False
#         else:
#             edge_index, _ = remove_self_loops(edge_index)
#             self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
#             if self_loop_edge_index.shape[1] != edge_index.shape[1]:
#                 edge_index = self_loop_edge_index
#             self.reweight = True
#         out = self.propagate(edge_index, x=x[0], size=None)
#
#         # if data_args.task == 'explain':
#         #     layer_extractor = []
#         #     hooks = []
#         #
#         #     def register_hook(module: nn.Module):
#         #         if not list(module.children()):
#         #             hooks.append(module.register_forward_hook(forward_hook))
#         #
#         #     def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
#         #         # input contains x and edge_index
#         #         layer_extractor.append((module, input[0], output))
#         #
#         #     # --- register hooks ---
#         #     self.nn.apply(register_hook)
#         #
#         #     nn_out = self.nn(out)
#         #
#         #     for hook in hooks:
#         #         hook.remove()
#         #
#         #     fc_steps = []
#         #     step = {'input': None, 'module': [], 'output': None}
#         #     for layer in layer_extractor:
#         #         if isinstance(layer[0], nn.Linear):
#         #             if step['module']:
#         #                 fc_steps.append(step)
#         #             # step = {'input': layer[1], 'module': [], 'output': None}
#         #             step = {'input': None, 'module': [], 'output': None}
#         #         step['module'].append(layer[0])
#         #         if kwargs.get('probe'):
#         #             step['output'] = layer[2]
#         #         else:
#         #             step['output'] = None
#         #
#         #     if step['module']:
#         #         fc_steps.append(step)
#         #     self.fc_steps = fc_steps
#         # else:
#         nn_out = self.nn(out)
#
#         return nn_out
#
#     def message(self, x_j: Tensor) -> Tensor:
#         if self.reweight:
#             edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
#             edge_weight.data[-self.num_nodes:] += self.eps
#             edge_weight = edge_weight.detach().clone()
#             edge_weight.requires_grad_(True)
#             self.edge_weight = edge_weight
#         return x_j * self.edge_weight.view(-1, 1)



class ExtractorMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm_type = config.norm_type
        dropout_p = config.dropout_p
        act_type = config.act_type
        # level = getattr(config, 'x_level', None)
        hidden_size = config.hidden_size

        self.pos_dim = getattr(config, "pos_dim", None)
        self.edge_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size * 2, 1], dropout_p, self.norm_type, act_type)
        self.node_extractor = MLP([hidden_size, hidden_size * 2, hidden_size, 1], dropout_p, self.norm_type, act_type)
        if self.pos_dim:
            self.gaussian_extractor = MLP([hidden_size, hidden_size * 2, hidden_size, config.pos_dim ** 2 + 2], dropout_p, self.norm_type, act_type)

    def forward(self, node_emb, data, att_type):
        col, row = data.edge_index
        edge_emb = torch.cat([node_emb[row], node_emb[col]], dim=-1)
        emb = edge_emb if att_type == 'edge' else node_emb
        feature_extractor = self.edge_extractor if att_type == 'edge' else self.node_extractor if att_type == 'node' else self.gaussian_extractor
        if self.norm_type == 'instance':
            batch = data.batch if emb.shape[0] == data.num_nodes else data.batch[data.edge_index[0]]
            attn_log_logits = feature_extractor(emb, batch=batch)
        else:
            attn_log_logits = feature_extractor(emb)
        return attn_log_logits

class VGAE3MLP(VGAE):
    def __init__(self, config, dataset):
        dropout_p = config.dropout_p
        self.encode_size = config.encode_size #16
        decode_size = config.decode_size #32
        init_size = dataset.num_features
        self.output_size = dataset.num_classes
        encoder = GCNEncoder(init_size, self.encode_size, dropout_p, dataset)
        decoder = InnerProductDecoderMLP(self.encode_size, decode_size, dropout_p)
        super(VGAE3MLP, self).__init__(encoder, decoder)
        # self.dc = InnerProductDecoderMLP(output_dim, decoder_hidden_dim1, decoder_hidden_dim2, dropout, act=lambda x: x)

    def vgae_loss(self, recoverd_adj, org_adj, mu, logvar, data):
        n_nodes = org_adj.shape[0]
        adj = org_adj - torch.eye(n_nodes)
        pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm =adj.shape[0] * adj.shape[0] / ((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        bce = F.binary_cross_entropy_with_logits(recoverd_adj.flatten(1).T, org_adj.flatten(1).T, pos_weight=pos_weight,
                                                 reduction='none').mean()
        cost = norm * bce
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), -1))
        return cost

class MLP(MySequential):
    def __init__(self, channels, dropout_p, norm_type, act_type):
        norm = self.get_norm(norm_type)
        act = self.get_act(act_type)

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i]))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(act())
                m.append(nn.Dropout(dropout_p))

        super(MLP, self).__init__(*m)

    @staticmethod
    def get_norm(norm_type):
        if isinstance(norm_type, str) and 'batch' in norm_type:
            return BatchNorm
        elif norm_type == 'none' or norm_type is None:
            return nn.Identity
        elif norm_type == 'instance':
            return InstanceNorm
        else:
            raise ValueError('Invalid normalization type: {}'.format(norm_type))

    @staticmethod
    def get_act(act_type):
        if act_type == 'relu':
            return nn.ReLU
        elif act_type == 'silu':
            return nn.SiLU
        elif act_type == 'none':
            return nn.Identity
        else:
            raise ValueError('Invalid activation type: {}'.format(act_type))

class GCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, dataset):
        super(GCNEncoder, self).__init__()
        feat_info = getattr(dataset, "feat_info", None)
        if feat_info['node_categorical_feat']:
            self.emb = FeatEncoder(hidden_size, feat_info['node_categorical_feat'], feat_info['node_scalar_feat'])
            input_size = hidden_size
            assert dataset.name != 'plbind'
        self.gc1 = MySequential(GCNConv(input_size, hidden_size, dropout=dropout), nn.ReLU())
        self.gc1_1 = MySequential(GCNConv(hidden_size, hidden_size, dropout=dropout), nn.ReLU())
        self.gc2 = MySequential(GCNConv(hidden_size, hidden_size, dropout=dropout), nn.Identity())
        self.gc3 = MySequential(GCNConv(hidden_size, hidden_size, dropout=dropout), nn.Identity())

    def forward(self, x, edge_index):
        if getattr(self, "emb", None):
            x = self.emb(x)
        hidden1 = self.gc1(x, edge_index)
        hidden2 = self.gc1_1(hidden1, edge_index)
        return self.gc2(hidden2, edge_index), self.gc3(hidden2, edge_index)

class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, input_dim, hidden_dim, dropout):
        super(InnerProductDecoderMLP, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, z, sigmoid=True):
        z = F.relu(self.fc(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)
        if len(z.shape) == 3:
            adj = torch.bmm(z, torch.transpose(z, 1, 2))
        else:
            adj = torch.matmul(z, z.t())

        return torch.sigmoid(adj) if sigmoid else adj

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-6, scale_init = 1.):
        super().__init__()
        self.eps = eps
        # scale = torch.zeros(1).fill_(scale_init)
        # self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors #* self.scale


class FeatEncoder(torch.nn.Module):

    def __init__(self, hidden_size, categorical_feat, scalar_feat, n_categorical_feat_to_use=-1, n_scalar_feat_to_use=-1):
        super().__init__()
        self.embedding_list = torch.nn.ModuleList()

        self.num_categorical_feat = len(categorical_feat)
        self.n_categorical_feat_to_use = self.num_categorical_feat if n_categorical_feat_to_use == -1 else n_categorical_feat_to_use
        self.num_scalar_feat_to_use = scalar_feat if n_scalar_feat_to_use == -1 else n_scalar_feat_to_use

        for i in range(self.n_categorical_feat_to_use):
            num_categories = categorical_feat[i]
            emb = torch.nn.Embedding(num_categories, hidden_size)
            self.embedding_list.append(emb)

        if self.num_scalar_feat_to_use > 0:
            assert n_scalar_feat_to_use == -1
            self.linear = torch.nn.Linear(self.num_scalar_feat_to_use, hidden_size)

        total_cate_dim = self.n_categorical_feat_to_use*hidden_size
        self.dim_mapping = torch.nn.Linear(total_cate_dim + hidden_size, hidden_size) if self.num_scalar_feat_to_use > 0 else torch.nn.Linear(total_cate_dim, hidden_size)

    def forward(self, x):
        x_embedding = []
        for i in range(self.n_categorical_feat_to_use):
            x_embedding.append(self.embedding_list[i](x[:, i].long()))

        if self.num_scalar_feat_to_use > 0:
            x_embedding.append(self.linear(x[:, self.num_categorical_feat:]))

        x_embedding = self.dim_mapping(torch.cat(x_embedding, dim=-1))
        return x_embedding
