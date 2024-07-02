import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from ...backbones import GINConv
# from evaluation import control_sparsity
from ..base import BaseRandom


class GNNLRP(BaseRandom):

    def __init__(self, clf, criterion, config):
        super().__init__()
        self.name = 'gnnlrp'
        self.clf = clf
        if hasattr(clf, 'clf'):
            self.target_layers = clf.clf.model.convs
        else:
            self.target_layers = clf.model.convs
        n_layers = len(self.target_layers)
        self.gammas = config.get("gammas", np.linspace(3, 0, n_layers-1))

        self.criterion = criterion
        self.device = next(self.parameters()).device

    def start_tracking(self):
        if hasattr(self.clf, 'clf'):
            self.activations_and_grads = Transforms(self.clf.clf, self.target_layers)
        else:
            self.activations_and_grads = Transforms(self.clf, self.target_layers)

    def _new_forward_pass(self, data, epoch, do_sampling, fp_16=True):
        original_clf_logits = self.activations_and_grads(data)
        H_end = self.activations_and_grads.activations[-1].data
        H_end = H_end.half() if fp_16 else H_end
        sum_nodes = 0
        node_weights_list = []
        for graph in data.to_data_list():
            add_nodes = graph.num_nodes
            relevance = H_end[sum_nodes:sum_nodes+add_nodes, :]
            # transforms = self.get_single_graph_transforms(graph, self.gammas, start=sum_nodes, fp_16=fp_16)
            generator = self.reverse_transforms_generator(graph, self.gammas, start=sum_nodes, fp_16=fp_16)
            sum_nodes = sum_nodes+add_nodes

            for transform_wrapper in generator:
                transform = transform_wrapper[0]
                nbnodes = transform.shape[0]
                nbneurons_in = transform.shape[1]
                nbneurons_out = transform.shape[3]

                # transform = transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out)
                # relevance = relevance.reshape([nbnodes * nbneurons_out, 1])
                relevance = (transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out) @ relevance.reshape([nbnodes * nbneurons_out, 1])).reshape(nbnodes, nbneurons_in)

            graph_node_weights = relevance.sum(axis=1)
            node_weights_list.append(graph_node_weights)
        node_weights = torch.cat(node_weights_list)

        res_weights = self.node_attn_to_edge_attn(node_weights, data.edge_index) if hasattr(data, 'edge_label') else node_weights
        res_weights = self.min_max_scalar(res_weights)

        return -1, {}, original_clf_logits, res_weights.sigmoid()

    def forward_pass(self, data, epoch, do_sampling, fp_16=True):
        # return self._new_forward_pass(data, epoch, do_sampling, fp_16=True)
        original_clf_logits = self.activations_and_grads(data)
        if data.x.shape[0] > 1400:
            res_weights = torch.ones(data.x.shape[0], )
            print(f"Too many nodes ({data.x.shape[0]}) in this graph, cannot process")
            return -1, {}, original_clf_logits, res_weights.sigmoid()
        # original_clf_logits = self.clf(data)
        # H_end, transforms = self.get_H_transforms(data, gammas)
        H_end = self.activations_and_grads.activations[-1].data
        H_end = H_end.half() if fp_16 else H_end
        sum_nodes = 0
        node_weights_list = []
        for graph in data.to_data_list():
            add_nodes = graph.num_nodes
            relevance = H_end[sum_nodes:sum_nodes+add_nodes, :]
            transforms = self.get_single_graph_transforms(graph, self.gammas, start=sum_nodes, fp_16=fp_16)
            sum_nodes = sum_nodes+add_nodes
            for transform in reversed(transforms):
                # einsum slow
                # relevance_subgraph = torch.einsum('ijkl,kl->ij', transform, mask @ relevance_subgraph)
                nbnodes = transform.shape[0]
                nbneurons_in = transform.shape[1]
                nbneurons_out = transform.shape[3]

                def sparse_reshape(sparse_tensor, size):
                    mask = torch.sparse_coo_tensor(indices=sparse_tensor.indices(), 
                    values=torch.ones(sparse_tensor._nnz(),device=sparse_tensor.device, dtype=torch.bool),
                    size=sparse_tensor.size())

                    mask = mask.to(torch.int64).to_dense().reshape(size)

                    sparse_tensor = torch.sparse_coo_tensor(indices=mask.to_sparse().indices(), values=sparse_tensor.values(),size=size)
                    return sparse_tensor.coalesce()

                transform = transform.reshape(nbnodes * nbneurons_in, nbnodes * nbneurons_out)
                relevance = relevance.reshape([nbnodes * nbneurons_out, 1])
                relevance = (transform @ relevance).reshape(nbnodes, nbneurons_in)
                
                #! This has been for sparse matrix version
                # relevance = relevance.to_sparse_coo()
                # size = (nbnodes * nbneurons_in, nbnodes * nbneurons_out)
                # transform = sparse_reshape(transform, size=size)
                # relevance = sparse_reshape(relevance, size=(nbnodes * nbneurons_out, 1))
                # relevance = sparse_reshape(transform @ relevance, size=(nbnodes, nbneurons_in))

            # relevance = relevance.to_dense()
            graph_node_weights = relevance.sum(axis=1)
            node_weights_list.append(graph_node_weights)
        node_weights = torch.cat(node_weights_list)

        res_weights = self.node_attn_to_edge_attn(node_weights, data.edge_index) if hasattr(data, 'edge_label') else node_weights

        res_weights = self.min_max_scalar(res_weights)

        return -1, {}, original_clf_logits, res_weights.sigmoid()

    @staticmethod
    def get_adj(data):
        adj = torch.eye(data.num_nodes, device=data.edge_index.device)
        for i, j in data.edge_index.T:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def get_last_nth_transform(self, graph, gammas, start, n=-1):
    # 首先，计算生成器的长度
        length = self.get_generator_length()
        # 如果n大于生成器长度，返回None或抛出异常
        if n > length or n <= 0:
            raise ValueError("Wrong index")
        # 计算倒数第n个值的索引（正向计数）
        target_index = length - n
        
        # 重新构造或获取生成器
        generator = self.get_transforms_generator(graph, gammas, start)  # 需要一种方法来重新获得或重置生成器
        # 遍历生成器到达目标索引
        for i, value in enumerate(generator):
            if i == target_index:
                return value[0]

    def get_generator_length(self):
        activations_list = [a.data for a in self.activations_and_grads.activations] 
        return len(activations_list)

    def reverse_transforms_generator(self, graph, gammas, start, fp_16=False):
        transforms = [0]
        activations_list = [a.data for a in self.activations_and_grads.activations]
        weight_list = [g.data for g in self.activations_and_grads.weights]
        bias_list = [b.data for b in self.activations_and_grads.biases]
        A = self.get_adj(graph)
        for W, b, H, gamma in reversed(list(zip(weight_list, bias_list, activations_list, gammas))):
            torch.cuda.empty_cache()
            W = W + gamma * W.clamp(min=0)
            b = b + gamma * b.clamp(min=0) + 1e-6
            b = b.half() if fp_16 else b
            H = H[start:start+graph.num_nodes, :]
            neuron_weight = (A.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)).permute(0, 3, 1, 2)
            neuro_feature = H.unsqueeze(0).unsqueeze(0)
            if fp_16:
                neuro_feature = neuro_feature.half()
                neuron_weight = neuron_weight.half()
            transform = (neuron_weight * neuro_feature)
            transform = transform / (transform.sum(axis=0).sum(axis=0) + b).unsqueeze(0).unsqueeze(0)
            transforms[0] = transform
            # 修改开始
            # transforms.append(transform)  # 删除这行
            yield transforms
            # 每次生成同一个列表 但元素不同 使得使用同一片内存
        torch.cuda.empty_cache()

    def get_single_graph_transforms(self, graph, gammas, start, fp_16=False):
        def memory_size(tensor):
            return tensor.element_size() * tensor.nelement() / 1024**3
        def is_sparse(tensor):
            total_elements = tensor.numel()
            non_zero_elements = torch.count_nonzero(tensor)
            sparsity_ratio = 1 - (non_zero_elements / total_elements)

            print(f"非零元素比例: {non_zero_elements / total_elements:.2f}")
            print(f"稀疏比例: {sparsity_ratio:.2f}")

        activations_list = [a.data for a in self.activations_and_grads.activations]
        weight_list = [g.data for g in self.activations_and_grads.weights]
        bias_list = [b.data for b in self.activations_and_grads.biases]
        transforms = []
        A = self.get_adj(graph)
        for W, b, H, gamma in zip(weight_list, bias_list, activations_list, gammas):
            torch.cuda.empty_cache()
            W = W + gamma * W.clamp(min=0)
            b = b + gamma * b.clamp(min=0) + 1e-6
            b = b.half() if fp_16 else b
            H = H[start:start+graph.num_nodes, :]
            neuron_weight = (A.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)).permute(0, 3, 1, 2)
            neuro_feature = H.unsqueeze(0).unsqueeze(0)

            if fp_16:
                neuro_feature = neuro_feature.half()
                neuron_weight = neuron_weight.half()
            torch.cuda.empty_cache()
            transform = (neuron_weight * neuro_feature) 
            # focus on the computation here especially the 'b'
            transform = transform / (transform.sum(axis=0).sum(axis=0) + b).unsqueeze(0).unsqueeze(0)
            # transform = transform.to_sparse()
            transforms.append(transform)
        torch.cuda.empty_cache()
        return transforms

    def get_H_transforms(self, data, gammas):
        start = 0
        activations_list = [a.cpu().data for a in self.activations_and_grads.activations]
        weight_list = [g.cpu().data for g in self.activations_and_grads.weights]
        bias_list = [b.cpu().data for b in self.activations_and_grads.biases]
        transforms = []

        for W, b, H, gamma in zip(weight_list, bias_list, activations_list, gammas):
            W = W + gamma * W.clamp(min=0)
            b = b + gamma * b.clamp(min=0)
            batch_transforms = []
            start = 0
            for i in range(data.num_graphs):
                graph = data.get_example(i)
                A = self.get_adj(graph)
                neuron_weight = (A.unsqueeze(-1).unsqueeze(-1) * W.unsqueeze(0).unsqueeze(0)).permute(0, 3, 1, 2)
                neuro_feature = H[start:start+graph.num_nodes, :].unsqueeze(0).unsqueeze(0)
                transform = neuron_weight*neuro_feature # 这里没加b？
                transform = transform / (transform.sum(axis=-1).sum(axis=-1)+b).unsqueeze(-1).unsqueeze(-1)
                start = start + graph.num_nodes
                batch_transforms.append(transform)

            batch_transform = self.to_batch_trans(batch_transforms, node_size=data.num_nodes)
            transforms.append(batch_transform)


        return activations_list[-1], transforms

    @staticmethod
    def to_batch_trans(batch_trans, node_size):
        hidden_size = batch_trans[0].shape[1]
        res = torch.zeros([node_size, node_size, hidden_size, hidden_size])
        num_nodes = 0
        for four_dim_tensor in batch_trans:
            four_dim_tensor.permute(0, 2, 1, 3)
            add_nodes = four_dim_tensor.shape[0]
            res[num_nodes:num_nodes+add_nodes, num_nodes:num_nodes+add_nodes, :, :] = four_dim_tensor
            num_nodes = num_nodes + add_nodes
        return res

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = (src_attn + dst_attn) / 2
        return edge_attn

class Transforms:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.activations = []
        self.weights = []
        self.biases = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            # self.handles.append(
            #     target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        reg_fee_dic = RegexMap(module.state_dict(), None)
        activation = output
        weight = reg_fee_dic['weight']
        bias = reg_fee_dic['bias']

        for mod in module.children():
            if isinstance(mod, GINConv):
                weight = torch.ones_like(weight)
        # weight = torch.ones_like(weight)
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.weights.append(weight.detach())
        self.biases.append(bias.detach())
        self.activations.append(activation.detach())

    def save_gradient(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.weights = []
        self.activations = []
        self.biases = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

from re import search, I
class RegexMap:
    def __init__(self, n_dic, val):
        self._items = n_dic
        self.__val = val

    def __getitem__(self, key):
        for regex in reversed(self._items.keys()):
            if search(key, regex, I):
                return self._items[regex]
        return self.__val