import copy
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from abc import ABC

def fidelity(intepretation, explainer, sparsity=0.2, symbol='+'):
    model = explainer.baseline.clf
    node_weights = control_sparsity(intepretation.node_imp, sparsity, symbol)
    device = intepretation.edge_index.device

    ## construct new subdata
    graph = intepretation
    idx = node_weights.reshape(-1).nonzero().reshape(-1)
    if not idx.numel():
        raise ValueError("Please Check the selection of sparsity")
    edge_list = []
    # print(node_weights)
    for edge_pair in graph.edge_index.T:
        edge_list += [edge_pair] if edge_pair[0] in idx and edge_pair[1] in idx else []
    if edge_list:
        edge_index = torch.vstack(edge_list).T
    else:
        edge_index = torch.tensor([], dtype=graph.edge_index.dtype, device=graph.edge_index.device).reshape(2, -1)

    if graph.edge_attr is not None:
        edge_attr_list = [graph.edge_attr[i] if (idx == edge_pair[0]).numel() and (idx == edge_pair[1]).numel()
                            else None for i, edge_pair in enumerate(graph.edge_index.T)]
        edge_attr = torch.vstack(edge_attr_list)
    else:
        edge_attr = None

    x = graph.x[idx]
    if graph.pos is not None:
        pos = graph.pos[idx]
    else:
        pos = None

    row = edge_index[0]
    node_idx = row.new_full((max(idx)+1,), -1)
    node_idx[idx] = torch.arange(idx.size(0), device=idx.device)
    edge_index = node_idx[edge_index]

    data = Data(x=x, y=graph.y, pos=pos, edge_index=edge_index, edge_attr=edge_attr, x_lig=graph.x_lig if hasattr(graph, "x_lig") else None, pos_lig=graph.pos_lig if hasattr(graph, "pos_lig") else None)

    ## compute two forward
    with torch.no_grad():
        origin_logits = model(intepretation)
        masked_logits = model(data.to(device))

    ## compute probability difference
    clf_labels = intepretation.y.clone()
    clf_labels[clf_labels == 0] = -1

    origin_pred = origin_logits.sigmoid()
    masked_pred = masked_logits.sigmoid()
    score = ((origin_pred - masked_pred) * clf_labels).item()
    return score





def x_rocauc(interpretation):
    interpretation = interpretation.cpu()
    label = interpretation.node_label
    preds = interpretation.node_imp
    try:
        score = roc_auc_score(label, preds)
    except:
        score = -1
        raise ValueError("Cannot compute roc_auc, please check whether the node labels has non-zero values.")
    return score


class BaseEvaluation(ABC):
    # def collect_batch(self, x_labels, weights, data, signal_class, x_level):
    #     pass

    def reset(self):
        pass

    def collect_batch(self, *args, **kwargs):
        pass

    def eval_epoch(self):
        pass


def partial_weights(node_weights, pos_neg):

    pos_weights = copy.deepcopy(node_weights)
    neg_weights = copy.deepcopy(1 - node_weights)

    pos_idx = torch.where(pos_weights == 1)[0]
    neg_idx = torch.where(neg_weights == 1)[0]

    pos_zero_num = int((1-pos_neg[0]) * pos_weights.sum().item())
    neg_zero_num = int((1-pos_neg[1]) * neg_weights.sum().item())
    np.random.seed(99)
    torch.cuda.manual_seed(99)
    torch.manual_seed(99)

    pos_idx = pos_idx[torch.randperm(pos_idx.size(0))[:pos_zero_num]]
    neg_idx = neg_idx[torch.randperm(neg_idx.size(0))[:neg_zero_num]]

    weights = torch.ones_like(node_weights)
    weights[pos_idx] = 0
    weights[neg_idx] = 0
    return weights


class FidelEvaluation(BaseEvaluation):
    def __init__(self, model, sparsity, type='acc', symbol='+', instance='all'):
        self.sparsity = sparsity
        self.perf = []
        self.valid = []
        self.test = []
        self.type = type
        self.symbol = symbol
        self.name = self.type.upper()+'fid'+symbol+instance+'@'+str(sparsity)
        self.instance = instance
        self.classifier = model
        self.device = next(model.parameters()).device

    def create_new_data(self, data, weights, weight_type='edge', signal_class=None, instance=None):
        sum_ = 0
        data_list = []
        count = 0
        pos_data_list = []
        for graph in data.to_data_list():
            if weight_type == 'node':
                node_weights = weights[sum_:sum_ + graph.num_nodes]
                sum_ += graph.num_nodes
                if instance == 'pos' and graph.y.item() != signal_class:
                    continue
                if instance == 'neg' and graph.y.item() == signal_class:
                    continue

                node_weights = control_sparsity(node_weights, self.sparsity, self.symbol)
                idx = node_weights.reshape(-1).nonzero().reshape(-1)
                if not idx.numel():
                    continue
                edge_list = []
                # print(node_weights)
                for edge_pair in graph.edge_index.T:
                    edge_list += [edge_pair] if edge_pair[0] in idx and edge_pair[1] in idx else []
                if edge_list:
                    edge_index = torch.vstack(edge_list).T
                else:
                    edge_index = torch.tensor([], dtype=graph.edge_index.dtype, device=graph.edge_index.device).reshape(2, -1)

                if graph.edge_attr is not None:
                    edge_attr_list = [graph.edge_attr[i] if (idx == edge_pair[0]).numel() and (idx == edge_pair[1]).numel()
                                      else None for i, edge_pair in enumerate(graph.edge_index.T)]
                    edge_attr = torch.vstack(edge_attr_list)
                else:
                    edge_attr = None

                x = graph.x[idx]
                if graph.pos is not None:
                    pos = graph.pos[idx]
                else:
                    pos = None

                row = edge_index[0]
                node_idx = row.new_full((max(idx)+1,), -1)
                node_idx[idx] = torch.arange(idx.size(0), device=idx.device)
                edge_index = node_idx[edge_index]
            else:
                graph_weights = weights[sum_:sum_ + graph.num_edges]
                sum_ += graph.num_edges
                if instance == 'pos' and graph.y.item() != signal_class:
                    continue
                if instance == 'neg' and graph.y.item() == signal_class:
                    continue

                graph_weights = control_sparsity(graph_weights, self.sparsity, self.symbol)
                idx = graph_weights.reshape(-1).nonzero().reshape(-1)
                assert idx.numel()
                x = graph.x
                edge_index = graph.edge_index[:, idx]
                edge_attr = graph.edge_attr[idx] if graph.edge_attr is not None else None

                # node relabel
                num_nodes = x.size(0)
                sub_nodes = torch.unique(edge_index)

                x = x[sub_nodes]

                if graph.pos is not None:
                    pos = graph.pos
                    pos = pos[sub_nodes]
                else:
                    pos = None


                row, col = edge_index
                # remapping the nodes in the explanatory subgraph to new ids.
                node_idx = row.new_full((num_nodes,), -1)
                node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
                edge_index = node_idx[edge_index]
            if hasattr(graph, "x_lig"):
                # print(graph)
                data_list += [Data(x=x, y=graph.y, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                                   x_lig=graph.x_lig, pos_lig=graph.pos_lig)]
            else:
                data_list += [Data(x=x, y=graph.y, pos=pos, edge_index=edge_index, edge_attr=edge_attr)]
            # data_list += [Data(x=x, y=graph.y, pos=pos, edge_index=edge_index, edge_attr=edge_attr)]
            pos_data_list += [graph]
        if "-fidelity" in self.name and count:
            print(f'There is {count} graphs with no edges in this batch. (len:{len(data_list)})')
        follow_batch = ['x_lig'] if hasattr(graph, "x_lig") else None

        if data_list and pos_data_list:
            org_data = Batch.from_data_list(pos_data_list, follow_batch=follow_batch)
            new_data = Batch.from_data_list(data_list, follow_batch=follow_batch)
            return org_data, new_data
        else:
            return None, None

    def collect_batch(self, x_labels, weights, data, signal_class, x_level):
        # data.edge_index = self.classifier.get_emb(data)[1]
        # print(data.edge_index is None)
        weights = weights.reshape(-1, 1)
        # pos_data, pos_new_data = self.get_pos_instances(data, weights, weight_type=weight_type)
        if hasattr(data, "edge_label"):
            weight_type = 'edge'
        elif x_level == 'geometric':
            weight_type = 'node'
        else:
            assert x_level == 'graph'
            weights = node_attn_to_edge_attn(weights, data.edge_index)
            weight_type = 'edge'
        # weight_type = 'node' if weights.shape[0] != data.edge_index.shape[1] else 'edge'
        # print(data.x.device, weights.device)
        pos_data, pos_new_data = self.create_new_data(data, weights, weight_type=weight_type, signal_class=signal_class, instance=self.instance)

        if pos_new_data is None:
            return -1

        # if weights.shape[0] != data.edge_index.shape[1]:    # node_weights
        #     weights = node_attn_to_edge_attn(weights, data.edge_index)
        with torch.no_grad():
            origin_logits = self.classifier(pos_data.to(self.device))
            masked_logits = self.classifier(pos_new_data.to(self.device))

        clf_labels = pos_data.y.clone()

        if self.type == 'prob':
            clf_labels[clf_labels == 0] = -1

            origin_pred = origin_logits.sigmoid()
            masked_pred = masked_logits.sigmoid()
            scores = (origin_pred - masked_pred) * clf_labels

        else:
            assert self.type == 'acc'
            # assert self.type == 'acc'
            origin_pred = (origin_logits.sigmoid() > 0.5).float()
            masked_pred = (masked_logits.sigmoid() > 0.5).float()
            scores = (origin_pred == clf_labels).float() - (masked_pred == clf_labels).float()

        self.perf.append(scores.reshape(-1))
        return scores.reshape(-1).mean().item()

    def eval_epoch(self):
        # in the phas 'train', the train_res will be -1
        if not self.perf:
            return -1
        else:
            perf = torch.cat(self.perf).cpu()

        if self.type == 'auc':
            auc_score = roc_auc_score()
        return perf.mean().item()

    def reset(self):
        self.perf = []

    def update_epoch(self, valid_res, test_res):
        self.valid.append(valid_res)
        self.test.append(test_res) if not test_res else None

        return self.valid, self.test


class PrecEvaluation(BaseEvaluation):

    def __init__(self, k) -> None:
        self.prec = []
        self.valid = []
        self.test = []
        self.k = k
        self.name = f'precision@{k}'
        self.scale = 'instance'

    def reset(self):
        self.prec = []

    def collect_batch(self, x_labels, node_att, data, signal_class, x_level):
        if node_att.dim() == 2:
            node_att = node_att.reshape(-1)
        x_labels, node_att, graph_ids = get_signal_class(x_labels, node_att, data, signal_class) 
        batch_precision = []
        for id in graph_ids.unique():
            extract = lambda data: data[graph_ids == id]
            gx_labels, gnode_att = extract(x_labels), extract(node_att)
            #* desending ranking top k
            ids_of_topk = np.argsort(-gnode_att) [:self.k]
            topk_x_labels = gx_labels[ids_of_topk]
            precision = topk_x_labels.sum().item() / self.k
            batch_precision.append(precision)

        self.prec += batch_precision
        if batch_precision:
            return sum(batch_precision) / len(batch_precision)
        else:
            return -1
    
    def eval_epoch(self):
        if not self.prec:
            return -1
        else:
            prec = self.prec
            return sum(prec) / len(prec)


class AUCEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.

    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.

    :funcion get_score: obtain the roc auc score.
    """

    def __init__(self):
        self.att = []
        self.gnd = []
        self.valid = []
        self.test = []
        self.scale = 'dataset'
        self.name = 'exp_auc'

    def collect_batch(self, x_labels, node_att, data, signal_class, x_level):
        x_labels, node_att, _ = get_signal_class(x_labels, node_att, data, signal_class)
        self.att.append(node_att)
        self.gnd.append(x_labels)
        # print(torch.where(torch.isnan(a), torch.full_like(a, 0), a))
        try:
            score = roc_auc_score(x_labels.cpu(), node_att.cpu())
        except:
            print("cannot compute roc_auc")
            score = -1
        return score


    def eval_epoch(self, return_att=False):
        # in the phase 'train', the train_res will be -1
        if not self.att:
            return -1
        else:
            att = torch.cat(self.att)
            label = torch.cat(self.gnd)
            return roc_auc_score(label.cpu(), att.cpu()) if not return_att else (att.cpu(), label.cpu())

    def reset(self):
        self.att = []
        self.gnd = []

    def update_epoch(self, valid_res, test_res):
        self.valid.append(valid_res)
        self.test.append(test_res) if not test_res else None
        return self.valid, self.test


def get_signal_class(x_labels, att, data, signal_class: int):
    # regard edges as a node set
    graph_id_of_node = data.batch.cpu()
    graph_id_of_edge = graph_id_of_node[data.edge_index[0]] if hasattr(data, 'edge_label') else None
    if len(x_labels) == data.num_nodes:
        graph_id = graph_id_of_node
    elif len(x_labels) == data.num_edges:
        graph_id = graph_id_of_edge

    graph_label = data.y.cpu()
    in_signal_class = (graph_label[graph_id] == signal_class).reshape(-1)

    return x_labels[in_signal_class], att[in_signal_class], graph_id[in_signal_class]


def control_sparsity(mask, sparsity=None, symbol='+'):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity_list we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity_list values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7

        # if len(mask.shape)>1:
        #     mask = mask.squeeze()
        _, indices = torch.sort(mask, dim=0, descending=True)
        mask_len = mask.shape[0]
        split_point = int((1 - sparsity) * mask_len)
        important_indices = indices[: split_point]
        unimportant_indices = indices[split_point:]
        trans_mask = mask.clone()
        if symbol == "+":  # larger indicates batter
            trans_mask[important_indices] = 0
            trans_mask[unimportant_indices] = 1
        else:
            assert symbol == '-' #lower indicates better
            trans_mask[important_indices] = 1
            trans_mask[unimportant_indices] = 0


        return trans_mask


def node_attn_to_edge_attn(node_attn, edge_index):
    src_attn = node_attn[edge_index[0]]
    dst_attn = node_attn[edge_index[1]]
    edge_attn = src_attn * dst_attn
    return edge_attn