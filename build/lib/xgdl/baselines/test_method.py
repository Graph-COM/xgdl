import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from collections import Counter
import torch.nn.functional as F
from .new_shapley import gnn_score, mc_shapley, l_shapley, mc_l_shapley, NC_mc_l_shapley
import networkx as nx
from typing import Callable, Optional, Tuple
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from .base import BaseRandom

def compute_scores(score_func, children):
    results = []
    not_good = 0
    label_nodes = set(torch.where(children[0].data.node_label)[0].cpu().numpy())
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
            score_func(child.coalition, child.data)
            if child.data.node_label.sum().item() != 0 and score < 0.5 and set(child.coalition).intersection(label_nodes) == label_nodes:
                not_good = 1
        else:
            score = child.P
        results.append(score)
    if max(results) < 0.5 and not_good:
        print([(set(child.coalition).intersection(label_nodes) == label_nodes) for child in children])
        print([score_func(child.coalition, child.data) for child in children])
    return results


def find_closest_node_result(results, max_nodes):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """

    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node


def find_closest_node_result_list(results, nodes_num_list):
    result_list = []
    for node_num in nodes_num_list:  # 10, 12 increase
        result_node = find_closest_node_result(results, node_num)
        results.remove(result_node)
        result_list.append(result_node)
    return result_list


def reward_func(reward_method, value_func,
                local_radius=4, sample_num=100,
                subgraph_building_method='split'):
    if reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_raduis=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_raduis=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)
    else:
        raise NotImplementedError


class MCTSNode(object):

    def __init__(self, coalition: list, data: Data, ori_graph: nx.Graph,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 mapping=None):
        self.data = data
        self.signal = 1
        self.coalition = coalition # Coalition of possible subsets of players
        self.ori_graph = ori_graph # Original input graph
        self.c_puct = c_puct # Hyperparameter in search algorithm
        self.children = [] # Children within MCTS tree
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

        self.mapping = mapping # ADDED from OWEN

    def Q(self): # Average of W
        return self.W / self.N if self.N > 0 else 0

    def U(self, n): # Action selection criteria for MCTS
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method
    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): The number of hops :math:`k`.
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node.
        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.
    """
    # score_threshold = 0.3
    def __init__(self, graph, num_hops: int, use_mcts=True, use_pruning=True,
                min_atoms: int = 3, c_puct: float = 10.0,
                 expand_atoms: int = 14, high2low: bool = False, score_func: Callable = None, score_threshold: float = 0.3):

        self.num_hops = num_hops
        self.data = graph
        self.graph = to_networkx(self.data, to_undirected=True)  # NETWORKX VERSION OF GRAPH
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low
        self.score_threshold = score_threshold
        self.use_mcts = use_mcts
        self.use_pruning = use_pruning

        inv_mapping = None

        # extract the sub-graph and change the node indices.

        self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct, mapping=inv_mapping)
        self.root = self.MCTSNodeClass(self.root_coalition)  # Root of tree
        self.state_map = {str(self.root.coalition): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, num_hops, relabel_nodes=True, num_nodes=num_nodes)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P
        if tree_node.signal == -1 and self.use_pruning:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
            all_nodes = [x[0] for x in node_degree_list]

            if len(all_nodes) < self.expand_atoms:
                expand_nodes = all_nodes
            else:
                expand_nodes = all_nodes[:self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                main_sub = subgraphs[0]
                for sub in subgraphs:
                    if sub.number_of_nodes() > main_sub.number_of_nodes():
                        main_sub = sub

                new_graph_coalition = sorted(subgraph_coalition)

                # check the state map and merge the same sub-graph
                Find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        Find_same = True

                if Find_same == False:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                Find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        Find_same_child = True

                if Find_same_child == False:
                    tree_node.children.append(new_node)


            scores = compute_scores(self.score_func, tree_node.children)

            # label_nodes = set(torch.where(self.data.node_label)[0].numpy())
            for child, score in zip(tree_node.children, scores):
                child.P = score
                if score < self.score_threshold and self.use_pruning:
                    child.signal = -1
                # if score < 0.5 and len(set(child.coalition).intersection(label_nodes)) > int(len(label_nodes)/2):
                    # print(f"Labeled Nodes: {label_nodes}, Subgraph: {set(child.coalition)}, Value: {score}")
                # elif score < self.score_threshold
                #     pass
                # elif score > self.score_threshold and set(child.coalition).isdisjoint(label_nodes) and score > 0.5:
                #     pass
                    # print(f"Labeled Nodes: {label_nodes}, Subgraph: {set(child.coalition)}, Value: {score}")

        sum_count = sum([c.N for c in tree_node.children])
        if self.use_mcts:
            selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
            v = self.mcts_rollout(selected_node)
            selected_node.W += v
            selected_node.N += 1
            return v
        else:
            assert self.use_mcts is False
            selected_node = max(tree_node.children, key=lambda x: x.P)
            # if selected_node.signal == -1:
            #     return selected_node.P
            v = self.mcts_rollout(selected_node)
            return v

    def mcts(self, verbose=True):

        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        # for rollout_idx in range(self.n_rollout):
        self.mcts_rollout(self.root)
        # if verbose:
        #     print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        # Sorts explanations based on P value (i.e. Score(.,.,.) function in MCTS)
        return explanations

class Test(BaseRandom):

    min_atoms: int = 3
    c_puct: float = 10.0
    expand_atoms = 15
    local_radius = 4
    sample_num = 100
    reward_method = 'mc_l_shapley'

    def __init__(self, clf, criterion, config):
        super().__init__()
        num_hops = None
        self.name = 'test'
        self.clf = clf
        self.device = next(self.parameters()).device
        # self.sparsity_set = config['sparsity_set']
        # extractor wasn't used
        # self.extractor = extractor
        self.num_hops = self.update_num_hops(num_hops)
        self.criterion = criterion
        self.score_threshold = 0.3
        self.use_mcts = False
        self.use_pruning = True
        self.high2low = False
        # self.subgraph_building_method = config['subgraph_building_method']  # "zero_filling"
        # # mcts hyper-parameters
        # self.min_atoms = min_atoms  # N_{min}
        # self.c_puct = c_puct
        # self.expand_atoms = expand_atoms
        # self.high2low = high2low
        #
        # # reward function hyper-parameters
        # self.local_radius = local_radius
        # self.sample_num = sample_num
        # self.reward_method = reward_method
        # self.subgraph_building_method = subgraph_building_method

    def forward_pass(self, data, epoch, do_sampling, **kwargs):
        x_level = 'geometric'
        clf_logits = self.clf(data)
        batch_imp = []
        for graph in data.to_data_list():
            imp = self.get_explanation_graph(graph, x_level)
            batch_imp += [imp]
        return -1, {}, clf_logits, torch.cat(batch_imp)

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.clf.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def get_reward_func(self, value_func):
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           local_radius=self.local_radius,
                           sample_num=self.sample_num)

    def get_mcts_class(self, graph, score_func: Callable = None):
        # only for graph classification
        return MCTS(graph, use_mcts=self.use_mcts,
                    use_pruning=self.use_pruning,
                    score_func=score_func,
                    num_hops=self.num_hops,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    expand_atoms=self.expand_atoms,
                    high2low=self.high2low,
                    score_threshold=self.score_threshold)

    def get_explanation_graph(self,
                              graph,
                              x_level: str,
                              sparsity_set=None,
                              max_nodes: int = 14,
                              ):
        '''
        Get explanation for a whole graph prediction.
        Args:
            x (torch.Tensor): Input features for every node in the graph.
            edge_index (torch.Tensor): Edge index for entire input graph.
            label (int, optional): Label for which to assume as a prediction from
                the model when generating an explanation. If `None`, this argument
                is set to the prediction directly from the model. (default: :obj:`None`)
            max_nodes (int, optional): Maximum number of nodes to include in the subgraph
                generated from the explanation. (default: :obj:`14`)
            forward_kwargs (dict, optional): Additional arguments to model.forward
                beyond x and edge_index. Must be keyed on argument name.
                (default: :obj:`{}`)
        :rtype: :class:`Explanation`
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature explanations are generated.
                exp['node_imp'] (torch.Tensor, (n,)): Node mask of size `(n,)` where `n`
                    is number of nodes in the entire graph described by `edge_index`.
                    Type is `torch.bool`, with `True` indices corresponding to nodes
                    included in the subgraph.
                exp['edge_imp'] (torch.Tensor, (e,)): Edge mask of size `(e,)` where `e`
                    is number of edges in the entire graph described by `edge_index`.
                    Type is `torch.bool`, with `True` indices corresponding to edges
                    included in the subgraph.
        '''
        if not sparsity_set:
            sparsity_set = {0.3, 0.4, 0.5, 0.6, 0.7}
        sub_nodes_list = []
        for sparsity in sorted(sparsity_set, reverse=True):
            sub_nodes_list += [int(graph.num_nodes * (1 - sparsity))]
        edge_index = graph.edge_index

        label = graph.y.item()
        # self.clf.eval()
        # pred = self.clf(graph).sigmoid()
        # label = int(pred.item())
        if label != graph.y.item():
            print(f"Gnd: {graph.y.item()}, Pred: {label}, SigNodes: {set(torch.where(graph.node_label)[0].cpu().numpy())}")
        # collect all the class index
        # logits = self.model(graph)
        # probs = F.softmax(logits, dim=-1)
        # probs = probs.squeeze()
        #
        # prediction = probs.argmax(-1)

        value_func = self._prob_score_func_graph(target_class=label)

        payoff_func = self.get_reward_func(value_func)
        self.mcts_state_map = self.get_mcts_class(graph, score_func=payoff_func)
        results = self.mcts_state_map.mcts(verbose=False)

        # best_result = find_closest_node_result(results, max_nodes=max_nodes)
        # node_mask, edge_mask = self.__parse_results(best_result, edge_index)

        best_results = find_closest_node_result_list(results, sub_nodes_list)
        node_mask, edge_mask = self.__parse_results_list(best_results, edge_index)


        node_imp = node_mask.float()
        edge_imp = edge_mask.float()

        # exp.node_imp = node_mask
        # exp.edge_imp = edge_mask

        # return {'feature_imp': None, 'node_imp': node_mask, 'edge_imp': edge_mask}
        return edge_imp if hasattr(graph, 'edge_label') else node_imp

    def _prob_score_func_graph(self, target_class):
        """
        Get a function that computes the predicted probability that the input graphs
        are classified as target classes.
        Args:
            target_class (int): the targeted class of the graph
        Returns:
            get_prob_score (callable): the probability score function
        """
        def get_prob_score(graph):
            emb, _ = self.clf.get_emb(graph)
            emb[graph.node_mask == 0] = torch.zeros((emb.shape[1],), device=graph.x.device)
            # emb[graph.node_mask == 0] = torch.mean(emb, dim=0)
            prob = self.clf.get_pred_from_emb(emb, graph.batch).sigmoid()
            score = prob if target_class == 1 else 1-prob
            # score = prob[:, target_class]
            return score

        return get_prob_score

    def __parse_results_list(self, subgraph_list, edge_index):
        num_nodes = maybe_num_nodes(edge_index)
        node_mask = torch.zeros(num_nodes, device=edge_index.device)
        imp_list = [0.1 * i for i in range(11-len(subgraph_list), 11, 1)]
        # the size of subgraph is increasing, so small subgraph has higher imp
        for subgraph, imp in zip(reversed(subgraph_list), imp_list):
            n_mask, e_mask = self.__parse_results(subgraph, edge_index, imp=imp)
            # node_mask = torch.where(node_mask == 0, n_mask, node_mask)
            node_mask = torch.where(node_mask < n_mask, n_mask, node_mask)
        edge_mask = node_mask[edge_index[0]] * node_mask[edge_index[1]]
        return node_mask, edge_mask

    def __parse_results(self, best_subgraph, edge_index, imp=1.0):
        # Function strongly based on torch_geometric.utils.subgraph function
        # Citation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#subgraph

        # Get mapping
        map = best_subgraph.mapping

        all_nodes = torch.unique(edge_index)

        subgraph_nodes = torch.tensor([map[c] for c in best_subgraph.coalition], dtype=torch.long) if map is not None \
            else torch.tensor(best_subgraph.coalition, dtype=torch.long)

        # Create node mask:
        node_mask = torch.zeros(max(all_nodes)+1, device=edge_index.device)
        node_mask[subgraph_nodes] = imp

        # Create edge_index mask
        num_nodes = maybe_num_nodes(edge_index)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        n_mask[subgraph_nodes] = imp

        edge_mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        return node_mask, edge_mask