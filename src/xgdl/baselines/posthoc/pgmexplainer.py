# https://github.com/pyg-team/pytorch_geometric/blob/72eb1b38f60124d4d700060a56f7aa9a4adb7bb0/torch_geometric/nn/models/pg_explainer.py

import math
import torch.nn as nn
import torch
import copy
import numpy as np
import pandas as pd
from scipy.special import softmax
from pgmpy.estimators.CITests import chi_square
from torch_geometric.data import Data, Batch
from ..base import BaseRandom
from tqdm import tqdm
# print = tqdm.write
# from evaluation import control_sparsity


class MetaPGMExplainer:
    def __init__(
            self,
            model,
            graph,
            pred_threshold=0.5,
            perturb_mode="mean",  # mean, zero, max or uniform
            perturb_indicator="diff",
    ):
        self.model = model
        # self.model.eval()
        self.graph = graph
        # self.num_layers = num_layers
        # self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        # self.perturb_pct = perturb_pct
        self.X_feat = graph.x
        self.geo_feat = graph.pos
        self.pred_threshold = pred_threshold

    def perturb_on_graph(self, node_idx, x_level):
        candi_feat = self.X_feat if x_level == 'graph' else self.geo_feat if x_level == 'geometric' else None

        # X_perturb = copy.deepcopy(candi_feat)
        perturb_array = copy.deepcopy(candi_feat)
        epsilon = 0.05 * torch.max(candi_feat, dim=0).values
        epsilon = epsilon.detach().cpu().numpy()

        for i in node_idx:
            if self.perturb_mode == "mean":
                perturb_array[i] = torch.mean(candi_feat, dim=0)
            elif self.perturb_mode == "zero":
                perturb_array[i] = torch.zeros((candi_feat.shape[1], 1))
            elif self.perturb_mode == "max":
                perturb_array[i] = torch.max(candi_feat, dim=0).values
            elif self.perturb_mode == "uniform":
                perturb_array[i] = perturb_array[i] + np.random.uniform(low=-epsilon, high=epsilon)
            else:
                assert self.perturb_mode == 'split'
                tmp_graph = self.graph.clone().cpu()
                all_idx = np.arange(self.graph.num_nodes)
                node_idx = np.setdiff1d(all_idx, node_idx)

                new_x = tmp_graph.x[node_idx]
                new_pos = None if tmp_graph.pos==None else tmp_graph.pos[node_idx]

                edge_index = tmp_graph.edge_index
                row = edge_index[0]
                new_idx = row.new_full((self.graph.num_nodes + 1,), -1)
                # print(new_idx.device)
                new_idx[node_idx] = torch.arange(node_idx.size)
                new_edge_index = new_idx[edge_index]

                new_edge_index = new_edge_index.T[(new_edge_index.T != torch.tensor([-1, -1])).all(dim=1)].T
                # delete the -1 ones

                tmp_graph.x = new_x
                tmp_graph.pos = new_pos
                tmp_graph.edge_index = new_edge_index

                return tmp_graph.to(self.graph.x.device)

        tmp_graph = self.graph.clone()
        if x_level == 'graph':
            tmp_graph.x = perturb_array
        elif x_level == 'geometric':
            tmp_graph.pos = perturb_array
        else:
            raise ValueError(f'Unknown x_level: {x_level}')

        return tmp_graph

    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                       percentage, x_level):
        clf_logits = self.model(self.graph)
        soft_pred = clf_logits.sigmoid().detach().cpu().numpy()
        num_nodes = self.X_feat.size(0)
        Samples = []
        for iteration in range(num_samples):
            # perturb_feat = copy.deepcopy(candi_feat)
            sample = np.zeros((num_nodes+1,))

            perturb_num = int(percentage / 100 * num_nodes)

            perturb_indexes = np.random.permutation(index_to_perturb)[:perturb_num]
            sample[perturb_indexes] = 1
            perturb_graph = self.perturb_on_graph(perturb_indexes, x_level)

            perturb_logits = self.model(perturb_graph)
            soft_pred_perturb = perturb_logits.sigmoid().detach().cpu().numpy()
            # self.model(tmp_g)
            # soft_pred_perturb = np.asarray(softmax(self.model.readout[0].detach().cpu().numpy()))

            pred_change = (soft_pred - soft_pred_perturb).item()
            sample[-1] = pred_change
            Samples.append(sample)
        Samples = np.asarray(Samples)
        # np.set_printoptions(precision=0, suppress=True, threshold=np.inf)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        #
        # for i in range(num_samples):
        count = 0
        reverse_val = 0
        for i in range(num_samples):
            if self.pred_threshold:
                if Samples[i, num_nodes] > self.pred_threshold:
                    count += 1
                    reverse_val += Samples[i, num_nodes]
                    Samples[i, num_nodes] = 1
                else:
                    Samples[i, num_nodes] = 0
            else:  # select most significant 1/8 changes and set to 1
                top = int(num_samples / 8)
                top_idx = np.argsort(Samples[:, num_nodes])[-top:]
                if i in top_idx:
                    count += 1
                    reverse_val += Samples[i, num_nodes]
                    Samples[i, num_nodes] = 1
                else:
                    Samples[i, num_nodes] = 0
        # if self.graph.y.item() == 1:
        #     print(f'The flipped ratio is {(count/num_samples):.2f}, average drop is {reverse_val}/{count}.')
        # if reverse_val == 0:
        #     print(Samples[:, num_nodes])
        return Samples

    def explain(self, x_level, num_samples=1000, percentage=20, top_node=5, p_threshold=0.05):

        num_nodes = self.X_feat.size(0)

        #       Round 1
        Samples = self.batch_perturb_features_on_node(int(num_samples / 2), range(num_nodes), percentage, x_level)

        data = pd.DataFrame(Samples)
        p_values = []

        target = num_nodes  # The entry for the graph classification result is at "num_nodes"
        for node in range(num_nodes):
            chi2, p, _ = chi_square(node, target, [], data, boolean=False)
            p_values.append(p)

        number_candidates = min(int(top_node * 2), num_nodes - 1)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]

        #         Round 2
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage, x_level)
        data = pd.DataFrame(Samples)

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in range(num_nodes):
            chi2, p, _ = chi_square(node, target, [], data, boolean=False)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)
        return torch.tensor(p_values, device=self.graph.x.device)


class PGMExplainer(BaseRandom):

    def __init__(self, clf, criterion, config):
        super().__init__()
        self.name = 'pgmexplainer'
        self.clf = clf
        self.device = next(self.parameters()).device
        # self.extractor = extractor
        # extractor wasn't used
        self.criterion = criterion
        self.pred_threshold = config['pred_threshold']
        self.percentage = config['percentage']
        self.perturb_mode = config['perturb_mode']

    def forward_pass(self, data, epoch, do_sampling):
        x_level = 'geometric'
        clf_logits = self.clf(data)
        batch_imp = []
        for graph in data.to_data_list():
            node_imp = self.explain_graph(graph, x_level)
            batch_imp += [node_imp]
        # imp = self.explain_graph(data, x_level)
        res_weights = self.min_max_scalar(torch.cat(batch_imp))
        return -1, {}, clf_logits, res_weights

    def explain_graph(self, graph, x_level):

        clf_logits = self.clf(graph)
        soft_pred = clf_logits.sigmoid()
        # self.model(graph)
        # soft_pred = self.model.readout

        # pred_threshold = 0.1 * torch.max(soft_pred)
        # perturb_features_list = [i ]
        explainer = MetaPGMExplainer(self.clf, graph,
                                     perturb_indicator="abs",
                                     perturb_mode=self.perturb_mode,
                                     pred_threshold=self.pred_threshold)
        pct = self.percentage / 100
        # int(graph.num_nodes / pct)
        p_values = explainer.explain(x_level=x_level, num_samples=500, p_threshold=0.05,
                                     percentage=self.percentage, top_node=int(pct * graph.num_nodes))
        # p_values = np.array(p_values)
        row, col = graph.edge_index #.detach().cpu()
        edge_imp = (1-p_values[row]) * (1-p_values[col])
        edge_imp -= torch.min(edge_imp)

        if isinstance(edge_imp, float):
            edge_imp = edge_imp.reshape([1])

        def norm_imp(imp):
            imp[imp < 0] = 0
            imp += 1e-16
            return imp / imp.sum()
        edge_imp = norm_imp(edge_imp)

        # edge_imp if x_level == 'graph' else
        return 1-p_values if x_level == 'geometric' else edge_imp


EPS = 1e-6


