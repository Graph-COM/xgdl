# https://github.com/pyg-team/pytorch_geometric/blob/72eb1b38f60124d4d700060a56f7aa9a4adb7bb0/torch_geometric/nn/models/pg_explainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data, Batch
class Orphicx(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.name = 'orphicx'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device
        self.sample_num = {'Nalpha': 25, 'Nbeta' : 25, 'NX': 2, 'NA': 1}
        self.causal_dim = 8
        self.spurious_dim = extractor.encode_size - self.causal_dim

        # self.pred_loss_coef = config.pred_loss_coef
        self.size_loss_coef = config.size_loss_coef
        self.kl_loss_coef = config.kl_loss_coef
        self.vgae_loss_coef = config.vgae_loss_coef

    def __loss__(self, att_adj, recovered_adj, org_adj, att_logits, org_logits, mu, logvar, data):

        size_loss = att_adj.mean()  # take mean makes it easier to tune coef for different datasets
        kl_loss = F.kl_div(F.log_softmax(att_logits, dim=1), F.softmax(org_logits, dim=1), reduction='batchmean')
            # self.criterion(recovered_adj.reshape(-1), org_adj.reshape(-1))
        vgae_loss = self.extractor.vgae_loss(recovered_adj, org_adj, mu, logvar, data)


        size_loss = self.size_loss_coef * size_loss
        kl_loss = self.kl_loss_coef * kl_loss
        vgae_loss = self.vgae_loss_coef * vgae_loss

        loss =  size_loss + kl_loss + vgae_loss
        loss_dict = {'loss': loss.item(), 'size': size_loss.item(), 'kl': kl_loss.item(), 'vgae':vgae_loss.item()}
        return loss, loss_dict

    def warming(self, data):
        clf_logits = self.clf(data)
        pred_loss = self.criterion(clf_logits, data.y.float())
        return pred_loss, clf_logits

    def explaining(self, data, x_level, do_sampling):
        original_clf_logits = self.clf(data, edge_attr=data.edge_attr)
        mu, logvar = self.extractor.encoder(data.x, data.edge_index) ####
        all_z = self.extractor.encode(data.x, data.edge_index)

        caul_z = torch.zeros_like(all_z)
        caul_z[:,:self.causal_dim] = all_z[:,:self.causal_dim]

        org_adj = self.get_adj(data)
        attn_adj = self.extractor.decode(caul_z)
        recovered_adj = self.extractor.decode(all_z)
        edge_attn = torch.zeros([data.num_edges, 1])

        for index, (i, j) in enumerate(data.edge_index.T):
            edge_attn[index, 0] = attn_adj[i, j]

        if x_level == 'edge':
            att = edge_attn
        else:
            att = self.edge_attn_to_node_attn(edge_attn, data)
        # edge_attn = self.node_attn_to_edge_attn(att, data.edge_index) if att_type == 'node' else att if att_type == 'edge' else None

        masked_clf_logits = self.clf(data, edge_attr=data.edge_attr, edge_attn=edge_attn)

        # batch = data.batch if att_type == 'node' else data.batch[data.edge_index[0]]

        other_loss, loss_dict = self.__loss__(attn_adj, recovered_adj, org_adj, masked_clf_logits, original_clf_logits, mu, logvar, data)
        loss_dict['caul_loss'] = self.compute_information_flow(data, self.extractor.decoder, self.clf).item()
        loss = other_loss + loss_dict['caul_loss']

        # att = node_mask.reshape(-1) if x_level == 'node' else edge_mask.reshape(-1)
        return loss, loss_dict, original_clf_logits, att.reshape(-1)

    def compute_information_flow(self, data, decoder, clf):
        # batch_loss = 0
        # for graph in data.to_data_list():
        causal_loss = []
        sub_adj_list = [self.get_adj(graph, sub_loop=True) for graph in data.to_data_list()]
        feat_list = [graph.x for graph in data.to_data_list()]
        pos_list = [graph.pos for graph in data.to_data_list()]

        for idx in random.sample(range(len(feat_list)), self.sample_num['NX']):
            causal_loss += [self.joint_uncond(decoder, clf, sub_adj_list[idx], feat_list[idx], pos_list[idx], act=torch.sigmoid)]
            # causal_loss += _causal_loss
            for A_idx in random.sample(range(len(feat_list)), self.sample_num['NA']-1):
                causal_loss += [self.joint_uncond(decoder, clf, sub_adj_list[A_idx], feat_list[idx], pos_list[idx], act=torch.sigmoid)]

        return torch.stack(causal_loss).mean()
        # batch_loss += graph_loss
    """
    joint_uncond:
        Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
    Inputs:
        - params['Nalpha'] monte-carlo samples per causal factor
        - params['Nbeta']  monte-carlo samples per noncausal factor
        - params['K']      number of causal factors
        - params['L']      number of noncausal factors
        - params['M']      number of classes (dimensionality of classifier output)
        - decoder
        - classifier
        - device
    Outputs:
        - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
        - info['xhat']
        - info['yhat']
    """


    def joint_uncond(self, decoder, classifier, adj, feat, pos, act=torch.sigmoid, mu=0, std=1,
                     device=None):
        params = self.sample_num
        params['M'] = self.extractor.output_size

        eps = 1e-8
        I = 0.0
        q = torch.zeros(params['M'], device=device)
        feat = feat.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
        adj = adj.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
        all_pos = pos.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
        if torch.is_tensor(mu):
            alpha_mu = mu[:, :self.causal_dim]
            beta_mu = mu[:, self.causal_dim:]

            alpha_std = std[:, :self.causal_dim]
            beta_std = std[:, self.causal_dim:]
        else:
            alpha_mu = 0
            beta_mu = 0
            alpha_std = 1
            beta_std = 1

        alpha = torch.randn((params['Nalpha'], adj.shape[-1], self.causal_dim), device=device).mul(alpha_std).add_(
            alpha_mu).repeat(1, params['Nbeta'], 1).view(params['Nalpha'] * params['Nbeta'], adj.shape[-1], self.causal_dim)
        beta = torch.randn((params['Nalpha'] * params['Nbeta'], adj.shape[-1], self.spurious_dim), device=device).mul(
            beta_std).add_(beta_mu)
        zs = torch.cat([alpha, beta], dim=-1)
        xhat = act(decoder(zs)) * adj
        data_list = []
        for index in range(zs.shape[0]):
            x = zs[index]
            one_pos = all_pos[index]
            edge_index = self.get_edge_index(xhat[index])
            edge_attn = self.get_edge_attn(xhat[index], edge_index)
            data_list += [Data(x=x, edge_index=edge_index, pos=one_pos, edge_attn=edge_attn)]
        data = Batch.from_data_list(data_list)
        # if node_idx is None:
        logits = classifier(data, edge_attn=data.edge_attn, skip_emb=True)
        # else:
        # logits = torch.cat()
        #     logits = classifier(feat, self.get_edge_index(xhat))[:, node_idx, :]
        # two-classes
        yhat = torch.cat([torch.sigmoid(logits) , 1-torch.sigmoid(logits)], dim=1)
        yhat = yhat.view(params['Nalpha'], params['Nbeta'], 2)

        p = yhat.mean(1)
        I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
        q = p.mean(0)
        I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
        return -I

    @staticmethod
    def edge_attn_to_node_attn(edge_attn, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        node_attn = torch.zeros([num_nodes, 1])
        for index, (i, j) in enumerate(edge_index.T):
            node_attn[i] += edge_attn[index, 0]
            node_attn[j] += edge_attn[index, 0]
        # edge_attn = 1 - (1-src_attn) * (1-dst_attn)
        max_ = max(node_attn)
        min_ = min(node_attn)
        node_attn = (node_attn - min_) / (max_ - min_)
        return node_attn

    @staticmethod
    def get_adj(data, sub_loop=False):
        if sub_loop:
            adj = torch.zeros([data.num_nodes, data.num_nodes])
        else:
            adj = torch.eye(data.num_nodes)
        for i, j in data.edge_index.T:
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    @staticmethod
    def get_edge_index(matrix):
        """输入图的邻接矩阵，输出邻接表"""
        col, row = [], []
        N = len(matrix)
        for i in range(N):
            tmp1 = {}
            for j in range(i):
                if (matrix[i][j] and (i != j)):
                    col += [i]
                    row += [j]
        col = torch.tensor(col).unsqueeze(0)
        row = torch.tensor(row).unsqueeze(0)
        return torch.cat([col, row], dim=0)

    @staticmethod
    def get_edge_attn(matrix, edge_index):
        edge_attr = torch.zeros([edge_index.shape[1], 1])
        for index, edge_pair in enumerate(edge_index.T):
            i, j = edge_pair
            edge_attr[index, 0] = matrix[i][j]

        return edge_attr