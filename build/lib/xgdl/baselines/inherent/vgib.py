import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from ..base import BaseRandom


class VGIB(BaseRandom):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.name = 'vgib'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device
        self.mse_loss = nn.MSELoss()
        self.mi_loss_coef = config['mi_loss_coef']
        self.reg_loss_coef = config['reg_loss_coef']
        self.noise_loss_coef = config['noise_loss_coef']
        self.sample = 'both'

    def __loss__(self, org_logits, noisy_logits, clf_labels, mi_loss, reg_loss):
        pred_loss = self.criterion(org_logits, clf_labels.float())
        noise_loss = self.criterion(noisy_logits, clf_labels.float())

        # r = self.get_r(self.epoch)
        # info_loss = (attn * torch.log(attn/r + 1e-6) + (1 - attn) * torch.log((1 - attn)/(1 - r + 1e-6) + 1e-6)).mean()

        mi_loss = self.mi_loss_coef * mi_loss
        reg_loss = self.reg_loss_coef * reg_loss
        noise_loss = self.noise_loss_coef * noise_loss

        loss = pred_loss + noise_loss + mi_loss + reg_loss
        loss_dict = {'pred': pred_loss.item(), 'noise': noise_loss.item(), 'mi': mi_loss.item(), 'reg': reg_loss.item()}
        return loss, loss_dict

    def forward(self, data, edge_attn=None):
        epsilon = 0.0000001
        emb, edge_index = self.clf.get_emb(data, edge_attn=edge_attn)
        original_clf_logits = self.clf.get_pred_from_emb(emb, data.batch)
        return original_clf_logits

    def forward_pass(self, data, epoch, do_sampling, **kwargs):
        epsilon = 0.0000001
        emb, edge_index = self.clf.get_emb(data)
        original_clf_logits = self.clf.get_pred_from_emb(emb, data.batch)


        attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        attn = attn_log_logits.sigmoid()
        # attn = F.softmax(attn_log_logits, dim=1)
        # edge_attn = self.node_attn_to_edge_attn(attn, data.edge_index) if att_type == 'node' else attn if att_type == 'edge' else None

        #this part is used to add noise
        # node_feature = emb
        static_node_feature = emb.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

        #this part is used to generate assignment matrix
        # abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))

        assignment = torch.cat([attn, 1-attn], dim=1)
        gumble_assign = F.gumbel_softmax(assignment)

        #This is the graph embedding
        # graph_feature = torch.sum(node_feature, dim=0, keepdim=True)

        #add noise to the node representation
        node_feature_mean = node_feature_mean.repeat(emb.shape[0], 1)

        #noisy_graph_representation
        lambda_pos = gumble_assign[:, 0].unsqueeze(dim=1)
        if self.sample == 'both':
            lambda_neg = gumble_assign[:, 1].unsqueeze(dim=1)
        else:
            assert self.sample == 'single'
            lambda_neg = (1-lambda_pos)


        #this is subgraph embedding
        # subgraph_representation = torch.sum(lambda_pos * node_feature, dim=0, keepdim=True)

        # the noise confer to the distribution of
        noisy_node_feature_mean = lambda_pos * emb + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std
        noisy_emb = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        # with torch.no_grad():
        noisy_clf_logits = self.clf.get_pred_from_emb(noisy_emb, data.batch)

        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
                    torch.sum(((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon))**2, dim=0) #+\
        #            torch.log(node_feature_std / (noisy_node_feature_std + epsilon) + epsilon)
        mi_loss = torch.mean(KL_tensor)

        EYE = torch.ones(2, device=KL_tensor.device)
        # Pos_mask = torch.FloatTensor([1, 0])

        Adj = to_dense_adj(edge_index)[0]
        Adj.requires_grad = False
        try:
            new_adj = torch.mm(torch.t(assignment), Adj)
            new_adj = torch.mm(new_adj, assignment)
            normalize_new_adj = F.normalize(new_adj, p=1, dim=1)

            norm_diag = torch.diag(normalize_new_adj)
            reg_loss = self.mse_loss(norm_diag, EYE)
        except:
            print(f"There is some graph in which some node has no edges. ({assignment.shape[0]}-{Adj.shape[0]})")
            reg_loss = torch.tensor(0, device=original_clf_logits.device)

        loss, loss_dict = self.__loss__(original_clf_logits, noisy_clf_logits, data.y, mi_loss, reg_loss)
        #cal preserve rate

        attn = self.node_attn_to_edge_attn(attn, edge_index) if hasattr(data, 'edge_label') else attn

        return loss, loss_dict, noisy_clf_logits, attn.reshape(-1)

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def sampling(self, attn_log_logits, do_sampling):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            attn_bern = ((attn_log_logits + random_noise) / self.temperature).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        # edge_attn = 1 - (1 - src_attn) * (1 - dst_attn)
        edge_attn = src_attn * dst_attn
        # edge_attn = (src_attn + dst_attn) / 2
        return edge_attn
