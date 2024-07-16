import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np
from ..base import BaseRandom


class LRIBern(BaseRandom):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()

        self.name = 'lri_bern'
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.info_loss_coef = config['info_loss_coef']
        self.temperature = config['temperature']

        self.final_r = config['final_r']
        self.decay_interval = config['decay_interval']
        self.decay_r = config['decay_r']
        self.init_r = config['init_r']

        self.attn_constraint = config['attn_constraint']

    def __loss__(self, attn, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels.float())

        r = self.get_r(epoch)
        info_loss = (attn * torch.log(attn/r + 1e-6) + (1 - attn) * torch.log((1 - attn)/(1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = self.pred_loss_coef * pred_loss
        info_loss = self.info_loss_coef * info_loss

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'r': r}
        return loss, loss_dict

    def forward(self, data, edge_attn=None):
        emb, edge_index = self.clf.get_emb(data, edge_attn=edge_attn)
        original_clf_logits = self.clf.get_pred_from_emb(emb, data.batch)

        node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=None)
        if self.attn_constraint == 'smooth_min':
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)

        node_attn = self.sampling(node_attn_log_logits, False)
        msk_edge = self.node_attn_to_edge_attn(node_attn, edge_index)
        # edge_attn should be the product of msk_edge and init_edge_attn?
        masked_clf_logits = self.clf(data, edge_attn=msk_edge)
        return masked_clf_logits

    def forward_pass(self, data, epoch, do_sampling):

        emb, edge_index = self.clf.get_emb(data)
        node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=None)

        if self.attn_constraint == 'smooth_min': # specially for plbind
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)

        node_attn = self.sampling(node_attn_log_logits, do_sampling)


        # node_attn = node_attn * (data.num_nodes / node_attn.sum())
        edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        # masked_clf_logits = self.clf.get_pred_from_emb(node_attn * emb, batch=data.batch)
        masked_clf_logits =  self.clf(data, edge_attn=edge_attn)
        original_clf_logits = self.clf(data)

        loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch)
        return loss, loss_dict, masked_clf_logits, node_attn.reshape(-1)

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

