import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np


class BaseRandom(nn.Module):

    def __init__(self):
        super().__init__()
        # self.clf = clf
        # self.extractor = extractor
        # self.criterion = criterion
        # self.device = next(self.parameters()).device

    def warming(self, data):
        clf_logits = self.clf(data)
        pred_loss = self.criterion(clf_logits, data.y.float())
        return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}, clf_logits, None
        

    def forward(self, data):
        original_clf_logits = self.clf(data)
        return original_clf_logits

    def forward_pass(self, data, epoch, do_sampling):
        node_labels = data.node_label
        emb, edge_index = self.clf.get_emb(data)

        # node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=None)

        node_attn = torch.rand_like(edge_index[0])
            # self.sampling(node_attn_log_logits, do_sampling)
        # edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        # masked_clf_logits = self.clf(data, edge_attn=edge_attn, node_attn=node_attn)
        original_clf_logits = self.clf(data)

        # loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch)
        return -1, {'pred': -1}, original_clf_logits, original_clf_logits, node_attn.reshape(-1)

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn

    @staticmethod
    def min_max_scalar(attn, const=0.5):
        if torch.isnan(attn).any():
            print(f"Remind there are nan in {torch.isnan(attn).nonzero()}!")
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        if attn.max() - attn.min() == 0:
            return torch.full_like(attn, fill_value=const)
        else:
            return (attn - attn.min()) / (attn.max() - attn.min())


class LabelPerturb(nn.Module):

    def __init__(self, model, mode):
        super().__init__()
        self.name = 'label_perturb'
        self.mode = mode
        self.clf = model
        self.device = next(self.parameters()).device

    def forward_pass(self, data, epoch, do_sampling):
        node_labels = data.node_label
        node_attn = torch.rand_like(node_labels)
        # emb, edge_index = self.clf.get_emb(data)
        original_clf_logits = self.clf(data)

        if self.mode == 1:
            node_attn[torch.where(node_labels == 1)[0]] = 1
        else:
            assert self.mode == 0
            node_attn[torch.where(node_labels == 1)[0]] = 0

        # from sklearn.metrics import roc_auc_score
        # msk_emb = emb.clone()
        # msk_emb[torch.where(data.node_label == 0)] = torch.zeros((msk_emb.shape[1],))
        # mask_clf_logits = self.clf.get_pred_from_emb(msk_emb, data.batch)
        # org_pred = original_clf_logits.sigmoid()
        # msk_pred = mask_clf_logits.sigmoid()
        # org_score = roc_auc_score(data.y.cpu(), original_clf_logits.sigmoid().cpu())
        # msk_score = roc_auc_score(data.y.cpu(), mask_clf_logits.sigmoid().cpu())
        # roc_auc_score(data.y.cpu(), original_clf_logits.sigmoid().cpu())

        return -1, {'pred': -1}, original_clf_logits, node_attn.reshape(-1)
