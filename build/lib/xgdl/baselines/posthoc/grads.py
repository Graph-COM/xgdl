# https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py

import numpy as np
import torch
from torch.autograd import Variable
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from ..base import BaseRandom


class Grad(BaseRandom):

    def __init__(self, clf, criterion, config) -> None:
        super().__init__()
        self.clf = clf
        self.criterion = criterion
        self.device = next(self.parameters()).device
        # assert signal_class is not None
        self.signal_class = config['signal_class']
        self.activations_and_grads = None

    def start_tracking(self):
        if hasattr(self.clf, 'clf'):
            self.activations_and_grads = ActivationsAndGradients(self.clf.clf, self.target_layers)
        else:
            self.activations_and_grads = ActivationsAndGradients(self.clf, self.target_layers)


    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class GradX(Grad):
    def __init__(self, clf, criterion, config):
        super(GradX, self).__init__(clf, criterion, config)
        if hasattr(clf, 'model'):
            self.target_layers = [clf.model.node_encoder]  # node_embedding
        else:
            self.target_layers = [clf.clf.model.node_encoder]
        # not used if the feature is not categorical
        self.name = 'gradx'

    def forward_pass(self, data, epoch, do_sampling):
        x_level = 'geometric'
        self.clf.eval()
        try:
            is_cat_feat = False
            if x_level == 'graph':
                data.x.requires_grad = True
            else:
                data.pos.requires_grad = True
        except(RuntimeError, ValueError):
            is_cat_feat = True


        original_clf_logits = self.activations_and_grads(data)
        # original_clf_logits = self.clf(data)
        masked_clf_logits = original_clf_logits
        # org_pred = original_clf_logits.sigmoid().int()
        targets = [BinaryClassifierOutputTarget(self.signal_class)] * original_clf_logits.shape[0]
        # targets = [BinaryClassifierOutputTarget(item) for item in org_pred]

        self.clf.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, original_clf_logits)])
        loss.backward(retain_graph=True)
        loss_dict = {'loss': loss.item(), 'pred': loss.item()}

        if is_cat_feat:
            grad = self.activations_and_grads.gradients[0].squeeze().to(loss.device)
            value = self.activations_and_grads.activations[0].squeeze().to(loss.device)
            node_att = (grad * value).sum(dim=1)
        else:
            node_att = data.x.grad.norm(dim=1, p=2) if x_level == 'graph' else data.pos.grad.norm(dim=1, p=2)
                # data.pos.grad.norm(dim=1, p=2)

        res_weights = self.node_attn_to_edge_attn(node_att, data.edge_index) if hasattr(data, 'edge_label') else node_att
        # res_weights = node_att
        res_weights = self.min_max_scalar(res_weights)

        return -1, loss_dict, original_clf_logits, res_weights.reshape(-1)


class GradCAM(Grad):
    def __init__(self, clf, criterion, config):
        super(GradCAM, self).__init__(clf, criterion, config)
        if hasattr(clf, 'model'):
            self.target_layers = [clf.model.convs[-1]]
        else:
            self.target_layers = [clf.clf.model.convs[-1]]
        self.name = 'gradcam'

    def forward_pass(self, data, epoch, do_sampling):
        self.clf.eval()
        original_clf_logits = self.activations_and_grads(data)
        masked_clf_logits = original_clf_logits
        targets = [BinaryClassifierOutputTarget(self.signal_class)] * original_clf_logits.shape[0]

        self.clf.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, original_clf_logits)])
        loss.backward(retain_graph=True)
        loss_dict = {'loss': loss.item(), 'pred': loss.item()}


        cam_per_layer = self.compute_cam_per_layer()
        node_att = torch.tensor(self.aggregate_multi_layers(cam_per_layer), device=loss.device)

        # res_weights = self.node_attn_to_edge_attn(node_att, data.edge_index) if x_level == "graph" else node_att
        res_weights = self.node_attn_to_edge_attn(node_att, data.edge_index) if hasattr(data,
                                                                                    'edge_label') else node_att
        res_weights = self.min_max_scalar(res_weights)

        return -1, loss_dict, original_clf_logits, res_weights.reshape(-1)

    def get_cam(self, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        assert activations.min() >= 0.0  # rectified
        cam = (grads * activations).sum(1)
        return cam

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.stack(cam_per_target_layer, axis=1)
        result = np.mean(cam_per_target_layer, axis=1)
        return result

    def compute_cam_per_layer(self) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam(layer_activations, layer_grads)
            cam_per_target_layer.append(cam)
        return cam_per_target_layer


class InterGrad(Grad):

    def __init__(self, clf, criterion, config):
        super(InterGrad, self).__init__(clf, criterion, config)
        if hasattr(clf, 'model'):
            self.target_layers = [clf.model.node_encoder]
        else:
            self.target_layers = [clf.clf.model.node_encoder]
        self.name = 'inter_grad'

    def forward_pass(self, data, baseline=0, steps=20, **kwargs):
        # node_base = torch.zeros_like(data.x) if node_base == None else node_base
        # edge_base = torch.zeros_like(data.edge_attr) if edge_base == None else edge_base
        x_level = 'geometric'
        self.clf.eval()
        original_clf_logits = self.activations_and_grads(data)
        scales = [baseline + (float(i) / steps) * (1 - baseline) for i in range(1, steps + 1)]
        node_grads = []
        step_len = float(1 - baseline) / steps
        for scale in reversed(scales):
            if x_level == 'geometric':
                new_data_list = []
                for graph in data.to_data_list():
                    new_instance = graph.clone()
                    new_instance.pos = scale * new_instance.pos
                    new_data_list += [new_instance]
            else:
                new_data_list = []
                for graph in data.to_data_list():
                    new_instance = graph.clone()
                    new_instance.x = scale * new_instance.x
                    new_data_list += [new_instance]
                # new_data_list = [Data(x=scale * graph.x, edge_index=graph.edge_index, y=graph.y) for graph in data.to_data_list()]
            follow_batch = None if not hasattr(data, 'x_lig') else ['x_lig']
            new_data = Batch.from_data_list(new_data_list, follow_batch=follow_batch)
            if x_level == 'geometric':
                new_data.pos.requires_grad = True
            else:
                try:
                    is_cat_feat = False
                    new_data.x.requires_grad = True
                except(RuntimeError, ValueError):
                    is_cat_feat = True
            pred = self.activations_and_grads(new_data)
            # pred = self.clf(new_data, edge_attr=data.edge_attr)
            sum(pred).backward(retain_graph=True)
            if x_level == 'geometric':
                score = new_data.pos.grad.norm(dim=1, p=2)
            else:
                if is_cat_feat:
                    grad = self.activations_and_grads.gradients[0].squeeze().to(pred.device)
                    # value = self.activations_and_grads.activations[0].squeeze().to(loss.device)
                    score = grad.norm(dim=1)
                else:
                    score = new_data.x.grad.norm(dim=1, p=2)
            # score = pow(new_data.pos.grad, 2).sum(dim=1).cpu().numpy()
            node_grads.append((score * step_len).cpu())

            self.clf.zero_grad()
        node_grads = torch.tensor(np.stack(node_grads), device=data.x.device).sum(axis=0)
        # node_grads[node_grads < 0] = 1e-16
        # node_imp = (node_grads - min(node_grads)) / (max(node_grads) - min(node_grads))

        res_weights = self.node_attn_to_edge_attn(node_grads, data.edge_index) if hasattr(data, 'edge_label') else node_grads
        res_weights = self.min_max_scalar(res_weights)

        return -1, {}, original_clf_logits, res_weights.reshape(-1)


class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        # self.emb_x = torch.tensor(0)
        self.reshape_transform = reshape_transform
        self.handles = []
        # self.handles.append(model.backbone.node_encoder.register_forward_hook(self.save_activation))
        # self.handles.append(model.backbone.node_encoder.register_forward_hook(self.save_gradient))
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]
        # if isinstance(module, nn.Embedding):
        #     self.emb_x = output
        #     return

        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]
        # if isinstance(module, nn.Embedding):
        #     def _store_embx_grad(grad):
        #         self.emb_x.grad = grad
        #     output.register_hook(_store_embx_grad)
        #     return

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        if self.handles:
            for handle in self.handles:
                handle.remove()
