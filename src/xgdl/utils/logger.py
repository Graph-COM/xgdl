import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import os
from pathlib import Path
from functools import partial
from .info_utils import inherent_models, post_hoc_explainers, post_hoc_attribution
def to_item(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor


def log_epoch(epoch, phase, loss_dict, log_dict=None, seed=None, writer=None, phase_info='phase'):
    comb_dict = dict(loss_dict, **log_dict) if log_dict is not None else loss_dict
    # print(comb_dict) if log_dict else None
    des_phase = phase + ' ' if phase in ['test', 'warm'] else phase  # align tqdm desc bar
    if phase_info == 'phase':
        if seed:
            init_desc = f'[Seed {seed}, {des_phase.capitalize()} Epoch {epoch}] '
        else:
            init_desc = f'[{des_phase.capitalize()} Epoch {epoch}] '
    else:
        assert phase_info == 'dataset'  # this api is used in test_sensitivity.py
        init_desc = f'[Seed {seed}, {des_phase.capitalize()} Dataset] '
    info_desc = ' '.join([k + f': {v:.3f}' for k, v in comb_dict.items()])
    # eval_desc, org_clf_acc, org_clf_auc, exp_auc = get_eval_score(epoch, phase, log_dict, writer, batch)
    if writer is not None:
        for metric, value in comb_dict.items():
            writer.add_scalar(metric, value, epoch)
    return init_desc + info_desc


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def update_and_save_best_epoch_res(baseline, best, valid, test, epoch, model_dir, backbone_seed, seed, writer, method_name, main_metric, sub_metric=None):
    valid_metric = f'valid_{main_metric}'
    valid_sub_metric = f'valid_{sub_metric}'
    better_val = valid[main_metric] > best[valid_metric]
    if sub_metric is not None:
        same_val_but_better = (valid[main_metric] == best[valid_metric]) and (valid[sub_metric] < best[valid_sub_metric])

    # update the best_metric
    if better_val:
        init_dict = {'best_epoch': epoch}
        valid_res = {'valid_' + k: v for k, v in valid.items()}
        test_res = {'test_' + k: v for k, v in test.items()}
        best.update({k: v for d in [init_dict, valid_res, test_res] for k, v in d.items()})
        # if model_dir is not None:
        if method_name in inherent_models + ['pgexplainer']:
            save_checkpoint(baseline, model_dir, model_name=method_name, backbone_seed=backbone_seed, seed=seed)
        ## for post-hoc explaining inherent models
        # elif method_name.startswith('pgexplainer_'):
        #     assert backbone_seed == seed
        #     save_checkpoint(baseline, model_dir, model_name=method_name, backbone_seed=backbone_seed, seed=seed)

        if writer is not None:
            for metric, value in best.items():
                writer.add_scalar(f'best_{metric}', value, epoch)

    # print the best result every 10 epochs
    if epoch % 10 == 0:
        more_readable = partial(map, lambda x: x.upper() if x in ['exp', 'clf'] else x)
        init_desc = f"[Seed {seed}, Best Epoch: {best['best_epoch']}] "
        desc = ', '.join(['_'.join(more_readable(k.split('_'))) + f': {v:.3f}' for k, v in best.items() if k != 'best_epoch'])
        print('-' * 80), print(init_desc + desc), print('-' * 80)

    return best, better_val


def load_checkpoint(model, model_dir, model_name, seed, map_location=None, backbone_seed=None, verbose=True):
    load_dir = model_dir / (model_name + str(seed) + '.pt') if model_name in inherent_models + ['erm'] else model_dir / (str(backbone_seed) + model_name + str(seed) + '.pt')
    if not os.path.exists(load_dir):
        print('There is no checkpoint in', load_dir)
    else:
        checkpoint = torch.load(load_dir, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])

        print('Successfully load checkpoint in', load_dir) if verbose else None
        return True


def save_checkpoint(model, model_dir, model_name, backbone_seed, seed, save_info=None):
    # assert info is not None
    if model_name == 'erm':
        assert backbone_seed == seed
        if save_info is None:
            save_dir = model_dir / (model_name + str(seed) + '.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_dir)
        else:
            epo, acc, auc = save_info['epoch'], save_info['acc'] * 1000, save_info['auc'] * 1000
            (model_dir / 'models').mkdir(parents=True, exist_ok=True)
            mid_dir = model_dir / 'models' / f'seed{backbone_seed}-{epo}epo-{int(auc)}auc.pt'
            # final_dir = model_dir / (model_name + str(seed) + '.pt')
            torch.save({'model_state_dict': model.state_dict()}, mid_dir)
            # torch.save({'model_state_dict': model.state_dict()}, final_dir)

    elif model_name in inherent_models:
        if save_info is None:
            assert backbone_seed == seed
            save_dir = model_dir / (model_name + str(seed) + '.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_dir)
        else:
            epo, acc, auc, xauc = save_info['epoch'], save_info['acc'] * 1000, save_info['auc'] * 1000, save_info['xauc'] * 1000
            #! This use the local path
            _, setting, method_name = str(model_dir).split('/')
            temp_dir = Path('/usr/scratch/jzhu617/suppl_inherent_models/', setting, method_name)
            temp_dir.mkdir(parents=True, exist_ok=True)
            mid_dir = temp_dir / f'seed{backbone_seed}-{epo}epo-{int(auc)}auc-{int(xauc)}xauc.pt'
            # final_dir = model_dir / (model_name + str(seed) + '.pt')
            torch.save({'model_state_dict': model.state_dict()}, mid_dir)
            # torch.save({'model_state_dict': model.state_dict()}, final_dir)

    elif model_name in post_hoc_explainers:
        save_dir = model_dir / (str(backbone_seed) + model_name + str(seed) + '.pt')
        torch.save({'model_state_dict': model.state_dict()}, save_dir)
    # else: # for post-hoc explaining inherent models
    #     save_dir = model_dir / (str(backbone_seed) + model_name + str(seed) + '.pt')
    #     torch.save({'model_state_dict': model.state_dict()}, save_dir)


