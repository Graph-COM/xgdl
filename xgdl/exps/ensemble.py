import yaml
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
import scipy.stats
import torch
import torch.nn as nn
from torch.nn import functional as F
from eval import FidelEvaluation, AUCEvaluation
from get_model import Model
from baselines import *
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, load_checkpoint, ExtractorMLP, get_optimizer
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from scipy.stats import spearmanr as SP
warnings.filterwarnings("ignore")

def eval_one_batch(baseline, data, epoch):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradx', 'inter_grad', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP

        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)

        if baseline.name == 'vgib':
            infer_clf_logits = baseline.forward(data)
        # from sklearn.metrics import roc_auc_score
        # roc_auc_score(infer_clf_logits, data.y.cpu())
        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def ensemble(config, method_name, model_name, backbone_seed, dataset_name, log_dir, device):
    set_seed(backbone_seed)
    print('The logging directory is', log_dir), print('=' * 80)

    # batch_size = config['optimizer']['batch_size']
    batch_size = 1
    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, dataset_seed=0)
    signal_class = dataset.signal_class

    clf = Model(model_name, config[model_name],  # backbone_config
                method_name, config[method_name],  # method_config
                dataset).to(device)

    extractor = ExtractorMLP(config[model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) \
        if method_name in inherent_models + ['pgexplainer'] else nn.Identity()
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits

    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    baseline = eval(name_mapping[method_name])(clf, extractor, criterion, config[method_name]) if method_name in inherent_models + ['pgexplainer'] \
        else eval(name_mapping[method_name])(clf, criterion, config[method_name])

    if method_name in inherent_models:
        assert load_checkpoint(baseline, log_dir, model_name=method_name, seed=backbone_seed, map_location=map_location,
                               backbone_seed=backbone_seed)
    elif method_name == 'pgexplainer':
        assert load_checkpoint(baseline, log_dir, model_name=method_name, seed=backbone_seed, map_location=map_location,
                        backbone_seed=backbone_seed)
    else:
        print(f'The method {method_name} has no extra model.')

    if method_name not in inherent_models:
        erm_dir = log_dir.parent / 'erm'
        assert load_checkpoint(baseline.clf, erm_dir, model_name='erm', seed=backbone_seed, map_location=map_location)
    # establish the model and the metrics

    baseline.start_tracking() if 'grad' in method_name or method_name == 'gnnlrp' else None
    metric_list = [AUCEvaluation()] if batch_size > 64 else None

    pbar, avg_loss_dict = tqdm(loaders['test']), dict()
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    epoch_attn, epoch_label, epoch_sig, epoch_bkg = [], [], [], []
    # sp_list = []
    for idx, data in enumerate(pbar):
        if data.y.item() == 0:
            continue

        eval_dict = {}        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = eval_one_batch(baseline, data.to(baseline.device), 1)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
        # print()
        # min_max_sclarer
        attn = (attn - attn.min()) / (attn.max() - attn.min()) if method_name not in inherent_models else attn

        epoch_attn.append(attn), epoch_label.append(ex_labels)
        epoch_sig.append(attn[torch.where(data.node_label == 1)]), epoch_bkg.append(attn[torch.where(data.node_label == 0)])

        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        eval_dict.update({f'exp_auc': roc_auc_score(ex_labels, attn)})

        # print('The spearman correlation between avg and std:', )
        desc = log_epoch(backbone_seed, 1, 'test', loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    # epoch_dict = {eval_metric.name: eval_metric.eval_epoch(return_att=True) for eval_metric in metric_list} if metric_list else {}
    # assert or  metric_list[0].name == 'exp_auc'
    epoch_attn, epoch_label, epoch_sig, epoch_bkg = torch.cat(epoch_attn), torch.cat(epoch_label), torch.cat(epoch_sig), torch.cat(epoch_bkg)
    epoch_dict = {'exp_auc': roc_auc_score(epoch_label, epoch_attn)}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    print('-' * 80, '\n', 'epoch_dict:', epoch_dict, '\n', '-' * 80)

    return epoch_attn, epoch_label, epoch_sig, epoch_bkg


def save_multi_bseeds(multi_methods_res, method_name, exp_method, seeds):
    seeds += ['avg', 'std']
    indexes = [method_name+'_'+str(seed) for seed in seeds]
    # from itertools import product
    # indexes = ['_'.join(item) for item in product(methods, seeds)]
    df = pd.DataFrame(multi_methods_res, index=indexes)

    day_dir = Path('../result') / config_name / datetime.now().strftime("%m_%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join([exp_method, method_name, 'sensitivity.csv']))
    with open(csv_dir, mode='w') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    return df


def save_multi_result(method_res, all_methods, config_name):

    df = pd.DataFrame(method_res, index=all_methods)

    day_dir = Path('../result') / config_name / 'data_insights'
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join(all_methods) + '_insights.csv')
    with open(csv_dir, mode='w') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    now_df = pd.read_csv(csv_dir, index_col=0)
    return now_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack')
    parser.add_argument('--method', type=str, help='method used')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--seeds', type=int, nargs="+", help='random seed for data insights pipeline', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # parser.add_argument('--mseed', type=int, help='random seed for explainer', default=0)
    parser.add_argument('--use_csv', action="store_true")

    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)

    dataset_name, model_name, cuda_id, note = args.dataset, args.backbone, args.cuda, args.note
    config_name = '_'.join([model_name, dataset_name])
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('../configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    all_methods = post_hoc_attribution + post_hoc_explainers + inherent_models if args.method == 'all' else [args.method]
    assert args.use_csv
    method_res = []
    for method_name in all_methods:
        method_dict = {'single_attn_auc': []}
        if config[method_name].get(model_name, False):
            config[method_name].update(config[method_name][model_name])
        print('=' * 80)
        print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))

        model_dir = Path('../log') / config_name / method_name
        multi_attn, multi_sig_attn, multi_bkg_attn = None, None, None

        for all_seed in args.seeds:
            # we directly load attns from csv file
            mseed = [0] if method_name in post_hoc_attribution else [all_seed] if method_name in inherent_models else \
                list(range(10))
            save_dir = model_dir / f"bs{all_seed}_ms{mseed}_attns.csv"
            seed_df = pd.read_csv(save_dir, index_col=0)
            epoch_label = torch.tensor(seed_df['node_labels'])
            if 'attn' in seed_df:  # this means the method has no need to vary seeds for fixed backbone_seed
                seed_attn = torch.tensor(seed_df['attn']).unsqueeze(1)
            else:  # for post-hoc explainers, there are many seed res and only use the same seed with backbone_seed
                seed_attn = torch.tensor(seed_df[str(all_seed)]).unsqueeze(1)
            method_dict['single_attn_auc'].append(roc_auc_score(epoch_label, seed_attn))
            seed_sig_attn = torch.tensor(seed_attn[seed_df['node_labels'] == 1]).unsqueeze(1)
            seed_bkg_attn = torch.tensor(seed_attn[seed_df['node_labels'] == 0]).unsqueeze(1)
            multi_attn = seed_attn if multi_attn is None else torch.cat([multi_attn, seed_attn], dim=1)
            multi_sig_attn = seed_sig_attn if multi_sig_attn is None else torch.cat([multi_sig_attn, seed_sig_attn], dim=1)
            multi_bkg_attn = seed_bkg_attn if multi_bkg_attn is None else torch.cat([multi_bkg_attn, seed_bkg_attn], dim=1)

        avg_attn, std_attn = torch.mean(multi_attn, dim=1), torch.std(multi_attn, dim=1)
        avg_sig_attn, std_sig_attn = torch.mean(multi_sig_attn, dim=1), torch.std(multi_sig_attn, dim=1)
        avg_bkg_attn, std_bkg_attn = torch.mean(multi_bkg_attn, dim=1), torch.std(multi_bkg_attn, dim=1)

        exp_auc_avg, exp_auc_std = roc_auc_score(epoch_label, avg_attn), roc_auc_score(epoch_label, 1 - std_attn)
        all_spearman, sig_spear, bkg_spear = SP(avg_attn, std_attn)[0], SP(avg_sig_attn, std_sig_attn)[0], SP(avg_bkg_attn, std_bkg_attn)[0]
        method_dict.update({'avg_attn_auc': roc_auc_score(epoch_label, avg_attn), 'std_attn_auc': roc_auc_score(epoch_label, 1-std_attn),
                        'all_attn_avg': avg_attn.mean().item(), 'sig_attn_avg': avg_sig_attn.mean().item(), 'bkg_attn_avg': avg_bkg_attn.mean().item(),
                       'all_attn_std': std_attn.mean().item(), 'sig_attn_std': std_sig_attn.mean().item(), 'bkg_attn_std': std_bkg_attn.mean().item()})
        method_dict.update({'single_attn_auc_avg': np.mean(method_dict['single_attn_auc']),
                            'single_attn_auc_std': np.var(method_dict['single_attn_auc'])})
        method_res += [method_dict]
        print(json.dumps(method_dict, indent=4))

    save_multi_result(method_res, all_methods, config_name)


