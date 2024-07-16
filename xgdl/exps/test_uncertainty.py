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
from baselines import PGExplainer, SubgraphX, PGMExplainer, GNNExplainer
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, load_checkpoint, ExtractorMLP, get_optimizer
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
from statistics import mean
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")


def eval_one_batch(baseline, data, epoch):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradgeo', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)
        # from sklearn.metrics import roc_auc_score
        # roc_auc_score(infer_clf_logits, data.y.cpu())
        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def one_seed_attn(baseline, data_loader, epoch, phase, seed, signal_class, writer=None, metric_list=None):
    # log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)
    return

def test_uncertainty(config, method_name, model_name, backbone_seed, method_seeds, dataset_name, log_dir, device):
    set_seed(backbone_seed, method_name=method_name, backbone_name=model_name)
    writer = None
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

    # establish the model and the metrics

    metric_list = [AUCEvaluation()] if batch_size > 64 else None

    pbar, avg_loss_dict = tqdm(loaders['test']), dict()
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    epoch_avg_attn, epoch_std_attn, epoch_label, sig_spear, bkg_spear, all_spear = [], [], [], [], [], []
    # sp_list = []
    multi_epoch_attn = None
    for idx, data in enumerate(pbar):
        if data.y.item() == 0:
            continue

        multi_attn = None
        eval_dict = {}
        for method_seed in method_seeds:
            baseline = eval(name_mapping[method_name])(clf, criterion, config[method_name]) \
                if method_name != 'pgexplainer' else PGExplainer(clf, extractor, criterion, config['pgexplainer'])
            if method_name == 'pgexplainer':
                assert load_checkpoint(baseline, log_dir, model_name=method_name, seed=method_seed, map_location=map_location,
                                   backbone_seed=backbone_seed, verbose=False)
            else:
                pass
                # print(f'Running {method_name} for seed {method_seed}.')

            erm_dir = log_dir.parent / 'erm'
            assert load_checkpoint(baseline.clf, erm_dir, model_name='erm', seed=backbone_seed, map_location=map_location, verbose=False)
            # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
            loss_dict, clf_logits, attn = eval_one_batch(baseline, data.to(baseline.device), 1)
            ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
            # print()
            eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
            eval_dict.update({f'exp_auc_{method_seed}': roc_auc_score(ex_labels, attn)})
            multi_attn = attn.unsqueeze(1) if multi_attn is None else torch.cat([multi_attn, attn.unsqueeze(1)], dim=1)

        info_graph = torch.full((multi_attn.shape[0], 1), idx)
        multi_attn_plus = torch.cat([multi_attn, ex_labels.unsqueeze(1), info_graph], dim=1)
        multi_epoch_attn = multi_attn_plus if multi_epoch_attn is None else torch.cat([multi_epoch_attn, multi_attn_plus], dim=0)

        avg_attn, std_attn = torch.mean(multi_attn, dim=1), torch.std(multi_attn, dim=1)
        batch_exp_auc_avg, batch_exp_auc_std = roc_auc_score(ex_labels, avg_attn), roc_auc_score(ex_labels, 1-std_attn)
        batch_spearman = scipy.stats.spearmanr(avg_attn, std_attn)[0]
        sig_index, bkg_index = torch.where(data.node_label == 1), torch.where(data.node_label == 0)
        batch_sig_spear = scipy.stats.spearmanr(avg_attn[sig_index], std_attn[sig_index])[0]
        batch_bkg_spear = scipy.stats.spearmanr(avg_attn[bkg_index], std_attn[bkg_index])[0]
        eval_dict.update({'exp_auc_avg': batch_exp_auc_avg, 'exp_auc_std': batch_exp_auc_std, 'spearman': batch_spearman})

        epoch_avg_attn.append(avg_attn), epoch_std_attn.append(std_attn), epoch_label.append(ex_labels)
        all_spear.append(batch_spearman), sig_spear.append(batch_sig_spear), bkg_spear.append(batch_bkg_spear)

        # print('The spearman correlation between avg and std:', )
        desc = log_epoch(backbone_seed, 1, 'test', loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    # epoch_dict = {eval_metric.name: eval_metric.eval_epoch(return_att=True) for eval_metric in metric_list} if metric_list else {}
    # assert or  metric_list[0].name == 'exp_auc'
    epoch_avg_attn, epoch_std_attn, epoch_label = torch.cat(epoch_avg_attn), torch.cat(epoch_std_attn), torch.cat(epoch_label)
    if method_name in ['subgraphx', 'pgexplainer']:
        save_attn = pd.DataFrame(multi_epoch_attn, columns=method_seeds + ['node_labels', 'graph_id'])
        # indexes =
        save_dir = log_dir / f"bseed_{backbone_seed}_{method_name}_attns.csv"
        save_attn.to_csv(save_dir)

    epoch_dict = {'exp_auc_avg': roc_auc_score(epoch_label, epoch_avg_attn),
                  'exp_auc_std': roc_auc_score(epoch_label, 1-epoch_std_attn),
                  'all_spearman': sum(all_spear) / len(all_spear),
                  'sig_spearman': sum(sig_spear) / len(sig_spear),
                  'bkg_spearman': sum(bkg_spear) / len(bkg_spear)}
                  # 'all_spearman': scipy.stats.spearmanr(epoch_avg_attn, epoch_std_attn)[0]
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    print('-' * 80, '\n', 'epoch_dict:', epoch_dict, '\n', '-' * 80)

    return epoch_dict

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack')
    parser.add_argument('--method', type=str, help='method used')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--mseeds', type=int, nargs="+", help='random seed for training explainer', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--bseed', type=int, nargs="+", help='random seed for training backbone', default=0)
    # parser.add_argument('--mseed', type=int, help='random seed for explainer', default=0)

    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)

    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.method, args.backbone, args.cuda, args.note
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('../configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])

    print('=' * 80)
    print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))

    # the directory is like egnn_actstrack / bseed0 / lri_bern
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
    all_seeds_res = []
    model_dir = Path('../log') / config_name / method_name
    # cuda_id = args.cuda
    # for method_seed in args.mseeds:
    for backbone_seed in args.bseed:
        epoch_dict = test_uncertainty(config, method_name, model_name, backbone_seed, args.mseeds, dataset_name, model_dir, device)
    all_seeds_res += [epoch_dict]
    df = pd.DataFrame(all_seeds_res, index=['_'.join([f'bseed{bseed}', method_name]) for bseed in args.bseed])
    with open(f'result/egnn_synmol/uncertainty/{method_name}.csv', mode='a') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)

    # multi_seeds_res += [avg_report, std_report]
    # print(json.dumps(avg_std_report, indent=4))
    # save_multi_bseeds(multi_seeds_res, method_name, exp_method, args.bseeds)

