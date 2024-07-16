import yaml
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from eval import FidelEvaluation, AUCEvaluation, AucFidelity
from get_model import Model
from baselines import *
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
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradx', 'inter_grad', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)
        # from sklearn.metrics import roc_auc_score
        # roc_auc_score(infer_clf_logits, data.y.cpu())
        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def eval_one_epoch(baseline, data_loader, epoch, phase, seed, signal_class, writer=None, metric_list=None):
    pbar, avg_loss_dict = tqdm(data_loader), dict()
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    for idx, data in enumerate(pbar):
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = eval_one_batch(baseline, data.to(baseline.device), epoch)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
        # print()
        eval_dict = {metric.name: metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
                     for metric in metric_list}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        fidel_list = [eval_dict[k] for k in eval_dict if 'fid' in k]
        eval_dict.update({'mean_fid': mean(fidel_list) if fidel_list else 0})

        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    epoch_dict = {eval_metric.name: eval_metric.eval_epoch() for eval_metric in metric_list} if metric_list else {}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    epoch_fidel = [epoch_dict[k] for k in epoch_dict if 'fid' in k]
    epoch_dict.update({'mean_fid': mean(epoch_fidel) if epoch_fidel else 0})

    # log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    return epoch_dict


def test(config, method_name, exp_method, model_name, backbone_seed, method_seed, dataset_name, model_dir, device, metric_str):
    set_seed(backbone_seed, method_name=method_name, backbone_name=model_name)
    writer = None
    log_dir = model_dir.parent() / "post-inherent"
    log_dir.mkdir(parents=True, exist_ok=True)
    print('The logging directory is', log_dir), print('=' * 80)

    batch_size = config['optimizer']['batch_size']
    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, dataset_seed=0, num_workers=8)
    signal_class = dataset.signal_class

    clf = Model(model_name, config[model_name],  # backbone_config
                method_name, config[method_name],  # method_config
                dataset).to(device)

    extractor = ExtractorMLP(config[model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) \
        if method_name in inherent_models + ['pgexplainer'] else nn.Identity()
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits

    # establish the model and the metrics
    backbone = eval(name_mapping[method_name])(clf, extractor, criterion, config[method_name]) \
        if method_name in inherent_models else clf
    assert load_checkpoint(backbone, model_dir, model_name=method_name, seed=backbone_seed, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    if 'label_perturb' in exp_method:
        model = LabelPerturb(backbone, mode=int(exp_method[-1]))
    elif 'self_exp' == exp_method:
        model = backbone
    elif exp_method == 'pgexplainer':
        model = PGExplainer(backbone, extractor, criterion, config['pgexplainer'])
    else:
        model = eval(name_mapping[exp_method])(clf, criterion, config[exp_method])


    # metric_list = [AUCEvaluation()] + [FidelEvaluation(backbone, i/10) for i in range(2, 9)] + \
    #  [FidelEvaluation(backbone, i/10, instance='pos') for i in range(2, 9)] + \
    metric_list = []
    for one_metric in metric_str:
        if 'exp_auc' == one_metric:
            metric_list += [AUCEvaluation()]
        elif 'pos_fid' == one_metric:
            metric_list += [FidelEvaluation(backbone, i / 10, instance='pos', symbol='+') for i in reversed(range(2, 9))]
        elif 'neg_fid' == one_metric:
            metric_list += [FidelEvaluation(backbone, i / 10, instance='pos', symbol='-') for i in reversed(range(2, 9))]
        else:
            raise ValueError(f"Metric name {one_metric} besides exp_auc/pos_fid/neg_fid is not expected")

    # print('Use random explanation and fidelity w/ signal nodes to test the Model Sensitivity.')
    set_seed(method_seed, method_name=method_name, backbone_name=model_name)
    # metric_names = [i.name for i in metric_list] + ['clf_acc', 'clf_auc']
    # train_dict = eval_one_epoch(model, loaders['train'], 1, 'train', backbone_seed, signal_class, writer, metric_list)
    # valid_dict = eval_one_epoch(model, loaders['valid'], 1, 'valid', backbone_seed, signal_class, writer, metric_list)
    if exp_method == 'pgexplainer' and method_name != 'erm':
        optimizer = get_optimizer(clf, extractor, config['optimizer'], method_name, warmup=False)
        raise NotImplementedError('PGExplainer for inherent models are not implemented yet.')
    #
    else:
        test_dict = eval_one_epoch(model, loaders['test'], 1, 'test', backbone_seed, signal_class,  writer, metric_list)

    return {}, {}, test_dict

def save_multi_bseeds(multi_methods_res, method_name, exp_method, seeds, metric_str, exp_metric_str):
    # save_seeds = seeds +
    indexes = ['_'.join([method_name, str(seed), item]) for seed in seeds  for item in ['avg', 'std']]
    # from itertools import product
    # indexes = ['_'.join(item) for item in product(methods, seeds)]
    df = pd.DataFrame(multi_methods_res, index=indexes)

    day_dir = Path('../result') / config_name / 'sensitivity' / metric_str / datetime.now().strftime("%m_%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join([exp_method, method_name]) + '.csv')
    with open(csv_dir, mode='w') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    return df

def get_avg_std_report(reports):
    # print(reports)
    all_keys = {k: [] for k in reports[0]}
    for report in reports:
        for k in report:
            all_keys[k].append(report[k])
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    std_report = {k: np.std(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.3f} +/- {np.std(v):.3f}' for k, v in all_keys.items()}
    return avg_report, std_report, avg_std_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack')
    parser.add_argument('--clf_method', type=str, nargs="+", help='classification method used', default=['erm'])
    parser.add_argument('--exp_method', type=str, nargs="+", help='explanation method used', default=['label_perturb1', 'label_perturb0'])
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--bseeds', type=int, nargs="+", help='random seed for training backbone', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--mseeds', type=int, nargs="+", help='random seed for explainer', default=list(range(10)))
    parser.add_argument('--metrics', type=str, nargs="+", default=['exp_auc', 'pos_fid', 'neg_fid'])

    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)

    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.clf_method, args.backbone, args.cuda, args.note

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')

    exp_method = args.exp_method
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('../configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])

    print('=' * 80)
    print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))
    # the directory is like egnn_actstrack / bseed0 / lri_bern

    model_dir = Path('../log') / config_name / method_name
    multi_bseeds_res = []
    for backbone_seed in args.bseeds:
        mseeds_res = []
        # cuda_id = backbone_seed % 5
        for method_seed in args.mseeds:
            # method_seed = int(method_seed)
            # print('int or str:', type(method_seed))
            train_report, valid_report, test_report = test(config, method_name, exp_method, model_name, backbone_seed, method_seed, dataset_name, model_dir, device, metric_str=args.metrics)
            mseeds_res += [test_report]
            # print('Train Dataset Result: ', json.dumps(train_report, indent=4))
            # print('Valid Dataset Result: ', json.dumps(valid_report, indent=4))
            # print('Test Dataset Result: ', json.dumps(test_report, indent=4))
        avg_report, std_report, avg_std_report = get_avg_std_report(mseeds_res)
        multi_bseeds_res += [avg_report, std_report]
        print(json.dumps(avg_std_report, indent=4))
    save_multi_bseeds(multi_bseeds_res, method_name, exp_method, args.bseeds, args.metric, args.exp_metric)

