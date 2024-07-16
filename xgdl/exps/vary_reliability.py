import os
pre_cuda_id = int(os.environ.get('CUDA_VISIBLE_DEVICES')) if os.environ.get('CUDA_VISIBLE_DEVICES') else None
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import yaml
import json
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from itertools import product
import nni
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from eval import FidelEvaluation, LabelFidelity, AUCEvaluation
from get_model import Model
from baselines import *
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, update_and_save_best_epoch_res, load_checkpoint, save_checkpoint, ExtractorMLP, get_optimizer, sum_fid_score
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
from statistics import mean
import warnings
warnings.filterwarnings("ignore")


def eval_one_batch(baseline, data, epoch, phase):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradx', 'inter_grad', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # BernMaskP
        do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=do_sampling)

        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)


def test_one_epoch(baseline, data_loader, epoch, phase, seed, signal_class, writer=None, metric_list=None):
    use_tqdm = True
    run_one_batch = eval_one_batch
    pbar, avg_loss_dict = tqdm(data_loader) if use_tqdm else data_loader, dict()
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    save_epoch_attn = []
    for idx, data in enumerate(pbar):
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = run_one_batch(baseline, data.to(baseline.device), epoch, phase)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()


        eval_dict = {metric.name: metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
                     for metric in metric_list} if metric_list else {}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        batch_fid = [eval_dict[k] for k in eval_dict if 'fid' in k and 'all' in k]
        eval_dict.update({'fid_all': mean(batch_fid)}) if phase in ['valid', 'test'] and batch_fid else {}


        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict)
        pbar.set_description(desc) if use_tqdm else None
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')

    epoch_dict = {eval_metric.name: eval_metric.eval_epoch() for eval_metric in metric_list} if metric_list else {}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    sum_fid_score(epoch_dict)

    log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    return epoch_dict


def test(config, method_name, model_name, backbone_seed, seed, dataset_name, parent_dir, device, quick=False, load_dir=None):
    # writer = SummaryWriter(log_dir) if log_dir is not None else None
    writer = None
    model_dir, log_dir = (parent_dir / 'erm', None) if model_name in post_hoc_attribution \
        else (parent_dir / 'erm', parent_dir / method_name)
    # log_dir = parent_dir / method_name if method_name in inherent_models else None
    # log_dir.mkdir(parents=True, exist_ok=True) if log_dir is not None else None
    # model_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config['optimizer']['batch_size']
    model_cofig = config[method_name] if method_name in inherent_models else config['erm']
    warmup = model_cofig['warmup']
    # epochs = config[method_name]['epochs']
    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, dataset_seed=0, num_workers=0, device=device)
    signal_class = dataset.signal_class

    clf = Model(model_name, config[model_name],  # backbone_config
                method_name, config['erm'],  # method_config
                dataset).to(device)
    extractor = ExtractorMLP(config[model_name]['hidden_size'], config[method_name], config['data'].get('use_lig_info', False)) \
        if method_name in inherent_models + ['pgexplainer'] else nn.Identity()
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits


    # establish the model and the metrics
    if 'label_perturb' in method_name:
        baseline = LabelPerturb(clf, mode=int(method_name[-1]))
    else:
        assert method_name in post_hoc_attribution + post_hoc_explainers
        constructor = eval(name_mapping[method_name])
        baseline = constructor(clf, criterion, config[method_name]) if method_name != 'pgexplainer' else PGExplainer(clf, extractor, criterion, config['pgexplainer'])
    # assert load_checkpoint(baseline.clf, model_dir, model_name='erm', seed=backbone_seed, map_location=)
    map_location = torch.device('cpu') if not torch.cuda.is_available() else device
    checkpoint = torch.load(load_dir, map_location=map_location)
    baseline.clf.load_state_dict(checkpoint['model_state_dict'])

        # save_checkpoint(baseline.clf, model_dir, model_name='erm', backbone_seed=backbone_seed, seed=backbone_seed)
    metric_list = [AUCEvaluation()] + [FidelEvaluation(baseline.clf, i / 10, instance='pos', symbol='+') for i in range(2, 9)] + \
              [FidelEvaluation(baseline.clf, i/10, instance='pos', symbol='-') for i in range(2, 9)]
              # [FidelEvaluation(baseline.clf, i/10, instance='neg') for i in range(2, 9)] if quick==False else \
              # [AUCEvaluation()] + [FidelEvaluation(baseline.clf, i/10) for i in range(2, 9)]
    baseline.start_tracking() if 'grad' in method_name or method_name == 'gnnlrp' else None

    set_seed(seed, method_name=method_name, backbone_name=model_name)
    metric_names = [a + b for a, b in product(['valid_', 'test_'], [i.name for i in metric_list]+['clf_acc', 'clf_auc', 'fid_all', 'fid_pos', 'fid_neg'])]
    # metric_names = [j+i.name for i in metric_list for j in ['valid_', 'test_']]
    metric_dict = {}.fromkeys(metric_names, -1)
    # assert epochs == 1 # methods only one epoch
    test_dict = test_one_epoch(baseline, loaders['test'], 1, 'test', seed,
                                              signal_class,  writer, metric_list)
        # metric_dict, new_best = update_and_save_best_epoch_res(baseline, metric_dict, valid_dict, test_dict, epoch, log_dir, backbone_seed, seed, writer, method_name, main_metric)

    return test_dict


def test_one_seed(args, load_dir):
    print(args)
    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.method, args.backbone, args.cuda, args.note
    method_seed, backbone_seed = args.seed, args.bseed
    set_seed(backbone_seed, method_name=method_name, backbone_name=model_name)
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    # print('=' * 80)
    # print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))

    cuda_ = f'cuda' if cuda_id == 99 else f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu'
    device = torch.device(cuda_)
    # log_dir = None
    # if config['logging']['tensorboard'] or method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
    main_dir = Path('log') / config_name
    main_dir.mkdir(parents=True, exist_ok=True)
        # shutil.copy(config_path, log_dir / config_path.name)
    report_dict = test(config, method_name, model_name, backbone_seed, method_seed, dataset_name, main_dir, device, quick=args.quick, load_dir=load_dir)
    # want_keys = [a + b for a in ['valid_', 'test_'] for b in ['fid_all', 'fid_pos', 'fid_neg', 'exp_auc', 'clf_auc', 'clf_acc']]
    # sub_report = {k: v for k, v in report_dict.items() if k in want_keys}
    # print(sub_report)
    return report_dict


def main(args):
    # if args.gpu_ratio is not None:
    #     torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)
    report_dict, indexes = [], []

    def findAllFile(base, restrict=None):
        res = []
        for root, ds, fs in os.walk(base):
            for f in fs:
                if not restrict or restrict in f:
                    fullname = os.path.join(root, f)
                    res += [fullname]
        return res

    def findSubFile(base, restrict=None):
        res = []
        for f in os.listdir(base):
            if not restrict or restrict in f:
                fullname = os.path.join(base, f)
                res += [fullname]
        return res

    config_name = '_'.join([args.backbone, args.dataset])
    for model_path in findSubFile(Path('log') / config_name / 'erm', restrict='pt'):
        # args.method = method
        path_list = re.split(r'[\./-]', model_path) # /erm11.pt
        erm_name = path_list[-2] # like erm20
        print(f'The model to load is {erm_name}.')
        report_dict += [test_one_seed(args, load_dir=model_path)]
        indexes += [erm_name]
    # print(json.dumps(report_dict, indent=4))

    df_path = Path('result') / config_name / 'model_reliability'
    df_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report_dict, index=indexes)
    df.to_csv(df_path / f'{args.method}.csv')
    # nni_report = dict(report_dict, **{'default': report_dict[f'valid_{main_metric}']})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack_2T')
    parser.add_argument('-m', '--method', type=str, help='method used', default='')
    # parser.add_argument('--methods', type=str, nargs='+', help='methods used', default='')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)
    parser.add_argument('--bseed', type=int, help='random seed for training backbone', default=0)
    parser.add_argument('--quick', action="store_true", help='ignore some evaluation')
    parser.add_argument('--no_tqdm', action="store_true", help='disable the tqdm')

    exp_args = parser.parse_args()
    use_tqdm = False if exp_args.no_tqdm else True
    # sub_metric = 'avg_loss'
    main(exp_args)
