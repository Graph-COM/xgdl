import yaml
import json
import re
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from eval import FidelEvaluation, AUCEvaluation, PrecEvaluation
from get_model import Model
from baselines import *
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, load_checkpoint, ExtractorMLP, get_optimizer, sum_fid_score
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")


def eval_one_batch(baseline, data, epoch, return_attn=True):
    # with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradx', 'inter_grad', 'gnnexplainer']):
    with torch.set_grad_enabled(baseline.name in ['gradx', 'gradcam', 'inter_grad', 'gnnexplainer']):
        baseline.eval()
        # baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        # baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        if return_attn:
            loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)
        else:
            infer_clf_logits = baseline.forward(data) if baseline.name in inherent_models else baseline.clf(data)
            loss_dict, node_attn = {}, None

        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)



def eval_one_epoch(baseline, data_loader, epoch, phase, seed, signal_class, metric_list=None, attn_dir=None):
    return_attn = False if attn_dir else True
    pbar = tqdm(data_loader)
    [eval_metric.reset() for eval_metric in metric_list] if metric_list else None

    # treshhold_list = np.linspace(0.1, 1, 10)
    # for num in range(1, 10):
    #     exec(f"clf_acc_{num} = torchmetrics.Accuracy(task='binary', threshold={0.9+num/100})")

    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary', threshold=0.5), torchmetrics.AUROC(task='binary')
    for idx, data in enumerate(pbar):
        torch.cuda.empty_cache()
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = eval_one_batch(baseline, data.to(baseline.device), epoch, return_attn=return_attn)

        if return_attn is False:
            # if the seed are bigger or equal than 99, the attn_dir will be a list of file names.
            if seed > 99:
                b_seed = seed - 100
                seed_attn_dir = attn_dir[b_seed]
                all_df = pd.read_csv(seed_attn_dir)
                ens_cols = [col for col in all_df.columns if col.isdigit()]
                # ens_cols = [str(i) for i in range(10)]
                ensemble_df = all_df[ens_cols].mean(axis=1)
                attn = torch.tensor(ensemble_df[all_df['batch_idx'] == idx].to_numpy())
            elif seed == 99:
                multi_attn = None
                weight_list = []
                #// pattern = r'\/([^\/]+)\/bs\d+_ms\[\d+\]_attns\.csv'
                #// method = re.findall(pattern, str(attn_dir[0]))[0]
                _, setting, method, *_ = str(attn_dir[0]).split('/') 

                #~ TODO: add the logic of weighted
                # col can be "exp_auc", "ACCfidpos"
                def add_weight_list(cur_seed: int, setting, col='ACCfidpos'):
                    #* find the target df
                    regex = r"^attribution.*erm$"
                    root_path = Path("result", setting, "post-inherent/ensenmle_exp_auc_prec@20_fid+pos_fid-pos/")
                    target_dir_list = [dir for dir in root_path.iterdir() if dir.is_dir() and re.search(regex, dir.name)]
                    # print(target_dir_list)
                    target_dir = target_dir_list[0]
                    target_file = [file for file in target_dir.iterdir() if file.is_file() and f"bseed{cur_seed}" in file.name][0]
                    temp = pd.read_csv(target_file, index_col=0)

                    #! Important selection
                    result_df = temp.loc[temp['exp_method']==method]
                    if len(result_df) == 1:
                        weight_list.append(float(result_df[col]))
                    elif len(result_df) > 10:
                        row = result_df.index.str.contains(f'erm_{cur_seed}')
                        weight_list.append(float(result_df[row][col]))
                    else:
                        print(target_file)
                        print(temp)
                        print(method)
                        raise NotImplementedError(f"Unknown df length in 1-10, df: {result_df}")
                    return
                #* The number of digits in the name that indicates the number of explainer seed. Only used for subgraphx.
                pre_num = len(re.findall(r'\d+', str(attn_dir[0]))) - 1 if method == 'subgraphx' else 10
                for i in range(pre_num):
                    seed_attn_dir = attn_dir[i]
                    # seed_attn_dir = str(attn_dir).replace(f'bs{b_seed}', f'bs{i}')
                    seed_df = pd.read_csv(seed_attn_dir)
                    name = 'attn' if 'attn' in seed_df else str(i)
                    #! Error: Can not convert numpy.object_
                    seed_attn = torch.tensor(seed_df[name][seed_df['batch_idx'] == idx].to_numpy()).unsqueeze(1)
                    multi_attn = seed_attn if multi_attn is None else torch.cat([multi_attn, seed_attn], dim=1)
                    
                    #! Remember to uncomment it
                    add_weight_list(i, setting=setting)

                #! This is for testing the new Agg Strategy
                weights = torch.tensor(weight_list, dtype=torch.float64).view(-1, 1)
                if (weights > 0).any():
                    weights = torch.clamp(weights, min=0)
                else:
                    weights = torch.nn.functional.softmax(weights, dim=0)
                # weights = (weights + 1) / 2 
                # normalized_w = torch.nn.functional.softmax(weights, dim=0)
                if True:
                    attn = torch.matmul(multi_attn, weights) / weights.sum()                
                else:
                    attn = torch.mean(multi_attn, dim=1) ## shape == (N, 10) 

            else:
                seed_df = pd.read_csv(attn_dir)
                name = 'attn' if 'attn' in seed_df else str(seed)
                attn = torch.tensor(seed_df[name][seed_df['batch_idx'] == idx].to_numpy())


        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
        eval_dict = {metric.name: metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
                     for metric in metric_list}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})

        # acc_dict = {}
        # for num in range(1, 10):
        #     exec(f"acc_dict['clf_acc_{num}'] = clf_acc_{num}(clf_logits, clf_labels)")
        # print('='*80, acc_dict, '='*80)

        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc

    epoch_dict = {eval_metric.name: eval_metric.eval_epoch() for eval_metric in metric_list} if metric_list else {}
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    sum_fid_score(epoch_dict)
    # log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    return epoch_dict


def test(config, method_name, exp_method, model_name, backbone_seed, method_seed, loaders, dataset, config_dir, device, metric_str, csv_dir):
    set_seed(backbone_seed, method_name=method_name, backbone_name=model_name)
    writer = None
    model_dir = config_dir / method_name
    log_dir = config_dir / "post-inherent"
    log_dir.mkdir(parents=True, exist_ok=True)
    print('The logging directory is', log_dir), print('=' * 80)

    batch_size = config['optimizer']['batch_size']
    if exp_method == 'gnnlrp' and dataset == 'plbind':
        batch_size = 1
    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, batch_size, data_config, dataset_seed=0, num_workers=0, device=device)
    signal_class = dataset.signal_class

    clf = Model(model_name, config[model_name],method_name, config[method_name], dataset).to(device)
    backbone_config = config[model_name]
    method_config = config[method_name]

    extractor = ASAPooling(backbone_config['hidden_size'], ratio=method_config['casual_ratio'], dropout=method_config['dropout_p']) \
        if method_name == 'asap' else ExtractorMLP(backbone_config['hidden_size'], method_config, config['data'].get('use_lig_info', False)) \
        if method_name in inherent_models else nn.Identity()
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits
    map_location = torch.device('cpu') if not torch.cuda.is_available() else device
    # establish the clf model and the metrics
    backbone = eval(name_mapping[method_name])(clf, extractor, criterion, config[method_name]) \
        if method_name in inherent_models else clf

    # initialize extractor of pgexplainer
    if exp_method == 'pgexplainer':
        pg_extractor = ExtractorMLP(backbone_config['hidden_size'], config['pgexplainer'], config['data'].get('use_lig_info', False))
        pg_extractor = pg_extractor.to(device)
        model = PGExplainer(backbone, pg_extractor, criterion, config['pgexplainer'])
        if clf_method == 'erm':
            load_checkpoint(model, config_dir / 'pgexplainer', model_name=exp_method, backbone_seed=backbone_seed, seed=method_seed,
                        map_location=map_location) if method_seed < 10 else print(f'NO PGExplainer loaded for ensemble seed {method_seed}.')
        else:
            print(f"Prepare to train PGExplainer for Inherent Model {clf_method}.")

            pass

    # initialize erm or inherent_models from ckpt
    assert load_checkpoint(backbone, model_dir, model_name=method_name, seed=backbone_seed, map_location=map_location) \
        # if exp_method != 'pgexplainer' else True

    if 'label_perturb' in exp_method:
        model = LabelPerturb(backbone, mode=int(exp_method[-1]))
    elif 'self' == exp_method:
        model = backbone
    elif exp_method == 'pgexplainer':
        if clf_method == 'erm':
            pass  # solved in the former part to avoid overwrite the erm although they should be the same
        else:
            print("Train PGExplainer...")

    else:
        model = eval(name_mapping[exp_method])(backbone, criterion, config[exp_method])

    # metric_list = [AUCEvaluation()] + [FidelEvaluation(backbone, i/10) for i in range(2, 9)] + \
    #  [FidelEvaluation(backbone, i/10, instance='pos') for i in range(2, 9)] + \
    metric_list = []
    for one_metric in metric_str:
        if 'exp_auc' == one_metric:
            metric_list += [AUCEvaluation()]
        elif re.compile(r'prec').search(one_metric):
            _, k = one_metric.split("@")
            metric_list += [PrecEvaluation(int(k))]
        elif 'fid+pos' == one_metric:
            metric_list += [FidelEvaluation(backbone, i / 10, instance='pos', type='acc', symbol='+') for i in reversed(range(2, 9))]
        elif 'fid-pos' == one_metric:
            metric_list += [FidelEvaluation(backbone, i / 10, instance='pos', type='acc', symbol='-') for i in reversed(range(2, 9))]
        else:
            print('='*80, f'The unknown metric is set as {one_metric}', '='*80)
            # raise ValueError(f"Metric name {one_metric} besides exp_auc/pos_fid/neg_fid is not expected")

    model.start_tracking() if 'grad' in exp_method or exp_method == 'gnnlrp' else None
    set_seed(method_seed, method_name=method_name, backbone_name=model_name)
    # metric_names = [i.name for i in metric_list] + ['clf_acc', 'clf_auc']
    # train_dict = eval_one_epoch(model, loaders['train'], 1, 'train', backbone_seed, signal_class, writer, metric_list)
    # valid_dict = eval_one_epoch(model, loaders['valid'], 1, 'valid', backbone_seed, signal_class, writer, metric_list)
    # if exp_method == 'pgexplainer' and method_name != 'erm':
    #     optimizer = get_optimizer(clf, extractor, config['optimizer'], method_name, warmup=False)
    #     raise NotImplementedError('PGExplainer for inherent models are not implemented yet.')
    test_dict = eval_one_epoch(model, loaders['test'], 1, 'test', method_seed, signal_class, metric_list, attn_dir=csv_dir)

    if method_seed == 99:
        exp_method += '_mul_ens'
        method_name += '_mul_ens'
    elif method_seed >= 100:
        exp_method += '_one_ens'
        method_name += '_one_ens'
    test_dict.update({'exp_method': exp_method, 'clf_method': method_name})
    return test_dict


def save_multi_methods(multi_methods_res, indexes, seed_info, metric_list, exp_list, mod_list, only_ens=False):
    # save_seeds = seeds +
    # indexes = ['_'.join([method_name, str(seed), item]) for seed in seeds for item in ['avg', 'std']]
    # from itertools import product
    # indexes = ['_'.join(item) for item in product(methods, seeds)]
    df = pd.DataFrame(multi_methods_res, index=indexes)

    #! Remember to delete the "_only_ens"
    suffix = "_only_ens" if only_ens else ""
    day_dir = Path('result') / config_name / 'post-inherent' / '_'.join(metric_list) / ('_'.join(exp_list+mod_list) + suffix)
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join([f'bseed{seed_info}']+exp_list+mod_list) + '.csv')
    with open(csv_dir, mode='w') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    return df

def get_avg_std_report(reports, around=2):
    # print(reports)
    one_report = reports[0]
    str_dicts = {k: one_report[k] for k in one_report if isinstance(one_report[k], str)}
    all_keys = {k: [] for k in one_report if not isinstance(one_report[k], str)}
    for report in reports:
        for k in all_keys:
            all_keys[k].append(report[k])
    avg_report, std_report, avg_std_report = str_dicts.copy(), str_dicts.copy(), str_dicts.copy()
    avg_report.update({k: np.mean(v) for k, v in all_keys.items()})
    std_report.update({k: np.std(v) for k, v in all_keys.items()})
    avg_std_report.update({k: fr'{(100*np.mean(v)):.{around}f} \pm {(100*np.std(v)):.{around}f}' for k, v in all_keys.items()})
    return avg_report, std_report, avg_std_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='actstrack')
    parser.add_argument('--clf_method', type=str, nargs="+", help='classification method used', default=['erm'])
    parser.add_argument('--exp_method', type=str, nargs="+", help='explanation method used', default=['self'])
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--bseed', type=int, help='random seed for training backbone', default=0)
    # parser.add_argument('--mseeds', type=int, nargs="+", help='random seed for explainer', default=list(range(10)))
    parser.add_argument('--metrics', type=str, nargs="+", default=['exp_auc', 'fid+pos', 'fid-pos'])
    parser.add_argument('--ensemble', type=str, default='exp_auc fidelity', choices=['exp_auc', 'fidelity', 'exp_auc fidelity', ''])
    parser.add_argument('--use_csv', action="store_true")
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)


    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio, args.cuda)

    dataset_name, model_name, cuda_id, note = args.dataset, args.backbone, args.cuda, args.note

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')

    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))

    # the directory is like egnn_actstrack / lri_bern
    config_dir = Path('log') / config_name
    multi_method_res, indexes, exp_methods, clf_methods = [], [], ["label_perturb1", "label_perturb0", "self"], []
    #! remember to commet this
    if True:
        exp_methods = []
    #'self', 'label_perturb1', 'label_perturb0',

    backbone_seed = args.bseed

    #! this is explained in paper 
    if dataset_name == "plbind":
        args.metrics.remove('exp_auc')
        args.metrics.insert(0, 'prec@20')
        # print("!!has changed the metric!!")
        # post_hoc_attribution = ['gradcam', 'gradx', 'inter_grad', ]
    for item in args.exp_method:
        exp_methods += post_hoc_attribution if item == 'attribution' else post_hoc_explainers if item == 'explainer'\
                else [item]
    for item in args.clf_method:
        clf_methods += inherent_models if item == 'inherent' else [item] if item in inherent_models else ['erm']

    # if trigger ensemble
    if args.ensemble == 'fidelity':
        assert set(post_hoc_explainers) >= set(exp_methods)

    #! Totally different logic: just attn ensemble
    print('=' * 80)
    print(f"The clf_method: {clf_methods}.")
    print(f"The exp_method: {exp_methods}.")
    print(f"Ensemble: {args.ensemble}.")

    dataset, loaders = args.dataset, {'test': 0}

    for exp_method, clf_method in [(i, j) for i in exp_methods for j in clf_methods]:
        torch.cuda.empty_cache()
        if exp_method == 'self' and clf_method == 'erm':
            continue
        print('=' * 80)
        print(f'Using {exp_method} to explain {clf_method}.')
        mseeds_res = []
        # cuda_id = backbone_seed % 5
        if exp_method in (post_hoc_attribution + ['self']):
            mseeds = [0]
        else: # (exp_method in posthoc_explainer)
            if clf_method in inherent_models:
                mseeds = [backbone_seed]
            else: # clf_method == 'erm'
               mseeds = list(range(10)) 

        # the results of inherent explained by posthoc+self do not use the csv file with stored attn
        # the csv_dir is only used for the post-hoc explaining erm
        use_csv = args.use_csv and (clf_method == 'erm') and ('label_perturb' not in exp_method)
        # out of efficiency, we only compute 3/5 seeds for subgraphx in dataset ActsTrack/SynMol
        if exp_method == 'subgraphx':
            mseeds = list(range(3)) if dataset_name == 'actstrack' else list(range(5))

        #! remember to comment/uncomment
        # for method_seed in mseeds:
        #     csv_dir = config_dir / exp_method / f"bs{backbone_seed}_ms{mseeds}_attns.csv" if use_csv else None
        #     test_report = test(config, clf_method, exp_method, model_name, backbone_seed, method_seed, loaders, dataset, config_dir, device, metric_str=args.metrics, csv_dir=csv_dir)

        #     indexes += [f'{exp_method}_{clf_method}_{method_seed}']
        #     mseeds_res += [test_report] if method_seed < 10 else []

        # avg_report, std_report, avg_std_report = get_avg_std_report(mseeds_res)

        # mseeds_res += [avg_report, avg_std_report] if len(mseeds) > 1 else []
        # indexes += [f'{exp_method}_{clf_method}_avg', f'{exp_method}_{clf_method}_std'] if len(mseeds) > 1 else []
        # multi_method_res += mseeds_res

        # print(json.dumps(avg_std_report, indent=4))
    #! Remember to uncomment them and set only_ens
    only_ens = True

    # for ensemble to add some big seed as indicator
    # 99 indicate the multi model ensemble; 100+ indicates the one model ensemble and the last digit is the backbone seed
    if_ensemble = 'ensenmle' if args.ensemble else ''
    print('=' * 80, '\nBeginning Ensemble!\n', '=' * 80) if if_ensemble else print('Done!')
    for exp_method, clf_method in [(i, j) for i in exp_methods for j in clf_methods]:
        ens_seeds = []
        if exp_method == 'self' and clf_method in inherent_models:
            ens_seeds += [99] if 'exp_auc' in args.ensemble else []
            csv_dirs = [config_dir / clf_method / f"bs{bs}_ms{[bs]}_attns.csv" for bs in range(10)]
        elif exp_method in post_hoc_attribution and clf_method == 'erm':
            ens_seeds += [99] if 'exp_auc' in args.ensemble else []
            csv_dirs = [config_dir / exp_method / f"bs{bs}_ms[0]_attns.csv" for bs in range(10)]
        elif exp_method == 'subgraphx' and clf_method == 'erm':
            ens_seeds += [99] if 'exp_auc' in args.ensemble else []
            ens_seeds += [100 + backbone_seed] if 'fidelity' in args.ensemble else []
            seed_list = list(range(5)) if dataset_name == 'synmol' else list(range(3))
            csv_dirs = [config_dir / exp_method / f"bs{bs}_ms{seed_list}_attns.csv" for bs in range(10)]
        elif exp_method in post_hoc_explainers and clf_method == 'erm':
            ens_seeds += [99] if 'exp_auc' in args.ensemble else []
            ens_seeds += [100+backbone_seed] if 'fidelity' in args.ensemble else []
            csv_dirs = [config_dir / exp_method / f"bs{bs}_ms{list(range(10))}_attns.csv" for bs in range(10)]
        else:
            print(f"No Processing for Explanation Method {exp_method} and Classification Method {clf_method}.")
            continue
        for ens_seed in ens_seeds:
            test_report = test(config, clf_method, exp_method, model_name, backbone_seed, ens_seed, loaders, dataset,
                               config_dir, device, metric_str=args.metrics, csv_dir=csv_dirs)
            multi_method_res += [test_report]
            post_fix = 'multi_ens' if ens_seed == 99 else 'one_ens'
            indexes += [f'{exp_method}_{clf_method}_{post_fix}']

    result = save_multi_methods(multi_method_res, indexes, args.bseed, [if_ensemble]+args.metrics, args.exp_method, args.clf_method, only_ens=only_ens)
    print(result)