import json
import argparse
import os.path
from datetime import datetime
import numpy as np
import torch
from trainer import run_one_seed
import pandas as pd
import warnings
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution
from pathlib import Path
warnings.filterwarnings("ignore")


def get_avg_std_report(reports):
    # print(reports)
    all_keys = {k: [] for k in reports[0]}
    for report in reports:
        for k in report:
            all_keys[k].append(report[k])
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    std_report = {k: np.std(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.3f} \pm {np.std(v):.3f}' for k, v in all_keys.items()}
    return avg_report, std_report, avg_std_report


def save_multi_result(method_res, method_name, all_methods, bseed, seeds, config_name):
    seeds = [str(i) for i in seeds]
    aux_names = seeds + ['avg', 'std'] if len(seeds) > 1 else seeds
    indexes = [method_name+'_'+str(name) for name in aux_names]

    df = pd.DataFrame(method_res, index=indexes)

    day_dir = Path('result') / config_name / '_'.join(all_methods)
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join(['bs'+str(bseed)] + all_methods + seeds) + '.csv')
    with open(csv_dir, mode='a') as f:
        df.to_csv(f, lineterminator="\n", header=f.tell() == 0)
    now_df = pd.read_csv(csv_dir, index_col=0)
    return now_df


def main(args):
    initial_method_str = args.methods
    if len(args.methods) == 1:
        if args.methods[0] == 'all_inherent':
            args.methods = inherent_models
        elif args.methods[0] == 'all_attribution':
            args.methods = post_hoc_attribution
        elif args.methods[0] == 'all_explainer':
            args.methods = post_hoc_explainers # 'pgmexplainer',  post_hoc_explainers
        elif args.methods[0] == 'explainer_inherent':
            args.methods = ['gnnexplainer_lri_gaussian', 'gnnexplainer_lri_bern', 'pgexplainer_lri_gaussian', 'pgexplainer_lri_bern']
        else:
            print(args.methods)
            if args.methods[0] in inherent_models + post_hoc_attribution + post_hoc_explainers:
                pass
            else:
                raise ValueError(f'Unknown experiment {args.methods[0]} except for explainer/attribution/inherent.')

    #* set method seeds
    if args.seeds == -1:
        if set(args.methods) <= set(inherent_models):
            args.seeds = [args.bseed]
        elif set(args.methods) <= set(post_hoc_attribution):
            args.seeds = [0]
        elif set(args.methods) <= set(post_hoc_explainers):
            args.seeds = list(range(0, 10))
        elif any('pgexplainer_' in element or 'gnnexplainer_' in element for element in args.methods):
            args.seeds = [args.bseed]
        else:
            raise ValueError(f"Don't try to running methods of different kinds together. {set(args.methods)}")
    else:
        print(f'You have set seed {args.seeds} for {args.methods}.')

    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio, args.cuda)
    multi_methods_res = []
    config_name = '_'.join([args.backbone, args.dataset])

    #* method preparation
    for method in args.methods:
        args.method = method
        multi_seeds_res, multi_seeds_attn = [], None
        save_dir = Path('log') / config_name / method / f"bs{args.bseed}_ms{args.seeds}_attns.csv"

        for seed in args.seeds:
            args.seed = seed
            report_dict, attn_df = run_one_seed(args, None)
            if multi_seeds_attn is None:
                multi_seeds_attn = attn_df
            else:
                # four columns are: node_labels, graph_labels, batch_idx, graph_idx
                assert multi_seeds_attn.iloc[:, -4:].equals(attn_df.iloc[:, -4:])
                multi_seeds_attn.insert(seed-args.seeds[0], seed, attn_df[seed])
            multi_seeds_res += [report_dict]
            # print(json.dumps(report_dict, indent=4))

        if not os.path.exists(save_dir) or args.rewrite:
            multi_seeds_attn.to_csv(save_dir)
        elif args.methods == inherent_models or args.methods == post_hoc_attribution or args.methods == post_hoc_explainers:
            print('='*80, f'Notice: You have rewrite the attns.csv for all_inherent/attribution/explainer.')
        else:
            raise FileExistsError(f'{save_dir} exists and you do not set rewrite.')

        avg_report, std_report, avg_std_report = get_avg_std_report(multi_seeds_res)
        multi_seeds_res += [avg_report, std_report] if len(args.seeds) > 1 else []
        print(f'[{len(args.seeds)} seeds for {method} on {args.dataset} (classifier training seed: {args.bseed}) done]\n',
              json.dumps(avg_std_report, indent=4))
        multi_methods_df = save_multi_result(multi_seeds_res, method, initial_method_str, args.bseed, args.seeds, config_name)

    print("=" * 80), print(multi_methods_df)


if __name__ == '__main__':
    time = datetime.now().strftime("%m_%d-%H_%M")
    parser = argparse.ArgumentParser(description='EXP')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--method', type=str, help='method used')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--methods', type=str, help='methods tested', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', help='random seed', default=-1)

    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)
    parser.add_argument('--bseed', type=int, help='random seed for training backbone', default=0)
    parser.add_argument('--quick', action="store_true", help='ignore some evaluation')
    parser.add_argument('--save', action="store_true", help='save all erm models')
    parser.add_argument('--no_tqdm', action="store_true", help='disable the tqdm')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--rewrite', action="store_true", help='rewrite the attns')
    parser.add_argument('--only_clf', action="store_true", help='train erm models and then exit')


    args = parser.parse_args()
    use_tqdm = False if args.no_tqdm else True
    # main_metric = 'exp_auc'
    # sub_metric = 'avg_loss'
    main(args)
