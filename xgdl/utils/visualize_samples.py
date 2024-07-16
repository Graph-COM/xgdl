import yaml
import json
import argparse
import uuid
from tqdm import tqdm
from pathlib import Path
import torch
import networkx as nx
import torch.nn as nn
from torch.nn import functional as F
from eval import FidelEvaluation, AUCEvaluation
from get_model import Model
from baselines import LabelPerturb, VGIB, LRIBern, LRIGaussian, CIGA
from utils import to_cpu, log_epoch, get_data_loaders, set_seed, load_checkpoint, ExtractorMLP, get_optimizer
from utils import inherent_models, post_hoc_explainers, post_hoc_attribution, name_mapping
import torchmetrics
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, to_networkx
from rdkit import Chem
from torch_geometric.nn import knn_graph
import numpy as np
import pandas as pd
import warnings
from eval import control_sparsity
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def node_attn_to_edge_attn(node_attn, edge_index):
    src_attn = node_attn[edge_index[0]]
    dst_attn = node_attn[edge_index[1]]
    edge_attn = src_attn * dst_attn
    return edge_attn


def eval_one_sample(baseline, data, epoch):
    with torch.set_grad_enabled(baseline.name in ['gradcam', 'gradgeo', 'gnnexplainer']):
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # do_sampling = True if phase == 'valid' and baseline.name == 'pgexplainer' else False # we find this is better for BernMaskP
        loss, loss_dict, infer_clf_logits, node_attn = baseline.forward_pass(data, epoch=epoch, do_sampling=False)
        # edge_attn = node_attn_to_edge_attn(node_attn, data.edge_index) if baseline.name != 'lri_gaussian' else None

        return loss_dict, to_cpu(infer_clf_logits), to_cpu(node_attn)  #, to_cpu(edge_attn)

def get_viz_idx(test_set, dataset_name, num_viz_samples):
    y_dist = test_set.data.y.numpy().reshape(-1)
    num_nodes = np.array([each.x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name == 'Graph-SST2':
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
            candidate_set = np.nonzero(condi)[0]
        else:
            candidate_set = np.nonzero(y_dist == each_class)[0]
        idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
        res.append((idx, tag))

    if dataset_name == 'synmol':
        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
            res.append((idx, tag))
    return res

def visualize_results(gsat, test_set, num_viz_samples, dataset_name):
    all_viz_set = get_viz_idx(test_set, num_viz_samples, dataset_name)
    figsize = 10
    fig, axes = plt.subplots(len(all_viz_set), num_viz_samples, figsize=(figsize*num_viz_samples, figsize*len(all_viz_set)*0.8))

    for class_idx, (idx, tag) in enumerate(all_viz_set):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        batch_att, _, _ = eval_one_batch(gsat, data.to(gsat.device), epoch=500)

        # iterate each graph
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if dataset_name == 'synmol':
                ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']
                node_dict = dict(zip(list(range(len(ATOM_TYPES))), ATOM_TYPES))
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            else:
                raise NotImplementedError

            # select one graph
            node_subset = data.batch == i
            _, edge_mask = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            visualize_a_graph(viz_set[i].edge_index, edge_mask, node_label, node_type, axes[class_idx, i], norm=True, mol_type=mol_type, coor=coor)
            # axes[class_idx, i].axis('off')
        fig.tight_layout()

    each_plot_len = 1/len(viz_set)
    for num in range(1, len(viz_set)):
        line = plt.Line2D((each_plot_len*num, each_plot_len*num), (0, 1), color="gray", linewidth=1, linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)

    each_plot_width = 1/len(all_viz_set)
    for num in range(1, len(all_viz_set)):
        line = plt.Line2D((0, 1), (each_plot_width*num, each_plot_width*num), color="gray", linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)


def visualize_a_graph(edge_index, edge_att, node_label, node_type, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    # if dataset_name == 'synmol':
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']
    node_dict = dict(zip(list(range(len(ATOM_TYPES))), ATOM_TYPES))
    mol_type = {k: node_dict[v.item()] for k, v in enumerate(node_type)}

    if norm:  # for better visualization
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

def get_edges_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles.values[0])

    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # add edges in both directions
            edges_list.append((i, j))
            edges_list.append((j, i))
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
    return torch.tensor(edge_index)

def eval_one_epoch(baseline, data_loader, epoch, phase, seed, signal_class, writer=None, eval_metric=None, metric_list=None):
    view_mode = 0

    pbar, avg_loss_dict = tqdm(data_loader), dict()
    clf_ACC, clf_AUC = torchmetrics.Accuracy(task='binary'), torchmetrics.AUROC(task='binary')
    mis, neg, pos, masked_perf_drop = 0, 0, 0, 0
    mol_df = pd.read_csv('../data/synmol/raw' + '/logic8_smiles.csv')
    for idx, data in enumerate(pbar):
        # data = negative_augmentation(data, data_config, phase, data_loader, idx, loader_len)
        loss_dict, clf_logits, attn = eval_one_sample(baseline, data.to(baseline.device), epoch)
        ex_labels, clf_labels, data = to_cpu(data.node_label), to_cpu(data.y), data.cpu()
        label, pred = int(data.y.item()), int(clf_logits.sigmoid() > 0.5)
        neg += 1 if label == 0 else 0
        pos += 1 if label == 1 else 0
        if label == 1:
            pass
        #     fig, ax = plt.subplots(figsize=(10, 8))
        #     edge_index = get_edges_from_smiles(mol_df.iloc[data.mol_df_idx]['smiles'])
        #     edge_attn = node_attn_to_edge_attn(data.node_label, edge_index)
        #     visualize_a_graph(edge_index, edge_attn, data.node_label, data.x, ax, coor=None, norm=False,
        #                       mol_type=None, nodesize=300)
        #     plt.show()
        #     place = 0
        if label == view_mode:
            [metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric') for metric in metric_list]
            masked_perf_drop = eval_metric.collect_batch(ex_labels, attn, data, signal_class, 'geometric')
            if masked_perf_drop == 1 - view_mode:  # still_pos (view_mode=1 and masked_perf_drop=0) or mis_neg
                mis += 1
                fig, ax = plt.subplots(figsize=(10, 8))
                sparse_attn = control_sparsity(attn, sparsity=eval_metric.sparsity)
                edge_index = get_edges_from_smiles(mol_df.iloc[data.mol_df_idx]['smiles'])
                # edge_index = knn_graph(data.pos, k=2, batch=data.batch, loop=False)
                edge_attn = node_attn_to_edge_attn(sparse_attn, edge_index) if baseline.name != 'lri_gaussian' else None
                visualize_a_graph(edge_index, 1 - edge_attn, data.node_label, data.x, ax, coor=None, norm=False, mol_type=None, nodesize=300)
                # fig.tight_layout()
                masked_pred = 1 - pred if masked_perf_drop == 1 else pred
                plt.title(f'label: {label}, pred: {pred}, masked_pred: {masked_pred}, sparsity: {eval_metric.sparsity}')
                # uid = uuid.uuid1()
                plt.savefig(f'../../img/erm/' + str(data.mol_df_idx.item()) + '.png')
                # plt.show()
        # print()
        eval_dict = {}
        eval_dict.update({'clf_acc': clf_ACC(clf_logits, clf_labels), 'clf_auc': clf_AUC(clf_logits, clf_labels)})
        # eval_dict.update({'mean_fid': mean([eval_dict[k] for k in eval_dict if 'fid' in k])})

        desc = log_epoch(seed, epoch, phase, loss_dict, eval_dict, phase_info='dataset')
        pbar.set_description(desc)
        # compute the avg_loss for the epoch desc
        exec('for k, v in loss_dict.items():\n\tavg_loss_dict[k]=(avg_loss_dict.get(k, 0) * idx + v) / (idx + 1)')
    # print(f"mis_neg/neg/all: {mis}/{neg}/{len(pbar)}")
    epoch_dict = {eval_metric.name: eval_metric.eval_epoch()}
    epoch_dict.update({metric.name: metric.eval_epoch() for metric in metric_list})
    epoch_dict.update({'clf_acc': clf_ACC.compute().item(), 'clf_auc': clf_AUC.compute().item()})
    # epoch_dict.update({'mean_fid': mean([epoch_dict[k] for k in epoch_dict if 'fid' in k])})
    # log_epoch(seed, epoch, phase, avg_loss_dict, epoch_dict, writer)

    return epoch_dict

def test(config, method_name, exp_method, model_name, backbone_seed, dataset_name, log_dir, device):
    set_seed(backbone_seed, method_name=method_name, backbone_name=model_name)
    writer = None
    print('The logging directory is', log_dir), print('=' * 80)

    data_config = config['data']
    loaders, test_set, dataset = get_data_loaders(dataset_name, 1, data_config, dataset_seed=0)
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
    assert load_checkpoint(backbone, log_dir, model_name=method_name, seed=backbone_seed, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    if 'label_perturb' in exp_method:
        model = LabelPerturb(backbone, mode=int(exp_method[-1]))
    else:
        model = eval(name_mapping[exp_method])(clf, criterion, config[exp_method])

    # metric_list = [FidelEvaluation(backbone, i/10) for i in range(2, 9)]
    main_metric = FidelEvaluation(backbone, 0.8)
    metric_list = [FidelEvaluation(backbone, i/10) for i in range(2, 9)]
    print('Use random explanation and fidelity w/ signal nodes to test the Model Sensitivity.')
    # test_set =
    # DataLoader(test_set, batch_size=1, shuffle=False)
    train_dataset = loaders['train']  # [:100]
    valid_dataset = loaders['valid']  # [:100]
    test_dataset = loaders['test']
    # metric_names = [i.name for i in metric_list] + ['clf_acc', 'clf_auc']
    # train_dict = eval_one_epoch(model, train_dataset, 1, 'train', backbone_seed, signal_class, writer, metric_list)
    # valid_dict = eval_one_epoch(model, valid_dataset, 1, 'valid', backbone_seed, signal_class, writer, metric_list)
    test_dict = eval_one_epoch(model, test_dataset, 1, 'test', backbone_seed, signal_class,  writer, main_metric, metric_list)

    return test_dict

def save_multi_bseeds(multi_methods_res, method_name, exp_method, seeds):
    seeds += ['avg', 'std']
    indexes = [method_name+'_'+str(seed) for seed in seeds]
    # from itertools import product
    # indexes = ['_'.join(item) for item in product(methods, seeds)]
    df = pd.DataFrame(multi_methods_res, index=indexes)

    day_dir = Path('result') / config_name / datetime.now().strftime("%m_%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = day_dir / ('_'.join([exp_method, method_name, 'sensitivity.csv']))
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
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='synmol')
    parser.add_argument('--clf_method', type=str, help='method used', default='erm', choices=inherent_models+['erm'])
    parser.add_argument('--exp_method', type=str, help='method used', default='label_perturb1', choices=['label_perturb1', 'label_perturb0'])
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='egnn')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=-1)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    parser.add_argument('--bseeds', type=int, nargs="+", help='random seed for training backbone', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # parser.add_argument('--mseed', type=int, help='random seed for explainer', default=0)

    args = parser.parse_args()
    # sub_metric = 'avg_loss'
    print(args)

    dataset_name, method_name, model_name, cuda_id, note = args.dataset, args.clf_method, args.backbone, args.cuda, args.note
    exp_method = args.exp_method
    config_name = '_'.join([model_name, dataset_name])
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') / f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])

    print('=' * 80)
    print(f'Config for {method_name}: ', json.dumps(config[method_name], indent=4))



    # the directory is like egnn_actstrack / bseed0 / lri_bern
    multi_seeds_res = []
    for backbone_seed in args.bseeds:
        device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
        model_dir = Path('log') / config_name / method_name
        test_report = test(config, method_name, exp_method, model_name, backbone_seed, dataset_name, model_dir, device)
        multi_seeds_res += [test_report]
        # print('Train Dataset Result: ', json.dumps(train_report, indent=4))
        # print('Valid Dataset Result: ', json.dumps(valid_report, indent=4))
        print('Test Dataset Result: ', json.dumps(test_report, indent=4))
    avg_report, std_report, avg_std_report = get_avg_std_report(multi_seeds_res)
    multi_seeds_res += [avg_report, std_report]
    print(json.dumps(avg_std_report, indent=4))
    save_multi_bseeds(multi_seeds_res, method_name, exp_method, args.bseeds)

