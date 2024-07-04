import os
import sys
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import numpy as np
from Bio.PDB import ShrakeRupley
from statistics import mean

inherent_models = ['asap', 'lri_bern', 'lri_gaussian', 'vgib', 'ciga']
post_hoc_explainers = ['pgexplainer', 'gnnexplainer', 'subgraphx', 'pgmexplainer']
post_hoc_attribution = ['gradcam', 'gnnlrp', 'gradx', 'inter_grad']

name_mapping = {'lri_bern': 'LRIBern', 'lri_gaussian': 'LRIGaussian', 'vgib': 'VGIB', 'ciga': 'CIGA', 'dir': 'DIR',
                'gradcam': 'GradCAM', 'inter_grad': 'InterGrad', 'gradx': 'GradX', 'gnnlrp': 'GNNLRP',
                'pgexplainer': 'PGExplainer', 'gnnexplainer': 'GNNExplainer', 'subgraphx': 'SubgraphX',
                'pgmexplainer': 'PGMExplainer',
                'test': 'Test', 'asap': 'ASAP'}


sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                  n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

def sum_fid_score(all_dict):
    sub_dict = {k: v for k, v in all_dict.items() if '@' in k}
    fidel_type = set([i.split('@')[0] for i in sub_dict.keys()])
    for prefix in fidel_type:
        fid_list = [all_dict[k] for k in all_dict if prefix in k]
        all_dict[prefix] = mean(fid_list)
    for prefix in fidel_type:
        match_fix = prefix.replace('+', '-')
        if '+' in prefix and match_fix in fidel_type:
            name = prefix.replace('+', '')
            all_dict[name] = all_dict[prefix] - all_dict[match_fix]


def set_seed(seed, method_name=None, backbone_name=None):
    # from pytorch_lightning import seed_everything
    # # set the random seeds.
    # seed_everything(seed)
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if method_name not in ['asap'] and backbone_name not in ['pointtrans']:
        torch.use_deterministic_algorithms(True)
    else:
        
        torch.use_deterministic_algorithms(False, warn_only=True)

def to_cpu(tensor):
    return tensor.detach().cpu() if tensor is not None else None


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def pmap_multi(pickleable_fn, data, n_jobs, verbose=1, desc=None, **kwargs):
  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None, )(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data), desc=desc)
  )

  return results


def get_random_idx_split(dataset_len, split, seed):
    np.random.seed(seed)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(dataset_len)
    np.random.shuffle(idx)

    n_train, n_valid = int(split['train'] * len(idx)), int(split['valid'] * len(idx))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train+n_valid]
    test_idx = idx[n_train+n_valid:]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
