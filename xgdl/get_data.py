from pathlib import Path
import torch
import random
import yaml
import numpy
from torch_geometric.loader import DataLoader
from .datasets import ActsTrack, PLBind, Tau3Mu, SynMol
from torch_geometric.nn import knn_graph, radius_graph
from torch.utils.data import Subset

DATASET_CANDIS = ['tau3mu', 'plbind', 'synmol', 'actstrack']

def syn_transform(data):
    edge_index = knn_graph(data.pos, k=5, batch=data.batch, loop=True)
    data.edge_index = edge_index
    return data

def act_transform(data):
    # pos = data.pos / 2955.5000 * 100
    norm_pos = data.pos.norm(dim=-1, keepdim=True)
    pos = data.pos / norm_pos.clamp(min=1e-6)
    edge_index = knn_graph(pos, k=5, batch=data.batch, loop=True)
    data.edge_index = edge_index
    return data

def tau_transform(data):
    edge_index = radius_graph(data.pos, r=1.0, batch=data.batch, loop=True)
    data.edge_index = edge_index
    return data

def plb_transform(data):
    edge_index = radius_graph(data.pos, r=2.0, batch=data.batch, loop=True)
    data.edge_index = edge_index
    return data


class ScienceDataset:
    _initialized_datasets = {}

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_name(method_name)`."
        )

    @classmethod
    def filter_signal_class(cls, dataset, target_label=torch.tensor([[1]])):
        target_label = target_label.to(dtype=torch.float32)
        indices = [i for i, data in enumerate(dataset) if torch.equal(data.y, target_label)]
        return Subset(dataset, indices)

    @classmethod
    def from_name(cls, name, config=None) -> None:
        if name in cls._initialized_datasets:
            return cls._initialized_datasets[name]

        assert name in DATASET_CANDIS, f"Dataset name {name} is not supported, choose from {DATASET_CANDIS}"
        if config is None:
            cfg_path = Path(__file__).parent / 'configs' / f'{name}.yml'
            config = yaml.safe_load((cfg_path).open('r'))['data']
        data_dir = Path(config['data_dir'])
        if 'actstrack' in name:
            tesla = '2T' if len(name.split('_')) == 1 else name.split('_')[-1]
            dataset = ActsTrack(data_dir / 'actstrack', tesla=tesla, data_config=config, transform=act_transform)
        elif name == 'tau3mu':
            dataset = Tau3Mu(data_dir / 'tau3mu', data_config=config, transform=tau_transform)
        elif name == 'synmol':
            dataset = SynMol(data_dir / 'synmol', data_config=config, transform=syn_transform)
        elif name == 'plbind':
            dataset = PLBind(data_dir / 'plbind', data_config=config, transform=plb_transform, n_jobs=32, debug=False)

        cls._initialized_datasets[name] = dataset
        return dataset

    def get_train_loader(self, batch_size):
        follow_batch = 'x_lig' if self.dataset_name == 'plbind' else None
        train_ds = [self[self.index_split['train']]]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, follow_batch=follow_batch)
        return train_loader

def get_data_loaders(dataset_name, batch_size, data_config, dataset_seed, num_workers, device):
    data_dir = Path(data_config['data_dir'])
    assert dataset_name in ['tau3mu', 'plbind', 'synmol'] or 'acts' in dataset_name

    if 'actstrack' in dataset_name:
        def act_transform(data):
            # pos = data.pos / 2955.5000 * 100
            norm_pos = data.pos.norm(dim=-1, keepdim=True)
            pos = data.pos / norm_pos.clamp(min=1e-6)
            edge_index = knn_graph(pos, k=5, batch=data.batch, loop=True)
            data.edge_index = edge_index
            return data
        tesla = '2T' if len(dataset_name.split('_')) == 1 else dataset_name.split('_')[-1]
        dataset = ActsTrack(data_dir / 'actstrack', tesla=tesla, data_config=data_config, seed=dataset_seed, transform=act_transform, device=device)
        loaders, test_set = get_loaders(batch_size, dataset=dataset, idx_split=dataset.idx_split, num_workers=num_workers)

    elif dataset_name == 'tau3mu':
        def tau_transform(data):
            edge_index = radius_graph(data.pos, r=1.0, batch=data.batch, loop=True)
            data.edge_index = edge_index
            return data
        dataset = Tau3Mu(data_dir / 'tau3mu', data_config=data_config, seed=dataset_seed, transform=tau_transform, device=device)
        loaders, test_set = get_loaders(batch_size, dataset=dataset, idx_split=dataset.idx_split, num_workers=num_workers)

    elif dataset_name == 'synmol':
        def syn_transform(data):
            edge_index = knn_graph(data.pos, k=5, batch=data.batch, loop=True)
            data.edge_index = edge_index
            return data
        dataset = SynMol(data_dir / 'synmol', data_config=data_config, seed=dataset_seed, transform=syn_transform, device=device)
        loaders, test_set = get_loaders(batch_size, dataset=dataset, idx_split=dataset.idx_split, num_workers=num_workers)

    elif dataset_name == 'plbind':
        dataset = PLBind(data_dir / 'plbind', data_config=data_config, device=device, n_jobs=32, debug=False)
        loaders, test_set = get_loaders(batch_size, dataset=dataset, idx_split=dataset.idx_split, dataset_name=dataset_name, num_workers=num_workers)

    return loaders, test_set, dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders(batch_size, dataset, num_workers=8):
    idx_split = dataset.idx_split
    dataset_name = dataset.dataset_name
    #
    # def collate_gpu(batch, device='cpu'):
    #     x, t = torch.utils.data.dataloader.default_collate(batch)
    #     return x.to(device=device), t.to(device=device)
    # from functools import partial
    # collate_certain_gpu = partial(collate_gpu, device=device)

    follow_batch = None if dataset_name != 'plbind' else ['x_lig']
    # train_index = idx_split["train"]#[1335:1400]

    train_loader = DataLoader(dataset[idx_split["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, follow_batch=follow_batch)
    valid_loader = DataLoader(dataset[idx_split["valid"]], batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, follow_batch=follow_batch)
    test_loader = DataLoader(dataset[idx_split["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, follow_batch=follow_batch)

    # test_set = dataset.copy(idx_split["test"])  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
