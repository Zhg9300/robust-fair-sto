import math
import os

import numpy as np
import torch

from components.param_utils import FULL_BATCH, parse_batch_size, parse_min_client_samples
from components.dataloaders.separate_data import build_partition_maps


INDEX_CACHE_VERSION = 1


class IndexedBatchData:
    """A lightweight batch view over one shared feature/label tensor pair."""

    def __init__(self, inputs, targets, indices, batch_size):
        self.inputs = inputs
        self.targets = targets
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=inputs.device)
        self.batch_size = parse_batch_size(batch_size)

    def __len__(self):
        if len(self.indices) == 0:
            return 0
        if self.batch_size == FULL_BATCH:
            return 1
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        yield from self.iter_batches(0)

    def __getitem__(self, position):
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        if self.batch_size == FULL_BATCH:
            start, end = 0, len(self.indices)
        else:
            start = position * self.batch_size
            end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        return (
            torch.index_select(self.inputs, 0, batch_indices),
            torch.index_select(self.targets, 0, batch_indices),
        )

    def iter_batches(self, physical_batch_size=0):
        if len(self.indices) == 0:
            return
        if physical_batch_size:
            chunk_size = int(physical_batch_size)
        elif self.batch_size == FULL_BATCH:
            chunk_size = len(self.indices)
        else:
            chunk_size = self.batch_size
        if chunk_size <= 0:
            raise ValueError('physical batch size must be positive.')
        for start in range(0, len(self.indices), chunk_size):
            batch_indices = self.indices[start:start + chunk_size]
            yield (
                torch.index_select(self.inputs, 0, batch_indices),
                torch.index_select(self.targets, 0, batch_indices),
            )

    def physical_batch_count(self, physical_batch_size=0):
        if len(self.indices) == 0:
            return 0
        if physical_batch_size:
            return math.ceil(len(self.indices) / int(physical_batch_size))
        return len(self)


def resolve_min_client_samples(value, train_count, client_count):
    value = parse_min_client_samples(value)
    if value == 'auto':
        return max(1, math.floor((train_count / client_count) * 0.1))
    return value


def partition_cache_name(dataset_name, partition, pool_size, partition_seed,
                         item_classes_num=None, alpha=None, balance=False,
                         min_client_samples=0):
    fields = [
        dataset_name,
        partition,
        f'N{pool_size}',
        f'seed{partition_seed}',
        f'v{INDEX_CACHE_VERSION}',
    ]
    if alpha is not None:
        fields.append(f'alpha{alpha}')
    if item_classes_num is not None:
        fields.append(f'NC{item_classes_num}')
    if partition == 'pat':
        fields.extend([
            'balanced' if balance else 'unbalanced',
            f'min{min_client_samples}',
        ])
    return '_'.join(str(field) for field in fields) + '.npz'


def _save_index_cache(path, train_map, test_map, pool_size):
    payload = {'version': np.asarray([INDEX_CACHE_VERSION], dtype=np.int64)}
    for client_id in range(pool_size):
        payload[f'train_{client_id}'] = np.asarray(train_map[client_id], dtype=np.int64)
        payload[f'test_{client_id}'] = np.asarray(test_map[client_id], dtype=np.int64)
    np.savez_compressed(path, **payload)


def _load_index_cache(path, pool_size):
    with np.load(path, allow_pickle=False) as cache:
        version = int(cache['version'][0])
        if version != INDEX_CACHE_VERSION:
            raise ValueError(f'Unsupported partition cache version: {version}.')
        train_map = {
            client_id: np.asarray(cache[f'train_{client_id}'], dtype=np.int64)
            for client_id in range(pool_size)
        }
        test_map = {
            client_id: np.asarray(cache[f'test_{client_id}'], dtype=np.int64)
            for client_id in range(pool_size)
        }
    return train_map, test_map


def load_or_create_partition(cache_path, train_labels, test_labels, pool_size,
                             target_class_num, partition_seed, recreate=False,
                             item_classes_num=None, alpha=None, balance=False,
                             partition=None, min_client_samples=0):
    if os.path.exists(cache_path) and not recreate:
        return _load_index_cache(cache_path, pool_size)
    train_map, test_map = build_partition_maps(
        train_labels,
        test_labels,
        pool_size,
        target_class_num,
        item_classes_num=item_classes_num,
        alpha=alpha,
        niid=True,
        balance=balance,
        partition=partition,
        min_client_samples=min_client_samples,
        partition_seed=partition_seed,
    )
    _save_index_cache(cache_path, train_map, test_map, pool_size)
    return train_map, test_map


def label_statistics(labels, index_map, pool_size):
    labels = np.asarray(labels)
    result = [[] for _ in range(pool_size)]
    for client_id in range(pool_size):
        local_labels = labels[np.asarray(index_map[client_id], dtype=np.int64)]
        for label in np.unique(local_labels):
            result[client_id].append((int(label), int(np.sum(local_labels == label))))
    return result


def build_indexed_pool(train_inputs, train_targets, test_inputs, test_targets,
                       train_map, test_map, pool_size, batch_size, data_device,
                       model_device):
    residency_device = model_device if data_device == 'model' else torch.device('cpu')
    train_inputs = torch.as_tensor(train_inputs).float().to(residency_device)
    train_targets = torch.as_tensor(train_targets).long().to(residency_device)
    test_inputs = torch.as_tensor(test_inputs).float().to(residency_device)
    test_targets = torch.as_tensor(test_targets).long().to(residency_device)

    data_pool = []
    for client_id in range(pool_size):
        train_indices = np.asarray(train_map[client_id], dtype=np.int64)
        test_indices = np.asarray(test_map[client_id], dtype=np.int64)
        data_pool.append({
            'local_training_data': IndexedBatchData(
                train_inputs, train_targets, train_indices, batch_size),
            'local_training_number': len(train_indices),
            'data_name': str(client_id),
            'local_test_data': IndexedBatchData(
                test_inputs, test_targets, test_indices, batch_size),
            'local_test_number': len(test_indices),
        })
    return data_pool


def configure_indexed_loader(loader, dataset_name, train_inputs, train_targets,
                             test_inputs, test_targets, pool_size,
                             target_class_num, batch_size, data_device,
                             model_device, partition_seed, cache_folder,
                             recreate=False, item_classes_num=None, alpha=None,
                             balance=False, partition=None, shuffle=True,
                             min_client_samples='auto'):
    """Attach a shared indexed pool and reproducible partition metadata."""
    batch_size = parse_batch_size(batch_size)
    resolved_minimum = resolve_min_client_samples(
        min_client_samples, len(train_targets), pool_size,
    ) if partition == 'pat' else 0
    cache_name = partition_cache_name(
        dataset_name,
        partition,
        pool_size,
        partition_seed,
        item_classes_num=item_classes_num,
        alpha=alpha,
        balance=balance,
        min_client_samples=resolved_minimum,
    )
    cache_path = os.path.join(cache_folder, cache_name)
    train_map, test_map = load_or_create_partition(
        cache_path,
        torch.as_tensor(train_targets).cpu().numpy(),
        torch.as_tensor(test_targets).cpu().numpy(),
        pool_size,
        target_class_num,
        partition_seed,
        recreate=recreate,
        item_classes_num=item_classes_num,
        alpha=alpha,
        balance=balance,
        partition=partition,
        min_client_samples=resolved_minimum,
    )
    if shuffle:
        order_rng = np.random.default_rng(int(partition_seed))
        train_map = {
            client_id: order_rng.permutation(indices)
            for client_id, indices in train_map.items()
        }
        test_map = {
            client_id: order_rng.permutation(indices)
            for client_id, indices in test_map.items()
        }
    loader.partition_cache_path = cache_path
    loader.partition_seed = int(partition_seed)
    loader.min_client_samples = resolved_minimum
    loader.statistic = label_statistics(
        torch.as_tensor(train_targets).cpu().numpy(), train_map, pool_size)
    loader.test_statistic = label_statistics(
        torch.as_tensor(test_targets).cpu().numpy(), test_map, pool_size)
    loader.data_pool = build_indexed_pool(
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        train_map,
        test_map,
        pool_size,
        batch_size,
        data_device,
        model_device,
    )
