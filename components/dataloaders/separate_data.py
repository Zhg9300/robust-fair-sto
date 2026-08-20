import math
import random
from contextlib import contextmanager

import numpy as np


def _split_indices_by_weights(indices, clients, weights):
    indices = np.asarray(indices, dtype=int)
    clients = list(clients)
    if len(clients) == 0:
        return []
    if len(indices) == 0:
        return [(client, indices[:0]) for client in clients]

    weights = np.asarray(weights, dtype=float)
    if weights.size != len(clients) or np.sum(weights) <= 0:
        weights = np.ones(len(clients), dtype=float)
    proportions = weights / np.sum(weights)
    raw_counts = proportions * len(indices)
    counts = np.floor(raw_counts).astype(int)
    remainder = len(indices) - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for pos in order[:remainder]:
            counts[pos] += 1

    results = []
    start = 0
    for client, count in zip(clients, counts):
        end = start + int(count)
        results.append((client, indices[start:end]))
        start = end
    return results


def _build_partition_map(dataset_label, num_clients, target_class_num,
                         item_classes_num, min_client_samples, alpha, niid=False,
                         balance=False, partition=None, train_prob=None):
    dataset_label = np.asarray(dataset_label)
    all_indices = np.arange(len(dataset_label))
    dataidx_map = {client: np.array([], dtype=int) for client in range(num_clients)}

    if not niid:
        partition = "pat"
        item_classes_num = target_class_num

    if partition == "pat":
        if item_classes_num is None or item_classes_num <= 0:
            raise ValueError("item_classes_num must be positive for pat partition.")
        idx_for_each_class = [
            all_indices[dataset_label == label]
            for label in range(target_class_num)
        ]

        class_num_per_client = [item_classes_num for _ in range(num_clients)]
        min_required_clients = math.ceil(target_class_num / item_classes_num)
        if num_clients < min_required_clients:
            print(
                f"Warning: client count ({num_clients}) is too small for pathological "
                "partition. Falling back to IID allocation."
            )
            idxs = np.array(all_indices, dtype=int)
            np.random.shuffle(idxs)
            num_items = len(idxs) // num_clients
            for client in range(num_clients):
                start = client * num_items
                end = len(idxs) if client == num_clients - 1 else (client + 1) * num_items
                dataidx_map[client] = idxs[start:end]
            return dataidx_map

        least_samples = float(min_client_samples)

        max_clients_per_class = max(
            1,
            int(num_clients / target_class_num * item_classes_num),
        )
        for label in range(target_class_num):
            selected_clients = [
                client for client in range(num_clients)
                if class_num_per_client[client] > 0
            ][:max_clients_per_class]
            num_selected_clients = len(selected_clients)
            if num_selected_clients == 0:
                continue

            class_indices = idx_for_each_class[label]
            num_all_samples = len(class_indices)
            num_per = num_all_samples / num_selected_clients
            if balance or num_selected_clients == 1:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                # Each client receives ``item_classes_num`` class fragments;
                # enforce the configured per-client lower bound across those
                # fragments instead of coupling it to the dataset class count.
                lower = int(max(num_per / 10, least_samples / item_classes_num))
                upper = int(num_per)
                if upper <= lower:
                    num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
                else:
                    num_samples = np.random.randint(
                        lower,
                        upper,
                        num_selected_clients - 1,
                    ).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                dataidx_map[client] = np.append(
                    dataidx_map[client],
                    class_indices[idx:idx + num_sample],
                    axis=0,
                ).astype(int)
                idx += num_sample
                class_num_per_client[client] -= 1
        return dataidx_map

    if partition == "dir":
        min_size = 0
        K = target_class_num
        N = len(dataset_label)

        while min_size < target_class_num:
            idx_batch = [[] for _ in range(num_clients)]
            for label in range(K):
                idx_k = np.where(dataset_label == label)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                if proportions.sum() <= 0:
                    proportions = np.ones(num_clients) / num_clients
                else:
                    proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for client in range(num_clients):
            np.random.shuffle(idx_batch[client])
            dataidx_map[client] = np.asarray(idx_batch[client], dtype=int)
        return dataidx_map

    raise NotImplementedError


def _mirror_test_partition_map(train_label, test_label, trainidx_map,
                               num_clients, target_class_num):
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    testidx_map = {client: np.array([], dtype=int) for client in range(num_clients)}

    for label in range(target_class_num):
        test_indices = np.where(test_label == label)[0]
        np.random.shuffle(test_indices)

        clients = []
        weights = []
        for client in range(num_clients):
            train_indices = np.asarray(trainidx_map.get(client, []), dtype=int)
            count = int(np.sum(train_label[train_indices] == label)) if len(train_indices) > 0 else 0
            if count > 0:
                clients.append(client)
                weights.append(count)

        if len(clients) == 0:
            clients = list(range(num_clients))
            weights = [1] * num_clients

        for client, split_indices in _split_indices_by_weights(test_indices, clients, weights):
            testidx_map[client] = np.append(testidx_map[client], split_indices, axis=0).astype(int)

    for client in range(num_clients):
        np.random.shuffle(testidx_map[client])
    return testidx_map


@contextmanager
def _temporary_partition_seed(seed):
    """Build a deterministic partition without leaking global RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_partition_maps(train_labels, test_labels, num_clients,
                         target_class_num, item_classes_num=None, alpha=None,
                         niid=True, balance=False, partition=None,
                         min_client_samples=0, partition_seed=1):
    """Build reproducible client index maps without materializing client data."""
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    total_num = len(train_labels) + len(test_labels)
    train_prob = len(train_labels) / total_num if total_num else 1.0
    with _temporary_partition_seed(int(partition_seed)):
        train_map = _build_partition_map(
            train_labels,
            num_clients,
            target_class_num,
            item_classes_num,
            min_client_samples,
            alpha,
            niid=niid,
            balance=balance,
            partition=partition,
            train_prob=train_prob,
        )
        test_map = _mirror_test_partition_map(
            train_labels, test_labels, train_map, num_clients, target_class_num,
        )
    return train_map, test_map
