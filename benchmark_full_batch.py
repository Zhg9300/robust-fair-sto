"""Benchmark strict full-client gradients at several physical batch sizes."""

import argparse
import gc
import sys
import time

import torch

from components.main import initialize, read_params


def _default_params():
    original_argv = sys.argv
    try:
        sys.argv = ['run.py']
        return read_params()
    finally:
        sys.argv = original_argv


def _parse_sizes(value):
    sizes = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not sizes or any(size < 0 for size in sizes):
        raise argparse.ArgumentTypeError('sizes must be comma-separated non-negative integers')
    return sizes


def benchmark(device, sizes, data_device, repeats):
    rows = []
    for micro_batch_size in sizes:
        params = _default_params()
        params.update({
            'seed': 1,
            'device': device,
            'module': 'MLP',
            'algorithm': 'FedAvg',
            'dataloader': 'DataLoader_fashion_dir',
            'Diralpha': 0.5,
            'N': 10,
            'C': 1.0,
            'B': 'full',
            'micro_batch_size': micro_batch_size,
            'data_device': data_device,
            'R': 1,
            'E': 1,
            'weight_decay': 0.0,
        })
        _, algorithm = initialize(params)
        client = max(
            algorithm.client_list,
            key=lambda candidate: candidate.local_training_number,
        )
        timings = []
        if algorithm.device.type == 'cuda':
            torch.cuda.synchronize(algorithm.device)
            torch.cuda.reset_peak_memory_stats(algorithm.device)
        for _ in range(repeats):
            start = time.perf_counter()
            client.cal_gradient_loss()
            if algorithm.device.type == 'cuda':
                torch.cuda.synchronize(algorithm.device)
            timings.append(time.perf_counter() - start)
        peak_mib = None
        if algorithm.device.type == 'cuda':
            peak_mib = torch.cuda.max_memory_allocated(algorithm.device) / (1024 ** 2)
        rows.append({
            'micro_batch_size': micro_batch_size,
            'client_samples': client.local_training_number,
            'physical_batches': client.physical_training_batch_num,
            'mean_seconds': sum(timings) / len(timings),
            'peak_cuda_mib': peak_mib,
        })
        del algorithm, client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sizes', type=_parse_sizes, default=_parse_sizes('0,1024,2048'))
    parser.add_argument('--data_device', choices=('model', 'cpu'), default='model')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('repeats must be positive')

    print('micro_batch_size,client_samples,physical_batches,mean_seconds,peak_cuda_mib')
    for row in benchmark(args.device, args.sizes, args.data_device, args.repeats):
        peak = 'n/a' if row['peak_cuda_mib'] is None else f"{row['peak_cuda_mib']:.2f}"
        print(
            f"{row['micro_batch_size']},{row['client_samples']},"
            f"{row['physical_batches']},{row['mean_seconds']:.6f},{peak}"
        )


if __name__ == '__main__':
    main()
