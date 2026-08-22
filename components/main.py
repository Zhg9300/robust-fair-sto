
import components as cn
import numpy as np
import argparse
import torch
import sys
from components.param_utils import parse_batch_size, parse_bool, parse_min_client_samples


def _evaluation_client_ids(alg):
    client_ids = [client.id for client in alg.client_list]
    controller = getattr(alg, 'attack_controller', None)
    excluded_ids = set(getattr(alg, 'evaluation_excluded_ids', []))
    if not excluded_ids and controller is not None and getattr(controller, 'enabled', False):
        excluded_ids = set(getattr(controller, 'byzantine_ids', []))
    unknown_ids = excluded_ids.difference(client_ids)
    if unknown_ids:
        raise RuntimeError(f"Evaluation client ids are outside the client pool: {sorted(unknown_ids)}")

    evaluation_ids = [client_id for client_id in client_ids if client_id not in excluded_ids]
    if not evaluation_ids:
        raise RuntimeError("No honest clients available for evaluation metrics.")
    return evaluation_ids, sorted(excluded_ids)


def _evaluation_metric_histories(alg):
    metric_histories = alg.comm_log['client_metric_history']
    if len(metric_histories) != len(alg.client_list):
        raise RuntimeError("client_metric_history length does not match client_list length.")

    evaluation_ids, excluded_ids = _evaluation_client_ids(alg)
    evaluation_id_set = set(evaluation_ids)
    histories = [
        metric_history
        for client, metric_history in zip(alg.client_list, metric_histories)
        if client.id in evaluation_id_set
    ]
    return evaluation_ids, excluded_ids, histories


def outFunc(alg):
    evaluation_ids, excluded_ids, metric_histories = _evaluation_metric_histories(alg)

    loss_list = np.array([
        metric_history['test_loss'][-1]
        for metric_history in metric_histories
        if metric_history['test_loss'][-1] is not None
    ])
    has_accuracy = all(
        metric_history.get('test_accuracy')
        for metric_history in metric_histories
    )
    local_acc_list = np.array([
        metric_history['test_accuracy'][-1]
        for metric_history in metric_histories
    ]) if has_accuracy else np.array([])
    print(str(alg.params))

    stream_log = ""
    stream_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round) + '\n'
    stream_log += f'Mean Global Test loss: {format(np.mean(loss_list), ".3f")}' + \
        '\n' if len(loss_list) > 0 else ''
    stream_log += 'Global model test: \n'
    if excluded_ids:
        stream_log += f'Evaluation Client IDs: {evaluation_ids}\n'
        stream_log += f'Excluded Evaluation Client IDs: {excluded_ids}\n'
    stream_log += f'Test Loss List: {[f"{x:.6f}" for x in loss_list]}\n'
    if len(loss_list) > 0:
        stream_log += (
            f'Loss Average: {np.mean(loss_list):.6f}. '
            f'Loss Variance: {np.var(loss_list):.6f}. '
            f'Loss Min: {np.min(loss_list):.6f}. '
            f'Loss Max: {np.max(loss_list):.6f}\n'
        )
    if has_accuracy:
        stream_log += f'Test Acc List: {[f"{x:.3f}" for x in local_acc_list]}\n'
        stream_log += f'Average: {format(np.mean(local_acc_list), ".3f")}. Variance: {format(np.var(local_acc_list), ".3f")}. Min: {format(np.min(local_acc_list), ".3f")}. Max: {format(np.max(local_acc_list), ".3f")}' + '\n'
    theorem_metrics = getattr(alg.data_loader, 'theorem_metrics', None)
    if callable(theorem_metrics):
        values = theorem_metrics(alg.module)
        stream_log += f'Honest Loss Variance V_H(w): {np.var(loss_list):.9f}\n'
        stream_log += (
            'Honest Average-Loss Gap F_H(w)-F_H(w_H*): '
            f'{values["honest_average_loss_gap"]:.9f}\n'
        )
        stream_log += f'Distance ||w-v||_2: {values["distance_to_v"]:.9f}\n'
    stream_log += '\n'
    alg.stream_log = stream_log + alg.stream_log
    print(stream_log)


def read_params():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--seed', help='seed', type=int, default=1)

    parser.add_argument(
        '--device', help='device: -1, 0, 1, or ...', type=int, default=0)

    parser.add_argument('--module', help='module name;',
                        type=str, default='CNN')

    parser.add_argument('--algorithm', help='algorithm name;',
                        type=str, default='FedAvg')

    parser.add_argument('--dataloader', help='dataloader name;',
                        type=str, default='DataLoader_cifar10_pat')

    parser.add_argument(
        '--B', help="logical batch size: positive integer or 'full'", type=parse_batch_size, default=50)
    parser.add_argument(
        '--micro_batch_size', help='physical batch size used to accumulate a full-client update; 0 disables chunking',
        type=int, default=0)
    parser.add_argument(
        '--data_device', help="where federated data resides: 'model' or 'cpu'",
        choices=('model', 'cpu'), default='model')
    parser.add_argument(
        '--partition_seed', help='random seed used only for client data partitioning',
        type=int, default=1)
    parser.add_argument(
        '--min_client_samples',
        help="pathological unbalanced-partition minimum: 'auto' (10%% of mean) or a non-negative integer",
        type=parse_min_client_samples, default='auto')

    parser.add_argument('--linear_noise_std',
                        help='V.A linear-regression noise standard deviation',
                        type=float, default=1e-3)
    parser.add_argument('--linear_delta',
                        help='V.A exceptional-worker parameter separation',
                        type=float, default=1.0)
    parser.add_argument('--linear_train_batches',
                        help='number of generated V.A training batches per worker',
                        type=int, default=100)

    parser.add_argument('--NC', help='client_class_num', type=int, default=1)

    parser.add_argument(
        '--balance', help='balance or not for pathological separation', type=str, default='True')

    parser.add_argument(
        '--Diralpha', help='alpha parameter for dirichlet', type=float, default=0.1)

    parser.add_argument('--N', help='client num', type=int, default=10)

    parser.add_argument(
        '--C', help='select client proportion', type=float, default=1.0)

    parser.add_argument('--R', help='communication round',
                        type=int, default=2000)

    parser.add_argument('--E', help='local epochs', type=int, default=1)

    parser.add_argument('--test_interval',
                        help='test interval', type=int, default=50)

    parser.add_argument('--test_conflicts',
                        help='test conflicts', type=str, default='False')

    parser.add_argument('--sgd_step', help='sgd training',
                        type=str, default='False')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--decay', help='learning rate decay',
                        type=float, default=0.999)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.0)
    parser.add_argument(
        '--weight_decay',
        help='SGD weight decay coefficient; use 0 for paper-aligned runs',
        type=float,
        default=5e-4)

    parser.add_argument(
        '--cc_tau',
        help='Centered-Clipping L2 clipping radius',
        type=float,
        default=10.0)
    parser.add_argument(
        '--cc_iterations',
        help='Centered-Clipping inner iterations per communication round',
        type=int,
        default=1)
    parser.add_argument(
        '--median_max_iterations',
        help='maximum geometric-median iterations',
        type=int,
        default=100)
    parser.add_argument(
        '--aggregator_tolerance',
        help='geometric-median convergence tolerance',
        type=float,
        default=1e-6)
    parser.add_argument('--gradient_aggregator',
                        help='gradient aggregator',
                        choices=('mean', 'cwtm', 'cwm', 'median', 'faba',
                                 'krum', 'mda', 'centered_clipping', 'nbs'),
                        type=str, default='mean')
    parser.add_argument('--gradient_aggregator_f',
                        help='assumed Byzantine count (NBS may use f=beta*m)',
                        type=int, default=None)
    parser.add_argument('--alpha', help='alpha of FedFV',
                        type=float, default=0.1)
    parser.add_argument(
        '--tau', help='parameter tau in FedFV', type=int, default=1)

    parser.add_argument(
        '--lam', help='lambda parameter in AFL', type=float, default=0.8)

    parser.add_argument(
        '--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)
    parser.add_argument(
        '--pow', help='parameter pow in AdaFed', type=float, default=3)
    parser.add_argument('--q', help='parameter q in qFedAvg',
                        type=float, default=0.1)
    parser.add_argument(
        '--qffl_update_rule',
        help='q-FFL update: normalized qFedAvg or direct objective gradient descent',
        choices=('normalized', 'objective_gradient'),
        default='normalized')

    parser.add_argument('--dishonest_num',
                        help='dishonest number', type=int, default=0)
    parser.add_argument('--attack_mode',
                        help='Byzantine attack mode, e.g. adaptive_copying', type=str, default='None')
    parser.add_argument('--byzantine_ids',
                        help='comma-separated or list-style Byzantine client ids', type=str, default='None')
    parser.add_argument('--attack_start_round',
                        help='first communication round to apply attack', type=int, default=1)
    parser.add_argument('--attack_end_round',
                        help='last communication round to apply attack', type=str, default='None')
    parser.add_argument('--attack_seed',
                        help='seed for randomized attacks such as gaussian and label_random_flip', type=int, default=1)
    parser.add_argument('--attack_scale',
                        help='Byzantine attack scale', type=float, default=1.0)
    parser.add_argument('--alie_z',
                        help='ALIE z coefficient; omit to compute it from online client counts', type=float, default=None)
    parser.add_argument('--loss_bias',
                        help='reported loss bias for loss attacks', type=float, default=0.0)
    parser.add_argument('--attack_target_clients',
                        help='target client ids for impersonation attacks', type=str, default='None')
    parser.add_argument('--copy_loss',
                        help='copy target loss in impersonation attacks', type=str, default='False')
    parser.add_argument('--copy_gradient',
                        help='copy target gradient/update in impersonation attacks', type=str, default='False')
    parser.add_argument('--evaluation_excluded_ids',
                        help='client ids excluded from evaluation in attacks and baselines',
                        type=str, default='None')

    try:
        parsed = vars(parser.parse_args())
        return parsed
    except IOError as msg:
        parser.error(str(msg))


def validate_batch_configuration(params, model=None):
    """Normalize and validate logical/physical batch configuration."""
    params['B'] = parse_batch_size(params.get('B', 50))
    params['min_client_samples'] = parse_min_client_samples(
        params.get('min_client_samples', 'auto'))
    micro_batch_size = int(params.get('micro_batch_size', 0))
    if micro_batch_size < 0:
        raise ValueError('micro_batch_size must be non-negative.')
    params['micro_batch_size'] = micro_batch_size
    sgd_step = parse_bool(params.get('sgd_step', False), 'sgd_step')
    if params['B'] == 'full' and sgd_step:
        raise ValueError("B='full' requires sgd_step=False.")
    if params['B'] != 'full' and micro_batch_size:
        raise ValueError("micro_batch_size is only valid when B='full'.")
    if model is not None and micro_batch_size and any(
        isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
        for layer in model.modules()
    ):
        raise ValueError(
            "micro_batch_size cannot be used with BatchNorm in strict full-gradient mode; "
            "use micro_batch_size=0."
        )
    return sgd_step, micro_batch_size


def initialize(params):
    sgd_step, micro_batch_size = validate_batch_configuration(params)

    cn.setup_seed(seed=params['seed'])
    device = torch.device(
        'cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Module = getattr(sys.modules['components'], params['module'])
    module = Module(device)
    Dataloader = getattr(sys.modules['components'], params['dataloader'])
    data_loader = Dataloader(
        params=params, input_require_shape=module.input_require_shape, device=device)
    print(f"Inputs device: {data_loader.device}")

    module.generate_model(data_loader.input_data_shape,
                          data_loader.target_class_num)
    model_initializer = getattr(data_loader, 'initialize_model', None)
    if callable(model_initializer):
        model_initializer(module)
    validate_batch_configuration(params, module.model)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, module.model.parameters(
    )), lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    criterion_builder = getattr(data_loader, 'build_criterion', None)
    criterion = (
        criterion_builder()
        if callable(criterion_builder)
        else torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    )
    metric_builder = getattr(data_loader, 'build_metrics', None)
    metric_list = metric_builder() if callable(metric_builder) else [cn.Correct()]

    train_setting = {'criterion': criterion,
                     'optimizer': optimizer, 'lr_decay': params['decay'],
                     'sgd_step': sgd_step,
                     'micro_batch_size': micro_batch_size,
                     'momentum': params['momentum']}
    test_interval = params['test_interval']
    test_conflicts = parse_bool(params['test_conflicts'], 'test_conflicts')
    Algorithm = getattr(sys.modules['components'], params['algorithm'])
    algorithm = Algorithm(data_loader=data_loader,
                          module=module,
                          device=device,
                          train_setting=train_setting,
                          client_num=data_loader.pool_size,
                          online_client_num=int(
                              data_loader.pool_size * params['C']),
                          metric_list=metric_list,
                          max_comm_round=params['R'],
                          epochs=params['E'],
                          outFunc=outFunc,
                          write_log=True,
                          test_conflicts=test_conflicts,
                          params=params,)
    algorithm.test_interval = test_interval
    return data_loader, algorithm


if __name__ == '__main__':
    params = read_params()
    data_loader, algorithm = initialize(params)
    algorithm.run()
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
