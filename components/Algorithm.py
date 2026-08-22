
import components as cn
import os
import numpy as np
import torch
import csv
import json
from components.aggregator import build_aggregator
from components.attacks import ByzantineAttackController
from components.param_utils import build_optimizer_like, parse_id_list

EPS = 1e-12


class Algorithm:

    def __init__(self,
                 name='Algorithm',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 test_conflicts=False,
                 params=None):
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [
                cn.Client(i, module, device, train_setting, metric_list)
                for i in range(client_num)
            ]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')
        if online_client_num is None:
            online_client_num = client_num

        choose_client_indices = list(np.random.choice(
            client_num, online_client_num, replace=False))
        self.online_client_list = [client_list[i]
                                   for i in choose_client_indices]
        if client_num > online_client_num:
            print(choose_client_indices)
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        if max_comm_round is None:
            max_comm_round = 10**5
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.module = module
        self.train_setting = train_setting
        self.client_num = client_num
        self.client_list = client_list
        self.online_client_num = online_client_num
        self.max_comm_round = max_comm_round
        self.epochs = epochs
        self.save_name = save_name
        self.outFunc = outFunc
        self.current_comm_round = 0
        self.module.model.to(device)
        self.metric_list = metric_list
        self.write_log = write_log
        self.params = params or {}
        self.test_conflicts = test_conflicts
        self.save_folder = ''

        self.stream_log = ""
        self.comm_log = {'client_metric_history': []}

        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr

        self.optimizer = build_optimizer_like(
            train_setting['optimizer'],
            filter(lambda p: p.requires_grad, self.module.model.parameters()),
            lr=self.lr,
        )

        self.descent_log = []

        self.test_interval = 1

        self.attack_controller = ByzantineAttackController(
            params=self.params,
            client_num=client_num,
            device=device,
        )
        self.evaluation_excluded_ids = parse_id_list(
            self.params.get('evaluation_excluded_ids')
        )
        if len(self.evaluation_excluded_ids) != len(set(self.evaluation_excluded_ids)):
            raise ValueError('evaluation_excluded_ids must not contain duplicates.')
        if any(client_id < 0 or client_id >= client_num
               for client_id in self.evaluation_excluded_ids):
            raise ValueError('evaluation_excluded_ids contains an id outside the client pool.')
        if len(self.evaluation_excluded_ids) == client_num:
            raise ValueError('evaluation_excluded_ids must leave at least one evaluation client.')
        self._configure_aggregators()
        self._pending_attack_log_rows = []
        self._attack_event_id = 0
        self._attack_log_initialized = False

    @staticmethod
    def _parse_aggregator_count(value, name):
        if value is None:
            return None
        value = int(value)
        if value < 0:
            raise ValueError(f'{name} must be non-negative.')
        return value

    def _configure_aggregators(self):
        gradient_name = self.params.get('gradient_aggregator', 'mean')
        self.gradient_aggregator_f = self._parse_aggregator_count(
            self.params.get('gradient_aggregator_f'),
            'gradient_aggregator_f',
        )
        options = {
            'cc_tau': self.params.get('cc_tau', 10.0),
            'cc_iterations': self.params.get('cc_iterations', 1),
            'median_max_iterations': self.params.get('median_max_iterations', 100),
            'aggregator_tolerance': self.params.get('aggregator_tolerance', 1e-6),
        }
        self.gradient_aggregator = build_aggregator(gradient_name, **options)

        suffixes = []
        if 'weight_decay' in self.params:
            suffixes.append(f"wd{float(self.params['weight_decay']):g}")
        if self.gradient_aggregator.name != 'mean':
            suffixes.append(self._aggregator_suffix(
                'gradagg', self.gradient_aggregator, self.gradient_aggregator_f
            ))
        if self.evaluation_excluded_ids:
            ids = '-'.join(str(client_id) for client_id in self.evaluation_excluded_ids)
            suffixes.append(f'evalexclude_{ids}')
        if suffixes:
            self.save_name += ' ' + ' '.join(suffixes)

    def _aggregator_suffix(self, prefix, aggregator, configured_f):
        suffix = f'{prefix}_{aggregator.name}'
        if aggregator.name in {'cwtm', 'faba', 'krum', 'mda', 'nbs'}:
            suffix += f'_f{"auto" if configured_f is None else configured_f}'
        elif aggregator.name == 'centered_clipping':
            suffix += f'_tau{aggregator.clipping_radius}_iter{aggregator.iterations}'
        elif aggregator.name == 'median':
            suffix += f'_iter{aggregator.max_iterations}_tol{aggregator.tolerance:g}'
        return suffix

    def _resolved_byzantine_count(self, configured_f):
        if configured_f is not None:
            return configured_f
        return sum(
            int(self.attack_controller.is_byzantine(client.id))
            for client in self.online_client_list
        )

    def aggregate_gradients(self, gradients, weights=None):
        return self.gradient_aggregator.aggregate(
            gradients,
            weights=weights,
            byzantine_count=self._resolved_byzantine_count(
                self.gradient_aggregator_f
            ),
        )

    def run(self):
        raise RuntimeError(
            'error in Algorithm: This function must be rewritten in the child class.')

    @staticmethod
    def update_learning_rate(optimizer, lr):
        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self):
        self.lr = self.initial_lr * \
            self.train_setting['lr_decay']**self.current_comm_round
        self.update_learning_rate(self.optimizer, self.lr)

    def update_module(self, module, optimizer, lr, g):
        self.update_learning_rate(optimizer, lr)
        optimizer.zero_grad()
        for i, p in enumerate(module.model.parameters()):
            p.grad = g[module.Loc_reshape_list[i]].detach().clone()
        optimizer.step()

    def terminated(self):

        self.flush_attack_log()
        self.adjust_learning_rate()
        if self.current_comm_round > 0 and self.current_comm_round % self.test_interval == 0:
            self.test_and_save_log()

        if self.current_comm_round >= self.max_comm_round:
            if self.current_comm_round % self.test_interval != 0:
                self.test_and_save_log()
            return True
        else:
            if self.online_client_num < self.client_num:
                choose_client_indices = list(np.random.choice(
                    self.client_num, self.online_client_num, replace=False))
                self.online_client_list = [self.client_list[i]
                                           for i in choose_client_indices]
            self.current_comm_round += 1
            return False

    def attack_log_dir(self):
        save_folder = self.save_folder if self.save_folder else ''
        return os.path.join(save_folder, 'attacks', self.save_name)

    def attack_log_path(self):
        return os.path.join(self.attack_log_dir(), 'byzantine_attack_log.csv')

    def _queue_attack_log_rows(self, rows):
        if not rows:
            return
        if self._pending_attack_log_rows:
            pending_round = self._pending_attack_log_rows[-1]['round']
            if pending_round != rows[0]['round']:
                self.flush_attack_log()
        for row in rows:
            row['event_id'] = self._attack_event_id
            self._attack_event_id += 1
        self._pending_attack_log_rows.extend(rows)

    def flush_attack_log(self):
        if not self._pending_attack_log_rows:
            return
        os.makedirs(self.attack_log_dir(), exist_ok=True)
        path = self.attack_log_path()
        exists = os.path.exists(path)
        mode = 'a' if self._attack_log_initialized else 'w'
        with open(path, mode, newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=ByzantineAttackController.LOG_FIELDS,
            )
            if mode == 'w' or not exists:
                writer.writeheader()
            writer.writerows(self._pending_attack_log_rows)
        self._attack_log_initialized = True
        self._pending_attack_log_rows = []

    def record_attack_effective_metrics(self, target_client_list=None,
                                        effective_update_norms=None,
                                        effective_weights=None,
                                        report_path='train'):
        if not self._pending_attack_log_rows:
            return
        if target_client_list is None:
            target_client_list = self.online_client_list
        client_ids = [client.id for client in target_client_list]
        update_norms = self._to_metric_list(effective_update_norms)
        weights = self._to_metric_list(effective_weights)
        update_norm_by_id = {}
        weight_by_id = {}
        if update_norms is not None:
            update_norm_by_id = {client_id: update_norms[idx] for idx, client_id in enumerate(client_ids)}
        if weights is not None:
            weight_by_id = {client_id: weights[idx] for idx, client_id in enumerate(client_ids)}
        for row in self._pending_attack_log_rows:
            if row['round'] != self.current_comm_round:
                continue
            if row.get('path') != report_path:
                continue
            client_id = row['client_id']
            if client_id in update_norm_by_id:
                row['effective_update_norm'] = update_norm_by_id[client_id]
            if client_id in weight_by_id:
                row['effective_weight'] = weight_by_id[client_id]

    @staticmethod
    def _to_metric_list(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().reshape(-1).tolist()
        return [float(value) for value in values]

    def _apply_attack_reports(self, reports, path, old_model_params=None):
        attacked_reports, log_rows = self.attack_controller.apply(
            round_id=self.current_comm_round,
            reports=reports,
            path=path,
            old_model_params=old_model_params,
            lr=self.lr,
        )
        self._queue_attack_log_rows(log_rows)
        return attacked_reports

    def _label_mapping_for(self, client_id):
        return self.attack_controller.label_mapping_for(
            client_id,
            self.current_comm_round,
        )

    def client_update(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        with torch.no_grad():
            for client in target_client_list:
                for server_parameter, client_parameter in zip(
                    self.module.model.parameters(), client.module.model.parameters()
                ):
                    if client_parameter.requires_grad:
                        client_parameter.copy_(server_parameter)

    def weight_aggregate(self, m_locals, weights=None, update_module=True):
        if weights is None:
            weights = torch.Tensor(self.get_client_attr(
                'local_training_number')).float().to(self.device)
        denominator = torch.sum(weights)
        if not torch.isfinite(denominator) or float(torch.abs(denominator).detach().cpu().item()) <= EPS:
            weights = torch.ones_like(weights) / len(weights)
        else:
            weights = weights / denominator
        params_mat = torch.stack([m_local for m_local in m_locals])
        aggregate_params = weights @ params_mat

        if update_module:
            self.module.reshape_vec_to_model_params(aggregate_params)

    def evaluate(self, target_client_list=None, gradient_transform=None,
                 use_full_loss=False):
        if target_client_list is None:
            target_client_list = self.online_client_list
        if gradient_transform is not None and not callable(gradient_transform):
            raise TypeError('gradient_transform must be callable or None.')
        reports = []
        for client in target_client_list:
            true_gradient, loss = client.evaluate_gradient(
                self._label_mapping_for(client.id),
                use_full_loss=use_full_loss,
            )
            if gradient_transform is not None:
                true_gradient = gradient_transform(true_gradient, loss)
            reports.append({
                'client_id': client.id,
                'true_loss': loss,
                'true_gradient': true_gradient,
                'true_model': None,
            })

        reports = self._apply_attack_reports(reports, path='evaluate')
        g_locals = [report['reported_gradient'] for report in reports]
        l_locals = [report['reported_loss'] for report in reports]
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return g_locals, l_locals


    def train(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        old_model_params = self.module.span_model_params_to_vec()
        reports = []
        for client in target_client_list:
            true_model, loss = client.train_local(
                self.epochs, self.lr, self._label_mapping_for(client.id)
            )
            reports.append({
                'client_id': client.id,
                'true_loss': loss,
                'true_gradient': (old_model_params - true_model) / self.lr,
                'true_model': true_model,
            })
        reports = self._apply_attack_reports(reports, path='train', old_model_params=old_model_params)
        m_locals = [report['reported_model'] for report in reports]
        l_locals = [report['reported_loss'] for report in reports]
        return m_locals, l_locals

    def test(self):
        self.comm_log['client_metric_history'] = [
            client.test() for client in self.client_list
        ]

    def test_and_save_log(self):
        """Evaluate, format the current result, and then persist it."""
        self.test()
        if callable(self.outFunc):
            self.outFunc(self)
        if self.write_log:
            self.save_log()

    def save_log(self):

        save_dict = {'algorithm name': self.name}
        save_dict['info'] = 'data loader name_' + self.data_loader.name + '_module name_' + self.module.name + '_train setting_' + \
            str(self.train_setting) + '_client num_' + str(self.client_num) + \
            '_max comm round_' + str(self.max_comm_round) + \
            '_epochs_' + str(self.epochs)
        save_dict['communication round'] = self.current_comm_round
        save_dict['test interval'] = self.test_interval
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        file_name = self.save_folder + self.save_name + '.json'
        fileObject = open(file_name, 'w')
        fileObject.write(json.dumps(save_dict))
        fileObject.close()
        file_name = self.save_folder + 'log_' + self.save_name + '.log'
        fileObject = open(file_name, 'w')
        fileObject.write(self.stream_log)
        fileObject.close()

    def get_client_attr(self, attr='local_training_number', target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        return [getattr(client, attr) for client in target_client_list]

    def cal_vec_angle(self, vec_a, vec_b):
        denominator = torch.norm(vec_a) * torch.norm(vec_b)
        if float(denominator.detach().cpu().item()) <= EPS:
            return 0.0
        cosine = torch.clamp(vec_a @ vec_b / denominator, -1.0, 1.0)
        return float(torch.arccos(cosine)) / float(np.pi) * 180

    def cal_conflicts(self, g_locals, d):
        descent_vec = np.zeros(self.online_client_num)
        count = 0
        for i in range(self.online_client_num):
            angle = self.cal_vec_angle(g_locals[i], d)
            if angle > 90:
                descent_vec[i] = (angle - 90) / 90 * 100
                count += 1
        if count > 0:
            self.descent_log.append(
                (count, round(float(np.sum(descent_vec) / count), 2)))
        else:
            self.descent_log.append((count, 0))

        for l, layer_indices in enumerate(self.module.Loc_list):
            d_layer = d[layer_indices]
            descent_vec = np.zeros(self.online_client_num)
            count = 0
            for i in range(self.online_client_num):
                angle = self.cal_vec_angle(g_locals[i][layer_indices], d_layer)
                if angle > 90:
                    descent_vec[i] = (angle - 90) / 90 * 100
                    count += 1
