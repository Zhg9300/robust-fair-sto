
import math
import torch

import components as cn

EPS = 1e-12


class qFedAvg(cn.Algorithm):
    def __init__(self,
                 name='qFedAvg',
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
                 params=None,
                 q=0.1,
                 qffl_update_rule='normalized'):

        if params is not None:
            q = params['q']
            qffl_update_rule = params.get('qffl_update_rule', qffl_update_rule)
        q = float(q)
        if not math.isfinite(q) or q < 0.0:
            raise ValueError('q must be finite and non-negative.')
        qffl_update_rule = str(qffl_update_rule).strip().lower()
        if qffl_update_rule not in {'normalized', 'objective_gradient'}:
            raise ValueError(
                'qffl_update_rule must be "normalized" or "objective_gradient".'
            )
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' q' + str(q)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round,  epochs, save_name, outFunc, write_log,  test_conflicts, params)
        self.q = q
        self.qffl_update_rule = qffl_update_rule

        self.lr = self.train_setting['optimizer'].defaults['lr']
        if self.qffl_update_rule == 'objective_gradient':
            self._validate_objective_gradient_mode()
            self.save_name += ' qffl_objective_gradient'
        elif self.gradient_aggregator.name != 'mean':
            raise ValueError(
                'normalized qFedAvg does not support robust gradient aggregators; '
                'use qffl_update_rule="objective_gradient".'
            )

    def _validate_objective_gradient_mode(self):
        if self.epochs != 1:
            raise ValueError('objective_gradient requires E=1.')
        if self.online_client_num != self.client_num:
            raise ValueError('objective_gradient requires full client participation (C=1).')

    def _qffl_objective_gradient(self, gradient, loss):
        loss_tensor = torch.as_tensor(
            loss,
            device=gradient.device,
            dtype=gradient.dtype,
        ).clamp_min(EPS)
        return loss_tensor.pow(self.q) * gradient

    def _apply_objective_gradient_update(self, old_model_params, gradients):
        update_direction = self.aggregate_gradients(gradients)
        self.record_attack_effective_metrics(
            effective_update_norms=torch.linalg.vector_norm(
                self.lr * gradients,
                dim=1,
            ),
            report_path='evaluate',
        )
        self.module.reshape_vec_to_model_params(
            old_model_params - self.lr * update_direction
        )

    def _apply_normalized_update(self, old_model_params, local_models, losses):
        gradients = torch.stack([
            (old_model_params - local_model) / self.lr
            for local_model in local_models
        ])
        losses = torch.as_tensor(
            losses,
            device=gradients.device,
            dtype=gradients.dtype,
        ) + 1e-10
        deltas = losses.unsqueeze(1).pow(self.q) * gradients
        denominator_terms = (
            self.q
            * losses.pow(self.q - 1)
            * torch.linalg.vector_norm(gradients, dim=1).pow(2)
            + losses.pow(self.q) / self.lr
        )
        denominator = denominator_terms.sum()
        if (
            not torch.isfinite(denominator)
            or float(torch.abs(denominator).detach().cpu().item()) <= EPS
        ):
            denominator = torch.ones_like(denominator_terms).sum()

        self.record_attack_effective_metrics(
            effective_update_norms=torch.linalg.vector_norm(
                deltas / denominator,
                dim=1,
            ),
        )
        self.aggregate(old_model_params, deltas, denominator)

    def run(self):
        while not self.terminated():
            old_model_params = self.module.span_model_params_to_vec()
            if self.qffl_update_rule == 'objective_gradient':
                g_locals, _ = self.evaluate(
                    gradient_transform=self._qffl_objective_gradient,
                    use_full_loss=True,
                )
                self._apply_objective_gradient_update(old_model_params, g_locals)
            else:
                m_locals, l_locals = self.train()
                self._apply_normalized_update(
                    old_model_params,
                    m_locals,
                    l_locals,
                )
            self.client_update()

    def aggregate(self, old_model_params, Deltas, denominator):
        update = torch.sum(Deltas / denominator, dim=0)
        self.module.reshape_vec_to_model_params(old_model_params - update)
