
import components as cn
import torch

EPS = 1e-12


class DRFL(cn.Algorithm):
    def __init__(self,
                 name='DRFL',
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

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round,  epochs, save_name, outFunc, write_log,  test_conflicts, params)

    def _aggregation_messages(self, old_model_params, local_models, local_losses):
        """Return loss-weighted client gradients and their normalized weights."""
        gradients = torch.stack([
            (old_model_params - local_model) / self.lr
            for local_model in local_models
        ])
        losses = torch.as_tensor(
            local_losses,
            device=gradients.device,
            dtype=gradients.dtype,
        )

        weights = self.online_client_num / self.client_num * losses
        weight_sum = torch.sum(weights)
        if (
            not torch.isfinite(weight_sum)
            or float(torch.abs(weight_sum).detach().cpu().item()) <= EPS
        ):
            normalized_weights = torch.ones_like(weights) / len(weights)
        else:
            normalized_weights = weights / weight_sum

        # Robust aggregators do not consume external weights. Embed the DRFL
        # weight in each message and multiply by the online count so that mean
        # aggregation exactly recovers sum_i alpha_i * gradient_i.
        messages = (
            gradients.shape[0]
            * normalized_weights.reshape(-1, 1)
            * gradients
        )
        return messages, normalized_weights

    def run(self):
        while not self.terminated():
            old_model_params = self.module.span_model_params_to_vec()
            m_locals, l_locals = self.train()
            messages, normalized_weights = self._aggregation_messages(
                old_model_params,
                m_locals,
                l_locals,
            )
            update_direction = self.aggregate_gradients(messages)
            self.record_attack_effective_metrics(
                effective_update_norms=torch.linalg.vector_norm(
                    self.lr * messages,
                    dim=1,
                ),
                effective_weights=normalized_weights,
            )

            self.module.reshape_vec_to_model_params(
                old_model_params - self.lr * update_direction
            )
            self.client_update()
