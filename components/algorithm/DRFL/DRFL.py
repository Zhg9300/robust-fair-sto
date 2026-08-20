
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

    def run(self):
        while not self.terminated():
            old_model_params = self.module.span_model_params_to_vec()
            m_locals, l_locals = self.train()
            l_locals = torch.Tensor(l_locals).float().to(self.device)

            weights = self.online_client_num / self.client_num * l_locals
            weight_sum = torch.sum(weights)
            if not torch.isfinite(weight_sum) or float(torch.abs(weight_sum).detach().cpu().item()) <= EPS:
                normalized_weights = torch.ones_like(weights) / len(weights)
                weights = normalized_weights
            else:
                normalized_weights = weights / weight_sum
            effective_updates = normalized_weights.reshape(-1, 1) * (
                torch.stack(m_locals) - old_model_params
            )
            self.record_attack_effective_metrics(
                effective_update_norms=torch.norm(effective_updates, dim=1),
                effective_weights=normalized_weights,
            )

            self.weight_aggregate(m_locals, weights=weights)
            self.client_update()
