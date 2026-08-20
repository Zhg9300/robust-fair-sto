
import components as cn
import torch


class FedAvg(cn.Algorithm):
    def __init__(self,
                 name='FedAvg',
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
            old_model = self.module.span_model_params_to_vec()

            m_locals, _ = self.train()
            g_locals = torch.stack([(old_model - m_local) / self.lr for m_local in m_locals])
            weights = None
            if self.gradient_aggregator.name == 'mean':
                weights = torch.as_tensor(
                    self.get_client_attr('local_training_number'),
                    device=g_locals.device,
                    dtype=g_locals.dtype,
                )
            update_direction = self.aggregate_gradients(g_locals, weights=weights)
            new_model = old_model - self.lr * update_direction
            self.module.reshape_vec_to_model_params(new_model)
            self.client_update()
