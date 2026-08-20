
import components as cn
import torch
import numpy as np
from components.algorithm.common.utils import get_d_mgdaplus_d

EPS = 1e-12


class FedMGDA_plus(cn.Algorithm):
    def __init__(self,
                 name='FedMGDA+',
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
                 epsilon=0.1):

        if params is not None:
            epsilon = params['epsilon']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' epsilon' + str(epsilon)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round,  epochs, save_name, outFunc, write_log,  test_conflicts, params)
        self.epsilon = epsilon

        self.comm_log['d_optimality_history'] = []
        self.comm_log['d_descent_history'] = []

    def run(self):
        while not self.terminated():
            old_model = self.module.span_model_params_to_vec()
            m_locals, l_locals = self.train()

            g_locals = torch.stack([
                (old_model - local_model) / self.lr
                for local_model in m_locals
            ])
            grad_norms = torch.norm(g_locals, dim=1)
            if not torch.any(grad_norms > EPS):
                self.client_update()
                continue
            g_locals = g_locals / torch.clamp(grad_norms, min=EPS).reshape(-1, 1)
            training_nums = self.get_client_attr('local_training_number')
            lambda0 = np.array(training_nums) / sum(training_nums)

            d, _, _ = get_d_mgdaplus_d(
                g_locals, self.device, self.epsilon, lambda0)

            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()
