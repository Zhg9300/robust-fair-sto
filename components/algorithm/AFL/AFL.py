
import components as cn
import numpy as np
import torch

class AFL(cn.Algorithm):
    def __init__(self,
                 name='AFL',
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
                 lam=0.5):

        if params is not None:
            lam = params['lam']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' lam' + str(lam)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round,  epochs, save_name, outFunc, write_log,  test_conflicts, params)
        self.lam = lam

        self.dynamic_lambdas = np.ones(
            self.online_client_num) * 1.0 / self.online_client_num


    def run(self):
        while not self.terminated():
            m_locals, l_locals = self.train()
            old_model = self.module.span_model_params_to_vec()
            g_locals = torch.stack([
                (old_model - local_model) / self.lr
                for local_model in m_locals
            ])

            weights = torch.Tensor(
                self.dynamic_lambdas).float().to(self.device)

            d = []
            for i in range(len(self.module.Loc_list)):
                g_locals_layer = g_locals[:, self.module.Loc_list[i]]
                d_layer = weights @ g_locals_layer
                d.append(d_layer)
            d = torch.hstack(d)

            self.update_module(self.module, self.optimizer, self.lr, d)
            self.client_update()
            self.dynamic_lambdas = [
                lmb_i+self.lam * float(loss_i) for lmb_i, loss_i in zip(self.dynamic_lambdas, l_locals)]
            self.dynamic_lambdas = self.project(self.dynamic_lambdas)


    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0/(i + 1)) * (1 - np.sum(np.asarray(u)[:i+1]))) > 0:
                rho = i + 1
        lamb = (1.0/(rho+1e-6)) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i] + lamb, 0))
        return res
