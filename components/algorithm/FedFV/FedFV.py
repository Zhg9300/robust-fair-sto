
import components as cn
import torch
import copy
import math


class FedFV(cn.Algorithm):
    def __init__(self,
                 name='FedFV',
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

                 alpha=0.1,
                 tau=1):

        if params is not None:
            alpha = params['alpha']
            tau = params['tau']
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(
                train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' alpha' + str(alpha) + ' tau' + str(tau)

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round,  epochs, save_name, outFunc, write_log,  test_conflicts, params)
        self.alpha = alpha
        self.tau = tau

        self.lr = self.train_setting['optimizer'].defaults['lr']

        self.client_last_sample_round = [-1] * self.client_num
        self.client_grads_history = [0] * self.client_num

    @staticmethod
    def _model_updates_from_tensors(old_model, m_locals):
        return [old_model - m_local for m_local in m_locals]

    def run(self):
        while not self.terminated():
            old_models = self.module.span_model_params_to_vec()
            m_locals, l_locals = self.train()
            g_locals = self._model_updates_from_tensors(old_models, m_locals)

            for cid, gi in zip(self.get_client_attr('id'), g_locals):
                self.client_grads_history[cid] = gi
                self.client_last_sample_round[cid] = self.current_comm_round

            order_grads = copy.deepcopy(g_locals)
            order = [_ for _ in range(len(order_grads))]

            tmp = sorted(list(zip(l_locals, order)), key=lambda x: x[0])
            order = [x[1] for x in tmp]

            keep_original = []
            if self.alpha > 0:
                keep_original = order[math.ceil(
                    (len(order) - 1) * (1 - self.alpha)):]

            g_locals_L2_norm_square_list = []
            for g_local in g_locals:
                g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)

            for i in range(len(order_grads)):
                if i in keep_original:
                    continue
                for j in order:
                    if j == i:
                        continue
                    else:

                        dot = self.module.dot_vec(g_locals[j], order_grads[i])
                        if dot < 0:
                            order_grads[i] = order_grads[i] - dot / \
                                (g_locals_L2_norm_square_list[j] +
                                 1e-6) * g_locals[j]

            weights = torch.Tensor(
                [1 / len(order_grads)] * len(order_grads)).float().to(self.device)
            gt = weights @ torch.stack([order_grads[i]
                                       for i in range(len(order_grads))])

            if self.current_comm_round >= self.tau:
                for k in range(self.tau-1, -1, -1):

                    gcs = [self.client_grads_history[cid] for cid in range(
                        self.data_loader.pool_size) if self.client_last_sample_round[cid] == self.current_comm_round - k and self.module.dot_vec(gt, self.client_grads_history[cid]) < 0]
                    if gcs:
                        gcs = torch.vstack(gcs)
                        g_con = torch.sum(gcs, dim=0)

                        dot = self.module.dot_vec(gt, g_con)
                        if dot < 0:
                            gt = gt - dot / (torch.norm(g_con)**2+1e-6) * g_con

            gnorm = torch.norm(
                weights @ torch.stack([g_locals[i] for i in range(len(g_locals))]))
            gt = gt / (torch.norm(gt) + 1e-6) * gnorm

            for i, p in enumerate(self.module.model.parameters()):
                p.data -= gt[self.module.Loc_reshape_list[i]]
            self.client_update()

            if self.test_conflicts:
                self.cal_conflicts(g_locals, gt)
