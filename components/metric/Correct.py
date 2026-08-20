
import components as cn
import torch


class Correct(cn.Metric):
    def __init__(self):
        super().__init__(name='correct')

    @staticmethod
    def calc(network_output, target):
        _, predicted = torch.max(network_output, -1)
        return predicted.eq(target).sum().item()
