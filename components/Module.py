import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Module(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.name = "Module"
        self.device = device
        self.input_require_shape = None
        self.Loc_reshape_list = None
        self.Loc_list = None
        self.model = None

    def generate_model(self, input_data_shape, target_class_num):
        raise NotImplementedError

    def create_Loc_reshape_list(self):
        offset = 0
        self.Loc_reshape_list = []
        self.Loc_list = []
        for parameter in self.model.parameters():
            indices = torch.arange(offset, offset + parameter.numel())
            self.Loc_list.append(indices)
            self.Loc_reshape_list.append(indices.reshape(parameter.shape))
            offset += parameter.numel()

    def dot_vec(self, left, right):
        return sum(left[indices] @ right[indices] for indices in self.Loc_list)

    def span_model_grad_to_vec(self):
        gradients = [
            parameter.grad.detach().flatten().clone()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            return torch.cat(gradients)
        return torch.empty(0, device=self.device)

    def span_model_params_to_vec(self):
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def reshape_vec_to_model_params(self, vector):
        with torch.no_grad():
            vector_to_parameters(vector, self.model.parameters())
