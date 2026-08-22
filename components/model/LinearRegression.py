import torch
import torch.nn as nn

import components as cn


class LinearRegression(cn.Module):
    """Bias-free 20-dimensional linear model used by Section V.A."""

    DIMENSION = 20

    def __init__(self, device):
        super().__init__(device)
        self.name = "LinearRegression"
        self.input_require_shape = [self.DIMENSION]

    def generate_model(self, input_data_shape, target_class_num):
        if list(input_data_shape) != [self.DIMENSION]:
            raise ValueError(
                f"LinearRegression requires input shape [{self.DIMENSION}], "
                f"got {list(input_data_shape)}."
            )
        if int(target_class_num) != 1:
            raise ValueError("LinearRegression requires one scalar target.")
        self.model = nn.Linear(self.DIMENSION, 1, bias=False).to(self.device)
        with torch.no_grad():
            self.model.weight.zero_()
        self.create_Loc_reshape_list()
