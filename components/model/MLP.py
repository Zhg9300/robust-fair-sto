
import components as cn
import torch
import torch.nn.functional as F


class MLP(cn.Module):

    def __init__(self, device):
        super(MLP, self).__init__(device)
        self.name = 'MLP'
        self.model = None

        self.input_require_shape = [784]

        self.ignore_head = False

    def generate_model(self, input_data_shape, target_class_num):
        self.target_class_num = target_class_num
        self.model = MLP_Model(
            self.ignore_head, input_data_shape, target_class_num).to(self.device)
        self.create_Loc_reshape_list()


class MLP_Model(torch.nn.Module):
    def __init__(self, ignore_head, input_data_shape, target_class_num):
        super(MLP_Model, self).__init__()
        self.ignore_head = ignore_head
        input_dim = input_data_shape[0]
        self.fc1 = torch.nn.Linear(input_dim, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.predictor = torch.nn.Linear(200, target_class_num)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
