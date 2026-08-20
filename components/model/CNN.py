
import components as cn
import torch.nn as nn


class CNN(cn.Module):

    def __init__(self, device):
        super(CNN, self).__init__(device)
        self.name = 'CNN'
        self.model = None

        self.input_require_shape = [3, 32, 32]

        self.ignore_head = False

    def generate_model(self, input_data_shape, target_class_num):
        self.model = CNN_Model(self.ignore_head, input_data_shape,
                               target_class_num).to(self.device)
        self.create_Loc_reshape_list()


class CNN_Model(nn.Module):
    def __init__(self, ignore_head, input_data_shape, target_class_num):
        super(CNN_Model, self).__init__()
        self.ignore_head = ignore_head
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1600, 384),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(192, target_class_num)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
