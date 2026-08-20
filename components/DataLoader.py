import numpy as np


class DataLoader:
    def __init__(self, name="DataLoader", nickname="DataLoader", pool_size=0,
                 batch_size=0, input_require_shape=None):
        self.name = name
        self.nickname = nickname
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.input_require_shape = input_require_shape
        self.input_data_shape = None
        self.target_class_num = None
        self.data_pool = None

    def allocate(self, clients):
        if getattr(self, "preserve_pool_order", False):
            pool_indices = list(range(len(clients)))
        else:
            pool_indices = np.random.choice(
                self.pool_size, len(clients), replace=False
            )
        for client, pool_index in zip(clients, pool_indices):
            item = self.data_pool[pool_index]
            client.update_data(
                pool_index,
                item["local_training_data"],
                item["local_training_number"],
                item["local_test_data"],
                item["local_test_number"],
            )

    def cal_data_shape(self, raw_shape):
        if len(self.input_require_shape) == len(raw_shape) - 1:
            self.input_data_shape = list(raw_shape[1:])
            return
        data_shape = []
        for index in range(1, len(raw_shape)):
            if index < len(self.input_require_shape) + 1:
                data_shape.append(raw_shape[index])
            else:
                data_shape[-1] *= raw_shape[index]
        self.input_data_shape = data_shape
