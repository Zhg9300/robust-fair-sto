import torch
import torch.nn.functional as functional
import torchvision

from components.DataLoader import DataLoader
from components.param_utils import parse_batch_size, parse_bool
from components.dataloaders.indexed_data import configure_indexed_loader


_DATASET_CONFIG = {
    'fashion': {
        'class': torchvision.datasets.FashionMNIST,
        'classes': 10,
        'mean': None,
        'std': None,
    },
    'cifar10': {
        'class': torchvision.datasets.CIFAR10,
        'classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
    },
}


def _static_cifar_augmentation(inputs, seed):
    """Reproduce CIFAR-10 dir's historical one-time crop/flip deterministically."""
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    padded = functional.pad(inputs, (4, 4, 4, 4), mode='constant', value=0)
    offsets_y = torch.randint(0, 9, (len(inputs),), generator=generator)
    offsets_x = torch.randint(0, 9, (len(inputs),), generator=generator)
    flip_mask = torch.rand(len(inputs), generator=generator) < 0.5
    augmented = torch.empty_like(inputs)
    for index, (offset_y, offset_x) in enumerate(zip(offsets_y, offsets_x)):
        image = padded[index, :, offset_y:offset_y + 32, offset_x:offset_x + 32]
        augmented[index] = torch.flip(image, dims=(2,)) if flip_mask[index] else image
    return augmented


def _dataset_tensors(dataset_name, dataset, static_augmentation=False, seed=1):
    inputs = torch.as_tensor(dataset.data)
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    if dataset_name == 'fashion':
        inputs = inputs.unsqueeze(1).float().div_(255.0)
    else:
        inputs = inputs.permute(0, 3, 1, 2).float().div_(255.0)
        if static_augmentation:
            inputs = _static_cifar_augmentation(inputs, seed)
        config = _DATASET_CONFIG[dataset_name]
        mean = inputs.new_tensor(config['mean']).view(1, -1, 1, 1)
        std = inputs.new_tensor(config['std']).view(1, -1, 1, 1)
        inputs.sub_(mean).div_(std)
    return inputs, targets


class IndexedVisionDataLoader(DataLoader):
    """Shared implementation for all current image federation loaders."""

    def __init__(self, dataset_name, partition, pool_size=100,
                 item_classes_num=None, alpha=None, batch_size=100,
                 balance=True, input_require_shape=None, shuffle=True,
                 recreate=False, params=None, device='cpu'):
        if params is not None:
            pool_size = int(params['N'])
            batch_size = parse_batch_size(params['B'])
            if partition == 'dir':
                alpha = params.get('Diralpha', alpha)
                item_classes_num = None
            else:
                item_classes_num = params.get('NC', item_classes_num)
                alpha = None
            balance = parse_bool(params.get('balance', balance), 'balance')
            data_device = params.get('data_device', 'model')
            partition_seed = int(params.get('partition_seed', 1))
            min_client_samples = params.get('min_client_samples', 'auto')
        else:
            batch_size = parse_batch_size(batch_size)
            data_device = 'model'
            partition_seed = 1
            min_client_samples = 'auto'
        if data_device not in {'model', 'cpu'}:
            raise ValueError("data_device must be 'model' or 'cpu'.")

        config = _DATASET_CONFIG[dataset_name]
        balance_suffix = 'balanced' if balance else 'unbalanced'
        partition_text = (
            f'alpha{alpha}' if partition == 'dir'
            else f'NC{item_classes_num}_{balance_suffix}'
        )
        name = (
            f'{dataset_name}_{partition}_N{pool_size}_{partition_text}_B{batch_size}_indexed_v1'
        )
        nickname = (
            f'{dataset_name} {partition} {partition_text} B{batch_size} N{pool_size}'
        )
        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)
        self.device = torch.device(device)
        self.data_device = data_device
        self.preserve_pool_order = True
        self.target_class_num = config['classes']

        trainset = config['class'](
            root=self._data_folder(), train=True, download=True, transform=None)
        testset = config['class'](
            root=self._data_folder(), train=False, download=True, transform=None)
        train_inputs, train_targets = _dataset_tensors(
            dataset_name,
            trainset,
            static_augmentation=(dataset_name == 'cifar10' and partition == 'dir'),
            seed=partition_seed,
        )
        test_inputs, test_targets = _dataset_tensors(dataset_name, testset)
        self.cal_data_shape(train_inputs.shape)
        train_inputs = train_inputs.reshape(-1, *self.input_data_shape)
        test_inputs = test_inputs.reshape(-1, *self.input_data_shape)
        self.total_training_number = len(train_targets)
        self.total_test_number = len(test_targets)

        configure_indexed_loader(
            self,
            dataset_name,
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            pool_size,
            self.target_class_num,
            batch_size,
            data_device,
            self.device,
            partition_seed,
            self._pool_folder(),
            recreate=recreate,
            item_classes_num=item_classes_num,
            alpha=alpha,
            balance=balance,
            partition=partition,
            shuffle=shuffle,
            min_client_samples=min_client_samples,
        )

    @staticmethod
    def _data_folder():
        import components as cn
        return cn.data_folder_path

    @staticmethod
    def _pool_folder():
        import components as cn
        return cn.pool_folder_path
