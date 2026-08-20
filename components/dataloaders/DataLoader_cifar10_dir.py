from components.dataloaders.indexed_vision_loader import IndexedVisionDataLoader


class DataLoader_cifar10_dir(IndexedVisionDataLoader):
    def __init__(self, pool_size=100, alpha=0.1, batch_size=100,
                 input_require_shape=None, shuffle=True, recreate=False,
                 params=None, device='cpu'):
        super().__init__(
            'cifar10', 'dir', pool_size=pool_size, alpha=alpha,
            batch_size=batch_size, input_require_shape=input_require_shape,
            shuffle=shuffle, recreate=recreate, params=params, device=device)
