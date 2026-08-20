from components.dataloaders.indexed_vision_loader import IndexedVisionDataLoader


class DataLoader_cifar10_pat(IndexedVisionDataLoader):
    def __init__(self, pool_size=100, item_classes_num=2, batch_size=100,
                 balance=True, input_require_shape=None, shuffle=True,
                 recreate=False, params=None, device='cpu'):
        super().__init__(
            'cifar10', 'pat', pool_size=pool_size,
            item_classes_num=item_classes_num, batch_size=batch_size,
            balance=balance, input_require_shape=input_require_shape,
            shuffle=shuffle, recreate=recreate, params=params, device=device)
