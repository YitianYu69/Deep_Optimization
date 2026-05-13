import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler


from timm.data import create_transform
from torchvision import datasets

from pathlib import Path



class FixedImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform, class_to_idx):
        self._provide_cti = class_to_idx
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory):
        num_classes = len(self._provide_cti)
        class_to_idx = [None] * num_classes

        for class_name, idx in self._provide_cti.items():
            if idx < 0 or idx >= num_classes:
                raise ValueError(f"Wrong class idx: {idx} to class name: {class_name}")
            else:
                class_to_idx[idx] = class_name
        if any(c is None for c in class_to_idx):
            raise ValueError(f"Class to idx is not dense, contains None!")
        return class_to_idx, self._provide_cti
    
def get_transform(dtype, image_size):
    return create_transform(
        input_size=image_size,
        is_training=(dtype == 'train'),
        auto_augment='rand-m9-n2-mstd0.5' if dtype == 'train' else None
    )

def get_dataloader(batch_size, num_workers, drop_last,
                   train_dataset=None, valid_dataset=None, test_dataset=None,
                   root=None, image_size=None,
                   split_strategy=None,
                   *, ddp=False, global_rank=None, world_size=None, pin_memory_device=None):

    if root is not None and image_size is not None and split_strategy is not None:
        root_path = Path(root)
        train_path = root_path / 'train'
        valid_path = root_path / 'valid'

        train_transform = get_transform('train', image_size)
        valid_transform = get_transform('valid', image_size)

        train_dataset = datasets.ImageFolder(train_path, train_transform)
        check_dataset = datasets.ImageFolder(train_path, valid_transform)
        cti = train_dataset.class_to_idx

        train_len = int(split_strategy[0] * len(train_dataset))
        valid_len = len(train_dataset) - train_len
        train_indices, valid_indices = random_split(range(len(train_dataset)), [split_strategy[0], split_strategy[1]])

        train_dataset = Subset(train_dataset, train_indices.indices)
        valid_dataset = Subset(check_dataset, valid_indices.indices)
        test_dataset = FixedImageDataset(valid_path, valid_transform, cti)
    elif train_dataset is not None and valid_dataset is not None and test_dataset is not None:
        pass
    else:
        ValueError("Please eith provide the datases, or the root path")


    if ddp:
        assert (global_rank is not None and world_size is not None), "Both global rank and world size cannot be None for DDP"
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=drop_last, num_replicas=world_size, rank=global_rank)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=drop_last, num_replicas=world_size, rank=global_rank)
    else:
        train_sampler, valid_sampler = None, None
    
    worker_kwargs = {}
    if num_workers and num_workers > 0:
        worker_kwargs.update(dict(prefetch_factor=3, persistent_workers=True))
    
    
    if not ddp:
        train_shuffle = True
    else:
        train_shuffle = False

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_shuffle, batch_size=batch_size, drop_last=drop_last,
                                  num_workers=num_workers, pin_memory=True, **worker_kwargs)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=batch_size, drop_last=drop_last,
                                  num_workers=num_workers, pin_memory=True, **worker_kwargs)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=512, num_workers=num_workers)
    return train_dataloader, valid_dataloader, test_dataloader