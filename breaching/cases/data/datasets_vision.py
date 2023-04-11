"""Additional torchvision-like datasets."""

import torch
import torchvision

import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _build_dataset_vision(cfg_data, split, can_download=True):
    _default_t = torchvision.transforms.ToTensor()
    cfg_data.path = os.path.expanduser(cfg_data.path)
    # cfg_data.name == "ImageNetAnimals"
    dataset = torchvision.datasets.ImageNet(
        root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    indices = [idx for (idx, label) in dataset.lookup.items() if label < 397]
    dataset.classes = dataset.classes[:397]
    dataset.samples = [dataset.samples[i] for i in indices]  # Manually remove samples instead of using a Subset
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))

    transforms = _parse_data_augmentations(cfg_data, split)

    # Apply transformations
    dataset.transform = transforms if transforms is not None else None

    # Save data mean and data std for easy access:
    if cfg_data.normalize:
        dataset.mean = cfg_data.mean
        dataset.std = cfg_data.std
    else:
        dataset.mean = [0]
        dataset.std = [1]

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(dataset):
        dataset = Subset(dataset, torch.arange(0, cfg_data.size))

    collate_fn = _torchvision_collate
    return dataset, collate_fn


def _split_dataset_vision(dataset, cfg_data, user_idx=None, return_full_dataset=False):   # NOTE(dchu): FISHING
    if not return_full_dataset:   # NOTE(dchu): FISHING
        if user_idx is None:
            user_idx = torch.randint(0, cfg_data.default_clients, (1,))
        else:   # NOTE(dchu): FISHING
            if user_idx > cfg_data.default_clients:
                raise ValueError("This user index exceeds the maximal number of clients.")

        # Create a synthetic split of the dataset over all possible users if no natural split is given
        if cfg_data.partition == "balanced":
            data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
            if data_per_class_per_user < 1:
                raise ValueError("Too many clients for a balanced dataset.")
            data_ids = []
            for class_idx, _ in enumerate(dataset.classes):
                data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
                data_ids += data_with_class[
                    user_idx * data_per_class_per_user : data_per_class_per_user * (user_idx + 1)
                ]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "unique-class":
            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == user_idx]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "mixup":
            if "mixup_freq" in cfg_data:
                mixup_freq = cfg_data.mixup_freq
            else:
                # use default mixup_freq=2
                mixup_freq = 2
            data_per_user = len(dataset) // cfg_data.default_clients
            last_id = len(dataset) - 1
            data_ids = []
            for i in range(data_per_user):
                data_ids.append(user_idx * data_per_user + i)
                data_ids.append(last_id - user_idx * data_per_user - i)
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "feat_est":
            if "num_data_points" in cfg_data:
                num_data_points = cfg_data.num_data_points
            else:
                num_data_points = 1

            if "target_label" in cfg_data:
                target_label = cfg_data.target_label
            else:
                target_label = 0

            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == target_label]
            data_ids = data_ids[user_idx * num_data_points : (user_idx + 1) * num_data_points]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random-full":  # Data might be repeated across users (e.g. meme images)
            data_per_user = len(dataset) // cfg_data.default_clients
            data_ids = torch.randperm(len(dataset))[:data_per_user]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random":  # Data not replicated across users. Split is deterministic over reruns!   # NOTE(dchu): FISHING
            data_per_user = len(dataset) // cfg_data.default_clients
            generator = torch.Generator()
            generator.manual_seed(233)
            data_ids = torch.randperm(len(dataset))[user_idx * data_per_user : data_per_user * (user_idx + 1)]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "none":  # Replicate on all users for a sanity check!
            pass
        else:
            raise ValueError(f"Partition scheme {cfg_data.partition} not implemented.")
    return dataset


def _torchvision_collate(batch):
    """Small hack around the pytorch default collator to return a dictionary"""
    transposed = list(zip(*batch))

    def _stack_tensor(tensor_list):
        elem = tensor_list[0]
        elem_type = type(elem)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in tensor_list)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(tensor_list, 0, out=out)

    return dict(inputs=_stack_tensor(transposed[0]), labels=torch.tensor(transposed[1]))


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


def _parse_data_augmentations(cfg_data, split, PIL_only=False):
    def _parse_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            for key in cfg_dict.keys():
                try:  # ducktype iterable
                    transform = getattr(torchvision.transforms, key)(*cfg_dict[key])
                except TypeError:
                    transform = getattr(torchvision.transforms, key)(cfg_dict[key])
                list_of_transforms.append(transform)
        return list_of_transforms

    if split == "train":
        transforms = _parse_cfg_dict(cfg_data.augmentations_train)
    else:
        transforms = _parse_cfg_dict(cfg_data.augmentations_val)

    if not PIL_only:
        transforms.append(torchvision.transforms.ToTensor())
        if cfg_data.normalize:
            transforms.append(torchvision.transforms.Normalize(cfg_data.mean, cfg_data.std))
    return torchvision.transforms.Compose(transforms)
