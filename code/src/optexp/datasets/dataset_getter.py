from typing import Optional

from optexp import config
from optexp.datasets.loaders import (
    load_frozen,
    load_frozen_100,
    load_frozen_300,
    load_frozen_600,
    load_micro_mnist,
    load_mnist,
    load_ptb,
    load_tiny_mnist,
    load_tiny_ptb,
    load_tiny_wt2,
    load_wt2,
    load_wt103,
    load_tiny_stories,
    load_imagenet,
    load_imbalanced_imagenet,
    load_10_big_class_imagenet,
    load_1_major_class_imagenet,
    load_small_imagenet,
)


def get_dataset(
    dataset_name,
    batch_size,
    split_prop,
    shuffle,
    num_workers,
    normalize,
):
    raise NotImplementedError()


def get_image_dataset(
    dataset_name, batch_size, shuffle, num_workers, normalize, flatten
):
    if dataset_name == "ImageNet":
        return load_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "SmallImageNet":
        return load_small_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "ImbalancedImageNet":
        return load_imbalanced_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "TenBigClassImageNet":
        return load_10_big_class_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "OneMajorClassImageNet":
        return load_1_major_class_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "MNIST":
        return load_mnist(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "TinyMNIST":
        return load_tiny_mnist(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "MicroMNIST":
        return load_micro_mnist(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    else:
        raise Exception(f"No Image dataset with name {dataset_name} available.")


def get_text_dataset(
    dataset_name,
    batch_size,
    tgt_len,
    merge: Optional[int] = None,
    mixed_batch_size: Optional[bool] = False,
    eval_batch_size: Optional[int] = 0,
):
    if dataset_name == "PTB":
        return load_ptb(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "TinyPTB":
        return load_tiny_ptb(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "WikiText2":
        return load_wt2(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "TinyWikiText2":
        return load_tiny_wt2(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "WikiText103":
        return load_wt103(
            save_path=config.get_dataset_directory(),
            tokenizers_path=config.get_tokenizers_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            mixed_batch_size=mixed_batch_size,
            eval_batch_size=eval_batch_size,
        )
    elif dataset_name == "TinyStories":
        return load_tiny_stories(
            save_path=config.get_dataset_directory(),
            tokenizers_path=config.get_tokenizers_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
        )
    else:
        raise Exception(f"No Text dataset with name {dataset_name}")


def get_frozen_dataset(dataset_name, batch_size, porp):
    if dataset_name == "Frozen":
        return load_frozen(
            config.get_dataset_directory(), batch_size, config.get_device(), porp=porp
        )
    elif dataset_name == "Frozen_600":
        return load_frozen_600(
            config.get_dataset_directory(), batch_size, config.get_device(), porp=porp
        )
    elif dataset_name == "Frozen_300":
        return load_frozen_300(
            config.get_dataset_directory(), batch_size, config.get_device(), porp=porp
        )
    elif dataset_name == "Frozen_100":
        return load_frozen_100(
            config.get_dataset_directory(), batch_size, config.get_device(), porp=porp
        )
    else:
        raise Exception(f"No Frozen dataset with name {dataset_name}")
