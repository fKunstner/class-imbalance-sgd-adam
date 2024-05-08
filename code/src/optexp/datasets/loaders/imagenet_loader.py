import os
import torch
import numpy as np
from distutils.dir_util import copy_tree
from optexp.config import get_logger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




def load_small_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/SmallImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/SmallImageNet"
        # bad and hack
        if not os.path.isfile(local_disk + "/SMALL_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/SMALL_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, 'train')
    valdir = os.path.join(local_disk, 'val')
    
    if normalize:
        normalize_transform = transforms.Normalize(mean=[0.4840, 0.4531, 0.4013],
                                     std=[0.2180, 0.2138, 0.2126])
    else:
        normalize_transform = transforms.Normalize(mean=[0.0,0.0,0.0],
                                     std=[1.0,1.0,1.0])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)

def load_1_major_class_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/OneMajorClassImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/OneMajorClassImageNet"
        # bad and hack
        if not os.path.isfile(local_disk + "/ONE_MAJOR_CLASS_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/ONE_MAJOR_CLASS_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, 'train')
    valdir = os.path.join(local_disk, 'val')
    
    if normalize:
        # computed for this version of the dataset (if you rerun the script to create it this should be recomputed as well)
        #    tensor([0.4710, 0.4465, 0.4038])
        #    tensor([0.2197, 0.2165, 0.2151])

        normalize_transform = transforms.Normalize(mean=[0.4632, 0.4637, 0.4387],
                                     std=[0.2267, 0.2211, 0.2281])
    else:
        normalize_transform = transforms.Normalize(mean=[0.0,0.0,0.0],
                                     std=[1.0,1.0,1.0])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)

def load_10_big_class_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/TenBigClassImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/imagenet"
        # bad and hack
        if not os.path.isfile(local_disk + "/TEN_BIG_CLASS_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/TEN_BIG_CLASS_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, 'train')
    valdir = os.path.join(local_disk, 'val')
    
    if normalize:
        # computed for this version of the dataset (if you rerun the script to create it this should be recomputed as well)
        #    tensor([0.4710, 0.4465, 0.4038])
        #    tensor([0.2197, 0.2165, 0.2151])

        normalize_transform = transforms.Normalize(mean=[0.4710, 0.4465, 0.4038],
                                     std=[0.2197, 0.2165, 0.2151])
    else:
        normalize_transform = transforms.Normalize(mean=[0.0,0.0,0.0],
                                     std=[1.0,1.0,1.0])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)



def load_imbalanced_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/ImbalancedImageNet"
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is not None:
        local_disk += "/imagenet"
        # bad and hack
        if not os.path.isfile(local_disk + "/IMBALANCED_IMAGENET"):
            os.system(f"cp --recursive {dataset_dir} {local_disk}")
            os.system(f"touch {local_disk}/IMBALANCED_IMAGENET")
    else:
        local_disk = dataset_dir

    traindir = os.path.join(local_disk, 'train')
    valdir = os.path.join(local_disk, 'val')
    
    if normalize:
        # computed for this version of the dataset (if you rerun the script to create it this should be recomputed as well)
        #tensor([0.4770, 0.4470, 0.3857])
        #tensor([0.2218, 0.2150, 0.2130])

        normalize_transform = transforms.Normalize(mean=[0.4770, 0.4470, 0.3857],
                                     std=[0.2218, 0.2150, 0.2130])
    else:
        normalize_transform = transforms.Normalize(mean=[0.0,0.0,0.0],
                                     std=[1.0,1.0,1.0])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)


def load_imagenet(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):

    dataset_dir = str(save_path) + "/ImageNet"
        
    local_dataset = extract(dataset_dir)
        
    traindir = os.path.join(local_dataset, 'train')
    valdir = os.path.join(local_dataset, 'val')
    
    if normalize:
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize_transform = transforms.Normalize(mean=[0.0,0.0,0.0],
                                     std=[1.0,1.0,1.0])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
    targets = torch.tensor(train_dataset.targets)
    output_shape = np.array([targets.max().item() + 1])
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    features, _ = next(iter(train_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(targets)



def extract(dataset_dir):
    local_disk = os.getenv("SLURM_TMPDIR")
    if local_disk is None:
        raise ValueError("Cannot locate node scratch, it is only visible on the main node process.")
    if not os.path.isfile(local_disk + "/imagenet" + "/EXTRACTED"):
        get_logger().info("Extracting ImageNet Dataset")
        os.system(f"bash {dataset_dir}/extract_ILSVRC.sh {dataset_dir} {local_disk}")
    else:
        get_logger().info("Dataset Already Extracted")
    return local_disk + "/imagenet"


if __name__ == "__main__":
    extract("/home/alanmil/optexp/datasets/ImageNet")

