from optexp import config
from optexp.datasets.loaders import (
    download_micro_mnist,
    download_mnist,
    download_ptb,
    download_tiny_mnist,
    download_tiny_ptb,
    download_tiny_wt2,
    download_wt2,
    download_wt103,
    download_tiny_stories,
)


def download_dataset(dataset_name):
    if dataset_name == "dummy_class" or dataset_name == "MNIST":
        download_mnist(config.get_dataset_directory())
    elif dataset_name == "TinyMNIST":
        download_tiny_mnist(config.get_dataset_directory())
    elif dataset_name == "MicroMNIST":
        download_micro_mnist(config.get_dataset_directory())
    elif dataset_name == "PTB":
        download_ptb(config.get_dataset_directory())
    elif dataset_name == "TinyPTB":
        download_tiny_ptb(config.get_dataset_directory())
    elif dataset_name == "WikiText2":
        download_wt2(config.get_dataset_directory())
    elif dataset_name == "TinyWikiText2":
        download_tiny_wt2(config.get_dataset_directory())
    elif dataset_name == "WikiText103":
        download_wt103(config.get_dataset_directory())
    elif dataset_name == "ImageNet":
        print("Downloading ImageNet takes days. Find your own.")
    elif dataset_name == "SmallImageNet":
        print("This is an artisan hand crafed dataset. Refer to new_scripts to make your own.")
    elif dataset_name == "ImbalancedImageNet":
        print("This is an artisan hand crafed dataset. Refer to new_scripts to make your own.")
    elif dataset_name == "TenBigClassImageNet":
        print("This is an artisan hand crafed dataset. Refer to new_scripts to make your own.")
    elif dataset_name == "OneMajorClassImageNet":
        print("This is an artisan hand crafed dataset. Refer to new_scripts to make your own.")
    elif dataset_name == "TinyStories":
        download_tiny_stories(config.get_dataset_directory())
    elif dataset_name == "Frozen":
        pass
    elif dataset_name == "Frozen_600":
        pass
    elif dataset_name == "Frozen_300":
        pass
    elif dataset_name == "Frozen_100":
        pass
    elif dataset_name in ["dummy_reg", "dummy_class"]:
        return
    else:
        raise Exception(f"No dataset with name {dataset_name} available.")
