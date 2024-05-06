# Installing dependencies

The dependencies are split in the following files;
- Needed to run the experiments:
  - `main.txt`: Main packages to download from PyPI
  - `torch.txt`: Pytorch and other packages that need special installation depending on hardware
- Dev tools:
  - `docs.txt`: Requirements specific to the documentation
  - `dev.txt`: Additional tools for development

## Installing pytorch (`torch.txt`)

We use version `1.13.1`. To install them locally, use 

```
# ROCM 5.2 (Linux only)
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

Source: https://pytorch.org/get-started/previous-versions/#linux-and-windows-3  
The direct link might not work in the future, `ctrl+f` for `1.13.1` 
and check the `wheels` header.

To install them on a SLURM cluster with pre-compiled wheels, use `pip install -r --no-index torch.txt`

