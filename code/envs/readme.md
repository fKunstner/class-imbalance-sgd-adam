# Environment variables

To work on multiple machines with different configurations, 
we use environment variables to configure the library.

For plotting, only `OPTEXP_WORKSPACE` is required.
The other variables (`OPTEXP_WANDB_*` and `OPTEXP_SLURM_*`) are used for running experiments.

The following variables are used:
```
## Main config

OPTEXP_WORKSPACE=~/workspace                            # Root dir to save datasets/results

## Wandb  

OPTEXP_WANDB_ENABLED=true                               # true unless debugging
OPTEXP_WANDB_MODE=offline                               # offline unless debugging
OPTEXP_WANDB_PROJECT=wandb_project                      # project to upload runs to
OPTEXP_WANDB_ENTITY=wandb_entity                        # entity to upload runs to
WANDB_API_KEY=ssssssssssssssssssssssssssssssssssssssss  # API key 

## Slurm configuration

OPTEXP_SLURM_NOTIFICATION_EMAIL=mail@example.com        # Slurm notification email
OPTEXP_SLURM_ACCOUNT=acc-name                           # Slurm billing account
```

For templates, see `env.bat.example` (Windows) and `env.sh.example` (Unix).

Create a `.bat` or `.sh` for each machine and run it before running any scripts.  
You can also add it to a `~/.bashrc` file to run it automatically.  
Do not commit these files to the repository (`.gitignore` is configured to ignore them).



