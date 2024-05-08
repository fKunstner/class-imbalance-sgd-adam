import logging
import os
from logging import Logger
from pathlib import Path
from typing import Optional

import torch

ENV_VAR_WORKSPACE = "OPTEXP_WORKSPACE"
ENV_VAR_LOGGING = "OPTEXP_CONSOLE_LOGGING_LEVEL"
ENV_VAR_WANDB_ENABLED = "OPTEXP_WANDB_ENABLED"
ENV_VAR_WANDB_PROJECT = "OPTEXP_WANDB_PROJECT"
ENV_VAR_WANDB_ENTITY = "OPTEXP_WANDB_ENTITY"
ENV_VAR_WANDB_MODE = "OPTEXP_WANDB_MODE"
ENV_VAR_WANDB_API_KEY = "WANDB_API_KEY"
ENV_VAR_SLURM_EMAIL = "OPTEXP_SLURM_NOTIFICATION_EMAIL"
ENV_VAR_SLURM_ACCOUNT = "OPTEXP_SLURM_ACCOUNT"
LOG_FMT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


class UseWandbProject:
    """
    Context manager to set the wandb project for a block of code.
    Temporarily overrides the global project set in the environment variable.
    Used in get_wandb_project().
    """

    global_project: Optional[str] = None

    def __init__(self, project: Optional[str] = None):
        self.project_for_context = project

    def __enter__(self):
        self.project_outside_context = self.global_project
        UseWandbProject.global_project = self.project_for_context

    def __exit__(self, *args, **kws):
        UseWandbProject.global_project = self.project_outside_context


def get_wandb_key() -> str:
    api_key = os.environ.get(ENV_VAR_WANDB_API_KEY, None)
    if api_key is None:
        raise ValueError(
            f"WandB API key is not defined. Define the {ENV_VAR_WANDB_API_KEY} "
            "environment variable to set the API key"
        )
    return api_key


def get_wandb_timeout() -> int:
    """Timeout for data transfers, in seconds.

    Large timeout are needed to download runs with large logs (per class).
    """
    return 60


def get_wandb_status() -> bool:
    status = os.environ.get(ENV_VAR_WANDB_ENABLED, None)
    if status is None:
        raise ValueError(
            f"WandB status not set. Define the {ENV_VAR_WANDB_ENABLED} "
            "environment variable as True or False to define whether to use WandB"
        )
    return status.lower() == "true"


def get_wandb_project() -> str:
    project: Optional[str] = None
    if UseWandbProject.global_project is not None:
        project = UseWandbProject.global_project
    else:
        project = os.environ.get(ENV_VAR_WANDB_PROJECT, None)

    if project is None:
        raise ValueError(
            f"WandB project not set. Define the {ENV_VAR_WANDB_PROJECT} "
            "environment variable to define the project"
        )
    return str(project)


def get_wandb_entity() -> str:
    entity = os.environ.get(ENV_VAR_WANDB_ENTITY, None)
    if entity is None:
        raise ValueError(
            f"WandB entity not set. Define the {ENV_VAR_WANDB_ENTITY} "
            "environment variable to define the entity"
        )
    return str(entity)


def get_wandb_mode() -> str:
    mode = os.environ.get(ENV_VAR_WANDB_MODE, "offline")

    if mode != "online" and mode != "offline":
        raise ValueError(
            f"Invalid wandb mode set in environment variable {ENV_VAR_WANDB_MODE}."
            f"Expected 'online' or 'offline'. Got {mode}."
        )

    return mode


def get_workspace_directory() -> Path:
    workspace = os.environ.get(ENV_VAR_WORKSPACE, None)
    if workspace is None:
        raise ValueError(
            "Workspace not set. "
            f"Define the {ENV_VAR_WORKSPACE} environment variable"
            "To define where to save datasets and experiment results."
        )
    return Path(workspace)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        get_logger("GPU not available, running experiments on CPU.", logging.WARNING)
    return device


def get_dataset_directory() -> Path:
    return get_workspace_directory() / "datasets"


def get_tokenizers_directory() -> Path:
    return get_workspace_directory() / "tokenizers"


def get_experiment_directory() -> Path:
    return get_workspace_directory() / "experiments"


def get_wandb_cache_directory() -> Path:
    return get_workspace_directory() / "wandb_cache"


def get_plots_directory() -> Path:
    return get_workspace_directory() / "plots"


def get_final_plots_directory() -> Path:
    return get_workspace_directory() / Path("plots") / "RESULTS"


def get_console_logging_level() -> str:
    return os.environ.get(ENV_VAR_LOGGING, "DEBUG")


def get_slurm_email() -> str:
    """
    Email to use for slurm notifications, defined in an environment variable.

    Raises:
         ValueError: if the environment variable is not set.
    """
    email = os.environ.get(ENV_VAR_SLURM_EMAIL, None)
    if email is None:
        raise ValueError(
            "Notification email for Slurm not set. "
            f"Define the {ENV_VAR_SLURM_EMAIL} environment variable."
        )
    return email


def get_slurm_account() -> str:
    """
    Account to use to submit to slurm, defined in an environment variable.

    Raises:
         ValueError: if the environment variable is not set.
    """
    account = os.environ.get(ENV_VAR_SLURM_ACCOUNT, None)
    if account is None:
        raise ValueError(
            "Slurm Account not set. " f"Define the {account} environment variable."
        )
    return account


def get_logger(name: Optional[str] = None, level: Optional[str | int] = None) -> Logger:
    """Get a logger with a console handler.

    Args:
        name: Name of the logger.
        level: Logging level.
            Defaults to the value of the env variable OPTEXP_CONSOLE_LOGGING_LEVEL.
    """
    logger = logging.getLogger(__name__ if name is None else name)

    if not any(isinstance(x, logging.StreamHandler) for x in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level=get_console_logging_level() if level is None else level)
        formatter = logging.Formatter(LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(level=get_console_logging_level() if level is None else level)
    return logger


def set_logfile(path: Path, name: Optional[str] = None):
    handler = logging.FileHandler(path)
    handler.formatter = logging.Formatter(LOG_FMT)
    get_logger(name=name).addHandler(handler)
    return handler


def remove_loghandler(handler: logging.FileHandler, name: Optional[str] = None):
    get_logger(name=name).removeHandler(handler)
