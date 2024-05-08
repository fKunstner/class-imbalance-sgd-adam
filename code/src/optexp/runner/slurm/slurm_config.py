from dataclasses import dataclass
from typing import Optional


@dataclass
class SlurmConfig:
    """Configuration for Slurm.

    Args:
        time: Time limit for the job in format "1-23:45" for 1 day, 23h, 45m
        mem: Requested memory limit for the job in format "12000M" for 12G
        cpus: Number of CPUs per task
        gpu: Whether to use a GPU, or the name of a specific GPU.
    """

    time: str
    mem: str
    cpus: int
    gpu: Optional[bool | str] = False
    n_gpu: Optional[int] = 1

    def __post_init__(self):
        # TODO: Add sanity checks for gpu, mem, time, cpus_per_task
        pass


TESTING = SlurmConfig(time="0-02:00", mem="8000M", cpus=1, gpu=True)
SMALL_GPU_QUARTER = SlurmConfig(time="0-00:15", mem="8000M", cpus=1, gpu="p100")
SMALL_GPU_HALF = SlurmConfig(time="0-00:30", mem="8000M", cpus=1, gpu="p100")
SMALL_GPU_1H = SlurmConfig(time="0-01:00", mem="16G", cpus=2, gpu="p100")
SMALL_GPU_1H_HALF = SlurmConfig(time="0-01:30", mem="16G", cpus=2, gpu="p100")
SMALL_GPU_2H = SlurmConfig(time="0-02:00", mem="16GM", cpus=2, gpu="p100")
SMALL_GPU_3H = SlurmConfig(time="0-3:00", mem="8000M", cpus=1, gpu="p100")
SMALL_GPU_4H = SlurmConfig(time="0-04:00", mem="16G", cpus=2, gpu="p100")
SMALL_GPU_6H = SlurmConfig(time="0-06:00", mem="16G", cpus=2, gpu="p100")
SMALL_GPU_12H = SlurmConfig(time="0-12:00", mem="8000M", cpus=1, gpu="p100")
SMALL_GPU_72H = SlurmConfig(time="0-72:00", mem="16G", cpus=2, gpu="p100")
DEFAULT_GPU_1H = SlurmConfig(time="0-01:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_2H = SlurmConfig(time="0-02:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_3H = SlurmConfig(time="0-03:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_4H = SlurmConfig(time="0-04:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_8H = SlurmConfig(time="0-08:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_12H = SlurmConfig(time="0-12:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_GPU_24H = SlurmConfig(time="0-24:00", mem="8000M", cpus=1, gpu="v100l")
DEFAULT_4_GPU_24H = SlurmConfig(time="0-24:00", mem="32G", cpus=4, gpu="v100l", n_gpu=4)
DEFAULT_4_GPU_72H = SlurmConfig(time="0-72:00", mem="32G", cpus=4, gpu="v100l", n_gpu=4)
DEFAULT_2_H100_24H = SlurmConfig(time="0-24:00", mem="32G", cpus=4, gpu="h100", n_gpu=2)
DEFAULT_4_H100_24H = SlurmConfig(time="0-24:00", mem="32G", cpus=4, gpu="h100", n_gpu=4)
VISION_GPU_12H = SlurmConfig(time="0-12:00", mem="32G", cpus=16, gpu="v100l")

VISION_A100_6H = SlurmConfig(time="0-6:00", mem="64G", cpus=16, gpu="a100")
VISION_A100_3H = SlurmConfig(time="0-3:00", mem="64G", cpus=16, gpu="a100")
VISION_GPU_3H = SlurmConfig(time="0-3:00", mem="64G", cpus=16, gpu=True)
VISION_GPU_5H = SlurmConfig(time="0-5:00", mem="64G", cpus=16, gpu=True)
VISION_V100_8H = SlurmConfig(time="0-8:00", mem="64G", cpus=16, gpu="v100l")
VISION_V100_12H = SlurmConfig(time="0-12:00", mem="64G", cpus=16, gpu="v100l")


DEFAULT_2_A100_24H = SlurmConfig(
    time="0-24:00", mem="32G", cpus=4, gpu="a100l", n_gpu=2
)
DEFAULT_2_A100_48H = SlurmConfig(
    time="0-48:00", mem="32G", cpus=4, gpu="a100l", n_gpu=2
)
DEFAULT_2_A100_60H = SlurmConfig(
    time="0-60:00", mem="128G", cpus=24, gpu="a100", n_gpu=2
)
DEFAULT_4_A100_24H = SlurmConfig(
    time="0-24:00", mem="32G", cpus=4, gpu="a100l", n_gpu=4
)
DEFAULT_2_H100_48H = SlurmConfig(time="0-48:00", mem="32G", cpus=4, gpu="h100", n_gpu=2)
DEFAULT_4_H100_48H = SlurmConfig(time="0-48:00", mem="32G", cpus=4, gpu="h100", n_gpu=4)
DEFAULT_4_A100_48H = SlurmConfig(
    time="0-48:00", mem="32G", cpus=4, gpu="a100l", n_gpu=4
)
DEFAULT_2_H100_72H = SlurmConfig(time="0-72:00", mem="32G", cpus=8, gpu="h100", n_gpu=2)
