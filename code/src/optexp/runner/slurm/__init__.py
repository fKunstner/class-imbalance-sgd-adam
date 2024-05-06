"""Slurm runner for OptExp."""

from . import slurm_config
from .sbatch_writers import make_jobarray_content, make_sbatch_header

__ALL__ = ["make_sbatch_header", "make_jobarray_content", "slurm_config"]
