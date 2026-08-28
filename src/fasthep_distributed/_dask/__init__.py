from __future__ import annotations

from fasthep_distributed._dask._htcondor import compute_with_htcondor
from fasthep_distributed._dask._local import compute_with_local_cluster
from fasthep_distributed._dask._slurm import compute_with_slurm

__all__ = [
    "compute_with_htcondor",
    "compute_with_local_cluster",
    "compute_with_slurm",
]
