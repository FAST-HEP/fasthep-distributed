"""
Standalone HTCondor worker probe.

Submits a plain HTCondor job, without Dask, that checks whether a worker can
run the configured Python environment and read the configured paths. Site
paths belong in the probe configuration, not in this package.
"""

from __future__ import annotations

from ._config import (
    CondorSpec,
    DaskProbeSpec,
    PathSpec,
    ProbeConfig,
    ProbeConfigError,
    load_probe_config,
    probe_config_from_mapping,
)
from ._dask import format_dask_report, run_dask_probe
from ._report import ProbeCollection, collect_probe, format_collection
from ._submit import ProbeBundle, ProbeError, SubmitResult, prepare_probe, submit_probe
from ._worker import SCHEMA

__all__ = [
    "SCHEMA",
    "CondorSpec",
    "DaskProbeSpec",
    "PathSpec",
    "ProbeBundle",
    "ProbeCollection",
    "ProbeConfig",
    "ProbeConfigError",
    "ProbeError",
    "SubmitResult",
    "collect_probe",
    "format_collection",
    "format_dask_report",
    "load_probe_config",
    "prepare_probe",
    "probe_config_from_mapping",
    "run_dask_probe",
    "submit_probe",
]
