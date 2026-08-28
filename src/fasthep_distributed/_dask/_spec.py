from __future__ import annotations

from typing import Any

from hepflow.backends.model import BackendSpec

from fasthep_distributed._dask._common import (
    normalise_dask_strategy,
    validate_supported_dask_pools,
)

DASK_BUILD_DIRECTORIES = (
    "execution/dask/htcondor/submit",
    "execution/dask/htcondor/logs",
    "execution/dask/htcondor/out",
    "execution/dask/htcondor/err",
    "debug/dask",
)


def validate_dask_execution(execution: dict[str, Any]) -> None:
    strategy = normalise_dask_strategy(execution)
    validate_supported_dask_pools(execution, strategy=strategy)


DASK_BACKEND_SPEC = BackendSpec(
    validate_execution=validate_dask_execution,
    build_directories=DASK_BUILD_DIRECTORIES,
)
