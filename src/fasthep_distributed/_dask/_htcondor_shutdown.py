"""
Graceful shutdown of HTCondor Dask workers.

``cluster.close()`` (dask-jobqueue) removes worker jobs with ``condor_rm``. A
removed job returns no stdout/stderr and, more importantly, no output files:
HTCondor sends ``output_destination`` files only when the job exits on its own.
Workers are therefore asked to stop first and the jobs are given time to finish
(including the output transfer) before the cluster is closed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._condor_log import (
    job_ids,
    job_sort_key,
    job_state_record,
)

DEFAULT_WORKER_SHUTDOWN_GRACE = 120.0
# How long workers get to leave after retire_workers before the scheduler is
# shut down instead (a worker holding unique data refuses to retire).
RETIRE_WAIT = 30.0
POLL_SECONDS = 5.0
# Only jobs that have started can still send output; idle or held jobs are not
# waited for (the cluster removes them afterwards).
_RUNNING_STATES = {"running", "evicted", "released"}


def cluster_job_ids(cluster: Any) -> list[str]:
    """Job ids of the cluster's worker jobs, without failing on odd clusters."""

    try:
        jobs = list((getattr(cluster, "workers", None) or {}).values())
    except Exception:
        return []
    return [str(job.job_id) for job in jobs if getattr(job, "job_id", None)]


def shutdown_htcondor_workers(
    client: Any,
    *,
    logs_dir: Path,
    reported_job_ids: list[str],
    grace: float = DEFAULT_WORKER_SHUTDOWN_GRACE,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """
    Stop the workers, then wait up to ``grace`` seconds for their jobs to exit.

    Never raises: problems are recorded in the returned summary, so that the
    caller can still close the client and cluster. ``retired`` says whether the
    workers were asked to stop; ``jobs_exited`` whether no started job was
    still running when the wait ended; ``jobs`` gives each job's last state and
    whether its output transfer finished.
    """

    summary: dict[str, Any] = {
        "retired": False,
        "jobs_exited": False,
        "method": "retire_workers",
        "grace_s": grace,
    }
    started = monotonic()
    addresses = _connected_workers(client)
    if addresses:
        try:
            client.retire_workers(workers=addresses, close_workers=True)
            summary["retired"] = True
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["method"] = "shutdown"
            _shutdown_scheduler(client, summary)
    else:
        summary["retired"] = True

    ids = sorted(job_ids(logs_dir, reported_job_ids), key=job_sort_key)
    shutdown_tried = summary["method"] == "shutdown"
    deadline = started + grace
    while True:
        if ids and not any(_state(logs_dir, i) in _RUNNING_STATES for i in ids):
            summary["jobs_exited"] = True
            break
        if not ids and not _connected_workers(client):
            # No job logs to follow (not HTCondor, or no job reported).
            summary["jobs_exited"] = True
            break
        now = monotonic()
        if now >= deadline:
            break
        if not shutdown_tried and now - started >= min(RETIRE_WAIT, grace):
            if _connected_workers(client):
                # Workers that hold data refuse to retire; results have already
                # been gathered, so stop the scheduler and with it the workers.
                summary["method"] = "shutdown"
                _shutdown_scheduler(client, summary)
            shutdown_tried = True
        sleep(POLL_SECONDS)
    summary["waited_s"] = round(monotonic() - started, 3)
    summary["jobs"] = {i: _job_summary(logs_dir, i) for i in ids}
    return summary


def _connected_workers(client: Any) -> list[str]:
    try:
        return list(client.scheduler_info().get("workers", {}))
    except Exception:
        return []


def _shutdown_scheduler(client: Any, summary: dict[str, Any]) -> None:
    try:
        client.shutdown()
    except Exception as exc:
        summary.setdefault("error", f"shutdown: {type(exc).__name__}: {exc}")


def _state(logs_dir: Path, job_id: str) -> str:
    record = job_state_record(logs_dir, job_id)
    return record.state if record else "unknown"


def _job_summary(logs_dir: Path, job_id: str) -> dict[str, Any]:
    record = job_state_record(logs_dir, job_id)
    if record is None:
        return {"state": "unknown", "output_transferred": False}
    return {
        "state": record.state,
        "return_value": record.return_value,
        "output_transferred": record.output_transferred,
        "detail": record.detail,
    }
