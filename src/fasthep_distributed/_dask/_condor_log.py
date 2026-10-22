"""
Parsing of the HTCondor event logs written for Dask worker jobs.

Shared by the HTCondor backend (graceful shutdown) and the worker probe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EVENT_PATTERN = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\S+ \S+)(.*)$")
_RETURN_VALUE_PATTERN = re.compile(r"return value (-?\d+)")
_SIGNAL_PATTERN = re.compile(r"signal (\d+)")

_EVENT_STATES = {
    "000": "submitted",
    "001": "running",
    "004": "evicted",
    "005": "terminated",
    "009": "aborted",
    "012": "held",
    "013": "released",
}

TERMINAL_STATES = {"terminated", "aborted"}


@dataclass(slots=True)
class CondorJobState:
    state: str = "unknown"
    return_value: int | None = None
    detail: str | None = None
    # First timestamp of each state, as written in the event log.
    times: dict[str, str] = field(default_factory=dict)
    # True once HTCondor logged "Finished transferring output files".
    output_transferred: bool = False


def parse_condor_log(path: Path, *, cluster: int) -> dict[int, CondorJobState]:
    """Return the latest Condor state for each process of ``cluster``."""

    if not path.exists():
        return {}
    states: dict[int, CondorJobState] = {}
    current: CondorJobState | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EVENT_PATTERN.match(line)
        if match is not None:
            current = None
            code, event_cluster, process, timestamp, rest = match.groups()
            if int(event_cluster) != cluster:
                continue
            if code == "040":
                _record_transfer_event(states, int(process), timestamp, rest)
                continue
            if code not in _EVENT_STATES:
                continue
            previous = states.get(int(process))
            current = CondorJobState(
                state=_EVENT_STATES[code],
                times=dict(previous.times) if previous else {},
                output_transferred=previous.output_transferred if previous else False,
            )
            current.times.setdefault(current.state, timestamp)
            states[int(process)] = current
            continue
        if current is None or line.strip() in {"", "..."}:
            continue
        text = line.strip()
        if current.state == "terminated" and current.detail is None:
            current.detail = text
            value = _RETURN_VALUE_PATTERN.search(text)
            if value is not None:
                current.return_value = int(value.group(1))
            elif _SIGNAL_PATTERN.search(text):
                current.return_value = None
        elif current.state in {"held", "aborted"} and current.detail is None:
            current.detail = text
    return states


def _record_transfer_event(
    states: dict[int, CondorJobState], process: int, timestamp: str, rest: str
) -> None:
    if "output files" not in rest:
        return
    state = states.setdefault(process, CondorJobState())
    if "Finished" in rest:
        state.output_transferred = True
        state.times.setdefault("output_transferred", timestamp)
    elif "Started" in rest:
        state.times.setdefault("output_transfer_started", timestamp)


def split_job_id(job_id: str) -> tuple[str, str]:
    """Split ``<cluster>.<proc>``; HTCondor numbers the queue items of one submit."""

    cluster, _, proc = str(job_id).partition(".")
    return cluster, proc or "0"


def job_sort_key(job_id: str) -> tuple[int, int]:
    cluster, proc = split_job_id(job_id)
    return (
        int(cluster) if cluster.isdigit() else 0,
        int(proc) if proc.isdigit() else 0,
    )


def job_ids(logs_dir: Path, reported: list[Any]) -> set[str]:
    """
    All ``<cluster>.<proc>`` ids of the cluster's jobs.

    Ids reported by the cluster are used as they are; a bare cluster id (and
    each cluster found in the Condor logs) expands to every process in its log.
    """

    ids: set[str] = set()
    clusters: set[str] = set()
    for job_id in reported:
        cluster, _, proc = str(job_id).partition(".")
        if proc:
            ids.add(f"{cluster}.{proc}")
        clusters.add(cluster)
    clusters |= clusters_from_logs(logs_dir)
    for cluster in clusters:
        if not cluster.isdigit():
            continue
        states = parse_condor_log(
            logs_dir / f"worker-{cluster}.log", cluster=int(cluster)
        )
        ids.update(f"{cluster}.{proc}" for proc in states)
        if not states and not any(i.startswith(f"{cluster}.") for i in ids):
            ids.add(f"{cluster}.0")
    return ids


def clusters_from_logs(logs_dir: Path) -> set[str]:
    if not logs_dir.is_dir():
        return set()
    return {
        path.stem.removeprefix("worker-")
        for path in logs_dir.glob("worker-*.log")
        if path.stem.removeprefix("worker-").isdigit()
    }


def job_state_record(logs_dir: Path, job_id: str) -> Any:
    cluster, proc = split_job_id(job_id)
    if not cluster.isdigit() or not proc.isdigit():
        return None
    states = parse_condor_log(logs_dir / f"worker-{cluster}.log", cluster=int(cluster))
    return states.get(int(proc))


def job_state(logs_dir: Path, job_id: str) -> str:
    state = job_state_record(logs_dir, job_id)
    return state.state if state else "unknown"
