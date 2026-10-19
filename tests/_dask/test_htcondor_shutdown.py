from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fasthep_distributed._dask._condor_log import parse_condor_log
from fasthep_distributed._dask._htcondor import normalize_dask_htcondor_config
from fasthep_distributed._dask._htcondor_shutdown import (
    RETIRE_WAIT,
    cluster_job_ids,
    shutdown_htcondor_workers,
)

CLUSTER = 123


def _event(code: str, text: str, *, proc: int = 0, time: str = "10:00:00") -> str:
    return f"{code} ({CLUSTER}.{proc:03d}.000) 10/19 {time} {text}\n...\n"


def _log(tmp_path: Path, *events: str) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / f"worker-{CLUSTER}.log").write_text("".join(events), encoding="utf-8")
    return logs


def _append(logs: Path, event: str) -> None:
    with (logs / f"worker-{CLUSTER}.log").open("a", encoding="utf-8") as stream:
        stream.write(event)


class FakeClient:
    def __init__(self, *, workers: int = 1, retire_error: Exception | None = None):
        self.workers = workers
        self.retire_error = retire_error
        self.calls: list[str] = []

    def scheduler_info(self) -> dict[str, Any]:
        return {"workers": {f"tcp://w{i}": {} for i in range(self.workers)}}

    def retire_workers(self, *, workers: list[str], close_workers: bool) -> None:
        self.calls.append("retire")
        assert close_workers is True
        if self.retire_error is not None:
            raise self.retire_error
        self.workers = 0

    def shutdown(self) -> None:
        self.calls.append("shutdown")
        self.workers = 0


class Clock:
    """Fake time: sleeping advances it, and may run a hook (e.g. finish a job)."""

    def __init__(self, hook: Any = None) -> None:
        self.now = 0.0
        self.sleeps = 0
        self.hook = hook

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1
        if self.hook is not None:
            self.hook(self)

    def monotonic(self) -> float:
        return self.now


def test_condor_log_records_output_transfer(tmp_path: Path) -> None:
    logs = _log(
        tmp_path,
        _event("000", "Job submitted"),
        _event("001", "Job executing"),
        _event("040", "Started transferring output files", time="10:01:00"),
        _event("040", "Finished transferring output files", time="10:01:05"),
        _event("005", "Job terminated.\n\t(1) Normal termination (return value 0)"),
    )

    state = parse_condor_log(logs / f"worker-{CLUSTER}.log", cluster=CLUSTER)[0]

    assert state.state == "terminated"
    assert state.return_value == 0
    assert state.output_transferred is True
    assert state.times["output_transferred"] == "10/19 10:01:05"


def test_condor_log_ignores_input_transfer_events(tmp_path: Path) -> None:
    logs = _log(
        tmp_path,
        _event("000", "Job submitted"),
        _event("040", "Finished transferring input files"),
        _event("005", "Job terminated."),
    )

    state = parse_condor_log(logs / f"worker-{CLUSTER}.log", cluster=CLUSTER)[0]

    assert state.output_transferred is False


def test_shutdown_returns_at_once_when_jobs_already_exited(tmp_path: Path) -> None:
    logs = _log(
        tmp_path,
        _event("000", "Job submitted"),
        _event("001", "Job executing"),
        _event("040", "Finished transferring output files"),
        _event("005", "Job terminated."),
    )
    clock = Clock()
    client = FakeClient()

    summary = shutdown_htcondor_workers(
        client,
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=120,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert summary["retired"] is True
    assert summary["jobs_exited"] is True
    assert summary["method"] == "retire_workers"
    assert summary["jobs"][f"{CLUSTER}.0"]["output_transferred"] is True
    assert client.calls == ["retire"]
    assert clock.sleeps == 0


def test_shutdown_waits_until_running_job_exits(tmp_path: Path) -> None:
    logs = _log(
        tmp_path, _event("000", "Job submitted"), _event("001", "Job executing")
    )

    def finish(clock: Clock) -> None:
        if clock.sleeps == 2:
            _append(logs, _event("040", "Finished transferring output files"))
            _append(logs, _event("005", "Job terminated."))

    clock = Clock(finish)

    summary = shutdown_htcondor_workers(
        FakeClient(),
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=120,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert summary["jobs_exited"] is True
    assert clock.sleeps == 2
    assert summary["jobs"][f"{CLUSTER}.0"]["state"] == "terminated"
    assert summary["jobs"][f"{CLUSTER}.0"]["output_transferred"] is True


def test_shutdown_does_not_wait_for_jobs_that_never_started(tmp_path: Path) -> None:
    logs = _log(tmp_path, _event("000", "Job submitted"))
    clock = Clock()

    summary = shutdown_htcondor_workers(
        FakeClient(workers=0),
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=120,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert summary["jobs_exited"] is True
    assert clock.sleeps == 0
    assert summary["jobs"][f"{CLUSTER}.0"]["state"] == "submitted"


def test_shutdown_stops_scheduler_when_workers_refuse_to_retire(
    tmp_path: Path,
) -> None:
    logs = _log(
        tmp_path, _event("000", "Job submitted"), _event("001", "Job executing")
    )
    client = FakeClient()
    client.retire_workers = lambda **_: client.calls.append("retire")  # type: ignore[method-assign]
    clock = Clock()

    summary = shutdown_htcondor_workers(
        client,
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    # Retire left the worker connected, so after RETIRE_WAIT the scheduler is
    # stopped; the job never exits in this fake, so the grace period runs out.
    assert client.calls == ["retire", "shutdown"]
    assert summary["method"] == "shutdown"
    assert summary["jobs_exited"] is False
    assert clock.now >= min(RETIRE_WAIT, 60)


def test_shutdown_records_retire_failure_and_falls_back(tmp_path: Path) -> None:
    logs = _log(
        tmp_path,
        _event("000", "Job submitted"),
        _event("001", "Job executing"),
        _event("005", "Job terminated."),
    )
    client = FakeClient(retire_error=RuntimeError("boom"))

    summary = shutdown_htcondor_workers(
        client,
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=10,
        sleep=Clock().sleep,
        monotonic=Clock().monotonic,
    )

    assert summary["retired"] is False
    assert "boom" in summary["error"]
    assert summary["method"] == "shutdown"
    assert client.calls == ["retire", "shutdown"]


def test_shutdown_with_zero_grace_does_not_wait(tmp_path: Path) -> None:
    logs = _log(
        tmp_path, _event("000", "Job submitted"), _event("001", "Job executing")
    )
    clock = Clock()

    summary = shutdown_htcondor_workers(
        FakeClient(),
        logs_dir=logs,
        reported_job_ids=[f"{CLUSTER}.0"],
        grace=0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert summary["jobs_exited"] is False
    assert clock.sleeps == 0


def test_cluster_job_ids_tolerates_missing_workers() -> None:
    class Job:
        job_id = "7.0"

    class Cluster:
        def __init__(self) -> None:
            self.workers = {"a": Job(), "b": object()}

    assert cluster_job_ids(Cluster()) == ["7.0"]
    assert cluster_job_ids(object()) == []


@pytest.mark.parametrize(
    ("config", "expected"),
    [({}, 120.0), ({"worker_shutdown_grace": 30}, 30.0), ({"worker_shutdown_grace": 0}, 0.0)],
)
def test_worker_shutdown_grace_is_configurable(
    config: dict[str, Any], expected: float
) -> None:
    normalized = normalize_dask_htcondor_config(
        {
            "backend": "dask",
            "strategy": "htcondor",
            "resources": {"default": {"cpus": 1}},
            "config": {"workers": 1, **config},
        }
    )

    assert normalized["worker_shutdown_grace"] == expected


@pytest.mark.parametrize("value", [-1, "soon"])
def test_worker_shutdown_grace_is_validated(value: Any) -> None:
    with pytest.raises(ValueError, match="worker_shutdown_grace"):
        normalize_dask_htcondor_config(
            {
                "backend": "dask",
                "strategy": "htcondor",
                "resources": {"default": {"cpus": 1}},
                "config": {"workers": 1, "worker_shutdown_grace": value},
            }
        )
