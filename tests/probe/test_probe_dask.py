from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from hepflow.build_layout import BuildPaths

from fasthep_distributed.probe import (
    ProbeConfigError,
    format_dask_report,
    probe_config_from_mapping,
    run_dask_probe,
)
from fasthep_distributed.probe import _dask as dask_probe


def _config(tmp_path: Path, **dask: Any) -> Any:
    return probe_config_from_mapping(
        {
            "prefix": sys.prefix,
            "probe_root": str(tmp_path / "probes"),
            "imports": {"required": ["json"]},
            "paths": [{"name": "tmp", "path": str(tmp_path)}],
            "dask": {"retire_grace": 0, **dask},
        },
        base_dir=tmp_path,
    )


def test_dask_section_is_validated(tmp_path: Path) -> None:
    for raw, message in (
        ({"unknown": 1}, "unknown keys in dask"),
        ({"use": []}, "dask.use"),
        ({"retire_grace": -1}, "non-negative"),
    ):
        with pytest.raises(ProbeConfigError, match=message):
            _config(tmp_path, **raw)


def test_resolve_probe_execution_applies_cern_profile(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        use={"profiles": ["fasthep_distributed:dask_htcondor_cern"]},
        execution={
            # Restated: Flow resets a profile's backend/strategy (see below).
            "backend": "dask",
            "strategy": "htcondor",
            "resources": {"default": {"cpus": 1, "memory": "2GB"}},
            "config": {"queue": "espresso", "worker_start_timeout": 900},
        },
    )

    execution = dask_probe.resolve_probe_execution(config)

    assert execution["backend"] == "dask"
    assert execution["strategy"] == "htcondor"
    assert execution["config"]["scheduler_options"] == {"host": "fqdn", "port": 8786}
    assert execution["config"]["site_directives"]["MY.IsDaskWorker"] == "true"
    assert execution["config"]["queue"] == "espresso"
    assert execution["config"]["workers"] == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "fasthep-flow defect: normalize_global_execution defaults every layer to "
        "backend=local/strategy=default, so the workflow layer overrides a "
        "profile's backend and strategy"
    ),
)
def test_profile_backend_and_strategy_survive_workflow_layer(tmp_path: Path) -> None:
    config = _config(
        tmp_path, use={"profiles": ["fasthep_distributed:dask_htcondor_cern"]}
    )

    execution = dask_probe.resolve_probe_execution(config)

    assert (execution["backend"], execution["strategy"]) == ("dask", "htcondor")


@contextmanager
def _local_client(
    *, execution: dict[str, Any], build_paths: Any
) -> Iterator[tuple[Any, dict[str, Any]]]:
    distributed = pytest.importorskip("distributed")
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client, {"scheduler_options": {}, "pool_specs": {}}


def test_run_dask_probe_runs_checks_on_worker(tmp_path: Path) -> None:
    config = _config(tmp_path, execution={"backend": "dask"})

    report = run_dask_probe(config, client_factory=_local_client)

    assert report["error"] is None, report["error"]
    assert report["summary"] == {
        "verdict": "pass",
        "failed_required": [],
        "warnings": [],
    }
    [worker] = report["workers"]
    task_report = worker["task"]["report"]
    assert task_report["mode"] == "worker"
    assert task_report["summary"]["verdict"] == "pass"
    assert "scratch" in [check["name"] for check in task_report["checks"]]
    assert report["shutdown"]["retired"] is True
    assert set(report["timings"]) >= {"workers_connected_s", "tasks_s", "total_s"}
    probe_dir = Path(report["probe_dir"])
    written = json.loads((probe_dir / "dask-report.json").read_text(encoding="utf-8"))
    assert written["summary"]["verdict"] == "pass"
    assert (probe_dir / "execution.json").exists()
    assert "verdict: pass" in format_dask_report(report)


def test_run_dask_probe_records_startup_failure(tmp_path: Path) -> None:
    @contextmanager
    def failing_client(**_: Any) -> Iterator[tuple[Any, dict[str, Any]]]:
        raise TimeoutError("HTCondor workers did not start")
        yield  # pragma: no cover

    report = run_dask_probe(
        _config(tmp_path, execution={"backend": "dask"}),
        client_factory=failing_client,
    )

    assert report["error"] == "TimeoutError: HTCondor workers did not start"
    assert report["summary"]["verdict"] == "fail"
    assert report["summary"]["failed_required"] == ["cluster"]
    assert (Path(report["probe_dir"]) / "dask-report.json").exists()


class _FakeJob:
    job_id = "4242"


class _FakeClient:
    def __init__(self, workers: dict[str, Any]) -> None:
        self._workers = workers
        self.cluster = type("Cluster", (), {"workers": {"default-0": _FakeJob()}})()

    def scheduler_info(self) -> dict[str, Any]:
        return {"workers": self._workers}


def test_worker_records_check_worker_and_nanny_ports() -> None:
    config = {
        "pool_specs": {
            "default": {
                "job_kwargs": {"worker_extra_args": ["--worker-port", "10000:10100"]}
            }
        }
    }
    client = _FakeClient(
        {
            "tcp://10.0.0.1:10005": {
                "name": "default-0",
                "host": "10.0.0.1",
                "nanny": "tcp://10.0.0.1:10006",
            },
            "tcp://10.0.0.2:40000": {
                "name": "default-1",
                "host": "10.0.0.2",
                "nanny": "tcp://10.0.0.2:10007",
            },
        }
    )

    records = dask_probe._worker_records(client, config)

    assert [
        (r["name"], r["port"], r["nanny_port"], r["port_in_range"], r["job_id"])
        for r in records
    ] == [
        ("default-0", 10005, 10006, True, "4242"),
        ("default-1", 40000, 10007, False, None),
    ]


def test_summary_flags_port_range_and_missing_logs() -> None:
    report = {
        "error": None,
        "workers": [
            {
                "name": "default-0",
                "port_in_range": False,
                "task": {"report": {"summary": {"verdict": "pass"}}},
            }
        ],
        "condor": {"jobs": [{"job_id": "4242.0", "stderr_returned": False}]},
    }

    summary = dask_probe.summarise_dask_report(report)

    assert summary["verdict"] == "fail"
    assert summary["failed_required"] == ["worker.default-0.port_range"]
    assert summary["warnings"] == ["job.4242.0.worker_log_missing"]


def test_condor_facts_read_jobqueue_logs(tmp_path: Path) -> None:
    build_paths = BuildPaths(root=tmp_path)
    base = tmp_path / "execution" / "dask" / "htcondor"
    for kind in ("logs", "out", "err"):
        (base / kind).mkdir(parents=True)
    # Queue items of one submit share a cluster id and a log, with sub-ids 0, 1.
    (base / "logs" / "worker-4242.log").write_text(
        "000 (4242.000.000) 2026-10-01 10:00:00 Job submitted from host: <x>\n"
        "...\n"
        "000 (4242.001.000) 2026-10-01 10:00:00 Job submitted from host: <x>\n"
        "...\n"
        "001 (4242.000.000) 2026-10-01 10:01:10 Job executing on host: <y>\n"
        "...\n"
        "005 (4242.000.000) 2026-10-01 10:02:00 Job terminated.\n"
        "\t(1) Normal termination (return value 0)\n"
        "...\n",
        encoding="utf-8",
    )
    (base / "out" / "worker-4242.0.out").write_text("worker log\n", encoding="utf-8")

    # jobqueue reports "<cluster>.<proc>"; a bare cluster id expands the same way.
    for reported in ("4242.0", "4242"):
        facts = dask_probe._condor_facts(build_paths, [{"job_id": reported}])

        first, second = facts["jobs"]
        assert (first["job_id"], second["job_id"]) == ("4242.0", "4242.1")
        assert first["state"] == "terminated"
        assert first["return_value"] == 0
        assert first["times"] == {
            "submitted": "2026-10-01 10:00:00",
            "running": "2026-10-01 10:01:10",
            "terminated": "2026-10-01 10:02:00",
        }
        assert first["stdout_returned"] is True
        assert first["stderr_returned"] is False
        assert second["state"] == "submitted"
        assert second["stdout_returned"] is False
