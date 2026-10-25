"""
Dask worker probe: the worker checks, run as a task on HTCondor Dask workers.

Workers are started through :func:`fasthep_distributed._dask._htcondor.htcondor_client`,
the same start-up path used for plan execution. If the plain HTCondor probe
passes and this one fails, the problem is the network or Dask, not storage or
the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fasthep_distributed._dask._condor_log import (
    job_ids,
    job_sort_key,
    job_state_record,
    split_job_id,
)
from fasthep_distributed._dask._htcondor_shutdown import shutdown_htcondor_workers

from ._config import ProbeConfig
from ._submit import Runner, _probe_environment, _write_json, prepare_probe_dir
from ._worker import FAIL, PASS, SCHEMA, WARN

DASK_REPORT_FILE = "dask-report.json"
def run_dask_probe(
    config: ProbeConfig,
    *,
    now: datetime | None = None,
    runner: Runner = subprocess.run,
    client_factory: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """
    Start HTCondor Dask workers, run the worker checks on each, and shut down.

    Always writes ``dask-report.json`` into a new probe directory, also when
    worker start-up fails, and returns the report.
    """

    if config.dask is None:
        raise ValueError("probe config has no 'dask' section")
    from hepflow.build_layout import BuildPaths  # noqa: PLC0415

    from fasthep_distributed._dask._htcondor import (  # noqa: PLC0415
        _htcondor_execution_paths,
        htcondor_client,
    )

    probe_id, probe_dir, submit_facts = prepare_probe_dir(
        config, now=now, runner=runner
    )
    execution = resolve_probe_execution(config)
    _write_json(probe_dir / "execution.json", execution)
    build_paths = BuildPaths(root=probe_dir)
    worker_config = config.worker_config(probe_id)
    start_client = client_factory or htcondor_client

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "dask",
        "probe_id": probe_id,
        "probe_dir": str(probe_dir),
        "execution": execution,
        "submit": submit_facts,
        "timings": {},
        "scheduler": None,
        "workers": [],
        "error": None,
    }
    started = time.monotonic()
    try:
        with start_client(execution=execution, build_paths=build_paths) as (
            client,
            htcondor_config,
        ):
            report["timings"]["workers_connected_s"] = _elapsed(started)
            report["scheduler"] = {
                "address": client.scheduler.address,
                "dashboard": client.dashboard_link,
                "options": htcondor_config.get("scheduler_options", {}),
            }
            workers = _worker_records(client, htcondor_config)
            report["workers"] = workers
            task_started = time.monotonic()
            futures = {
                record["address"]: client.submit(
                    run_worker_checks,
                    worker_config,
                    submit_facts,
                    config.dask.task_timeout,
                    workers=[record["address"]],
                    allow_other_workers=False,
                    pure=False,
                )
                for record in workers
            }
            for record in workers:
                record["task"] = futures[record["address"]].result(
                    timeout=config.dask.task_timeout + 60
                )
            report["timings"]["tasks_s"] = _elapsed(task_started)
            # Results are copied into the records. Release them on the workers,
            # otherwise retire_workers refuses ("unique data could not be
            # moved") and the jobs only end through condor_rm, which returns no
            # stdout/stderr.
            client.cancel(list(futures.values()))
            killed = _kill_workers(workers, config.dask.kill_workers, sleep=sleep)
            if killed:
                report["killed_jobs"] = killed
            killed_set = set(killed)
            remaining = [r for r in workers if r.get("job_id") not in killed_set]
            retire_started = time.monotonic()
            report["shutdown"] = _retire_workers(
                client,
                remaining,
                logs_dir=_htcondor_execution_paths(build_paths)["logs"],
                grace=config.dask.retire_grace,
                sleep=sleep,
            )
            report["timings"]["retire_s"] = _elapsed(retire_started)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["timings"]["total_s"] = _elapsed(started)
    report["condor"] = _condor_facts(build_paths, report["workers"])
    report["summary"] = summarise_dask_report(report)
    _write_json(probe_dir / DASK_REPORT_FILE, report)
    return report


def resolve_probe_execution(config: ProbeConfig) -> dict[str, Any]:
    """Resolve ``dask.use`` profiles and ``dask.execution`` the way Flow does."""

    from hepflow.compiler.context import CompileContext  # noqa: PLC0415
    from hepflow.compiler.execution import resolve_workflow_execution  # noqa: PLC0415

    assert config.dask is not None
    workflow = {"use": config.dask.use, "execution": config.dask.execution}
    context = CompileContext.from_workflow_path(config.dask.base_dir / "probe.yaml")
    resolved = resolve_workflow_execution(workflow, context=context)["execution"]
    execution: dict[str, Any] = dict(resolved)
    exec_config = dict(execution.get("config") or {})
    if (
        not execution.get("pools")
        and exec_config.get("workers") is None
        and exec_config.get("n_workers") is None
    ):
        exec_config["workers"] = 1
    execution["config"] = exec_config
    return execution


def run_worker_checks(
    worker_config: dict[str, Any],
    submit_facts: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """
    Dask task: run ``_worker.py`` in a subprocess on the worker.

    A subprocess keeps the checks identical to the plain HTCondor probe and
    runs imports on a main thread (``dask_jobqueue`` installs signal handlers
    on import, which fails in Dask's worker threads).
    """

    from fasthep_distributed.probe import _worker  # noqa: PLC0415

    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    with tempfile.TemporaryDirectory(prefix="fasthep-probe-", dir=scratch) as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "config.json"
        facts_path = tmp_path / "submit_facts.json"
        output = tmp_path / "report.json"
        config_path.write_text(json.dumps(worker_config), encoding="utf-8")
        facts_path.write_text(json.dumps(submit_facts), encoding="utf-8")
        cmd = [
            sys.executable,
            str(Path(_worker.__file__)),
            "--mode",
            "worker",
            "--config",
            str(config_path),
            "--submit-facts",
            str(facts_path),
            "--output",
            str(output),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=scratch or tmp,
                env=_probe_environment(),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": None, "error": f"timeout after {timeout:g}s"}
        result: dict[str, Any] = {
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
        if output.exists():
            result["report"] = json.loads(output.read_text(encoding="utf-8"))
        return result


def summarise_dask_report(report: dict[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    warnings: list[str] = []
    if report.get("error"):
        failed.append("cluster")
    workers = report.get("workers") or []
    if not workers and not report.get("error"):
        failed.append("workers.connected")
    for record in workers:
        name = record.get("name") or record.get("address")
        task = record.get("task") or {}
        verdict = (task.get("report") or {}).get("summary", {}).get("verdict")
        if verdict != PASS:
            failed.append(f"worker.{name}.checks")
        if record.get("port_in_range") is False:
            failed.append(f"worker.{name}.port_range")
    for job in (report.get("condor") or {}).get("jobs", []):
        # The Dask worker logs to stderr; an empty stdout is normal.
        if not job.get("stderr_returned"):
            warnings.append(f"job.{job.get('job_id')}.worker_log_missing")
    return {
        "verdict": FAIL if failed else PASS,
        "failed_required": failed,
        "warnings": warnings,
    }


def format_dask_report(report: dict[str, Any]) -> str:
    lines = [f"dask probe {report.get('probe_id')}  {report.get('probe_dir')}"]
    scheduler = report.get("scheduler") or {}
    if scheduler:
        lines.append(
            f"scheduler: {scheduler.get('address')}  "
            f"dashboard: {scheduler.get('dashboard')}"
        )
    if report.get("error"):
        lines.append(f"error: {report['error']}")
    for record in report.get("workers", []):
        task = record.get("task") or {}
        worker_report = task.get("report") or {}
        verdict = worker_report.get("summary", {}).get("verdict", FAIL)
        lines.append(
            f"worker {record.get('name')}  {record.get('address')}  "
            f"job {record.get('job_id')}  port in range: "
            f"{record.get('port_in_range')}  checks: {verdict}"
        )
        for check in worker_report.get("checks", []):
            message = check.get("error") or (check.get("detail") or {}).get("warning")
            suffix = f"  {message}" if message else ""
            lines.append(f"  {check.get('status', '?'):<5} {check.get('name')}{suffix}")
        if not worker_report:
            lines.append(f"  no report: {task.get('error') or task.get('stderr_tail')}")
    for job in (report.get("condor") or {}).get("jobs", []):
        lines.append(
            f"condor job {job.get('job_id')}: {job.get('state')}  "
            f"times: {job.get('times')}  worker log returned: "
            f"{job.get('stderr_returned')}"
        )
    shutdown = report.get("shutdown") or {}
    if shutdown:
        lines.append(
            f"shutdown: retired={shutdown.get('retired')}  "
            f"jobs exited before close: {shutdown.get('jobs_exited')}"
        )
    lines.append(f"timings: {report.get('timings')}")
    summary = report.get("summary") or {}
    for name in summary.get("warnings", []):
        lines.append(f"{WARN}: {name}")
    lines.append(f"verdict: {summary.get('verdict')}")
    return "\n".join(lines) + "\n"


def _worker_records(
    client: Any, htcondor_config: dict[str, Any]
) -> list[dict[str, Any]]:
    jobs = getattr(getattr(client, "cluster", None), "workers", {}) or {}
    port_range = _port_range(htcondor_config)
    records = []
    for address, info in sorted(client.scheduler_info()["workers"].items()):
        name = str(info.get("name"))
        job = jobs.get(name)
        nanny = info.get("nanny")
        ports = [_port(address)]
        if nanny:
            ports.append(_port(str(nanny)))
        records.append(
            {
                "address": address,
                "name": name,
                "host": info.get("host"),
                "port": ports[0],
                "nanny": nanny,
                "nanny_port": ports[1] if nanny else None,
                "port_in_range": (
                    None
                    if port_range is None or None in ports
                    else all(
                        port_range[0] <= port <= port_range[1]
                        for port in ports
                        if port is not None
                    )
                ),
                "job_id": getattr(job, "job_id", None),
            }
        )
    return records


def _kill_workers(
    workers: list[dict[str, Any]],
    count: int,
    *,
    sleep: Callable[[float], None],
) -> list[str]:
    """
    ``condor_rm`` the first ``count`` workers (by job id) and return their job ids.

    Used to see what a worker leaves behind when its job is removed instead of
    exiting on its own.
    """

    if count <= 0:
        return []
    job_ids = sorted(
        (r["job_id"] for r in workers if r.get("job_id")), key=job_sort_key
    )[:count]
    for job_id in job_ids:
        subprocess.run(
            ["condor_rm", job_id],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    sleep(10.0)
    return job_ids


def _retire_workers(
    client: Any,
    workers: list[dict[str, Any]],
    *,
    logs_dir: Path,
    grace: float,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Shut the workers down the way production does and report the outcome."""

    return shutdown_htcondor_workers(
        client,
        logs_dir=logs_dir,
        reported_job_ids=[str(r["job_id"]) for r in workers if r.get("job_id")],
        grace=grace,
        sleep=sleep,
    )


def _condor_facts(build_paths: Any, workers: list[dict[str, Any]]) -> dict[str, Any]:
    from fasthep_distributed._dask._htcondor import (  # noqa: PLC0415
        _htcondor_execution_paths,
    )

    paths = _htcondor_execution_paths(build_paths)
    ids = sorted(
        job_ids(paths["logs"], [r["job_id"] for r in workers if r.get("job_id")]),
        key=job_sort_key,
    )
    jobs = []
    for job_id in ids:
        cluster, proc = split_job_id(job_id)
        state = job_state_record(paths["logs"], job_id)
        stdout = paths["out"] / f"worker-{cluster}.{proc}.out"
        stderr = paths["err"] / f"worker-{cluster}.{proc}.err"
        jobs.append(
            {
                "job_id": job_id,
                "state": state.state if state else "unknown",
                "return_value": state.return_value if state else None,
                "detail": state.detail if state else None,
                "times": state.times if state else {},
                "stdout_returned": stdout.exists() and stdout.stat().st_size > 0,
                "stderr_returned": stderr.exists() and stderr.stat().st_size > 0,
            }
        )
    return {"jobs": jobs, "paths": {key: str(value) for key, value in paths.items()}}


def _port_range(htcondor_config: dict[str, Any]) -> tuple[int, int] | None:
    pools = htcondor_config.get("pool_specs") or {}
    for spec in pools.values():
        args = list(spec.get("job_kwargs", {}).get("worker_extra_args") or [])
        if "--worker-port" in args:
            raw = str(args[args.index("--worker-port") + 1])
            low, _, high = raw.partition(":")
            return int(low), int(high or low)
    return None


def _port(address: str) -> int | None:
    _, _, port = address.rpartition(":")
    return int(port) if port.isdigit() else None


def _elapsed(start: float) -> float:
    return round(time.monotonic() - start, 3)
