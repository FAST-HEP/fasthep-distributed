from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fasthep_distributed.probe import (
    ProbeError,
    prepare_probe,
    probe_config_from_mapping,
    submit_probe,
)
from fasthep_distributed.probe import _submit as submit
from fasthep_distributed.probe import _worker as worker

NOW = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)


def _config(tmp_path: Path, **extra: Any) -> Any:
    return probe_config_from_mapping(
        {
            "prefix": sys.prefix,
            "probe_root": str(tmp_path / "probes"),
            "imports": {"required": ["json"]},
            **extra,
        }
    )


class FakeRunner:
    def __init__(
        self,
        *,
        failed_required: list[str] | None = None,
        submit_stdout: str = "1 job(s) submitted to cluster 4242.\n",
        submit_returncode: int = 0,
    ) -> None:
        self.calls: list[list[str]] = []
        self.failed_required = failed_required or []
        self.submit_stdout = submit_stdout
        self.submit_returncode = submit_returncode

    def __call__(
        self, cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[:2] == ["git", "-C"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
        if cmd[0] == "condor_version":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="$CondorVersion: 24.0.0 $\n", stderr=""
            )
        if cmd[0] == "condor_submit":
            return subprocess.CompletedProcess(
                cmd, self.submit_returncode, stdout=self.submit_stdout, stderr=""
            )
        if len(cmd) > 1 and cmd[1] == submit.WORKER_FILE:
            report = {
                "schema": worker.SCHEMA,
                "mode": "submit",
                "checks": [],
                "summary": {
                    "verdict": worker.FAIL if self.failed_required else worker.PASS,
                    "failed_required": self.failed_required,
                    "warnings": ["paths.eos"],
                },
            }
            (Path(kwargs["cwd"]) / submit.SUBMIT_FACTS_FILE).write_text(
                json.dumps(report), encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")


def test_prepare_writes_probe_directory(tmp_path: Path) -> None:
    runner = FakeRunner()

    bundle = prepare_probe(_config(tmp_path), now=NOW, runner=runner)

    assert bundle.probe_id == "20260929T153000Z-abc1234"
    assert bundle.probe_dir == tmp_path / "probes" / bundle.probe_id
    assert sorted(path.name for path in bundle.probe_dir.iterdir()) == [
        "_worker.py",
        "config.json",
        "probe.sh",
        "probe.sub",
        "submit_facts.json",
    ]
    assert os.access(bundle.executable, os.X_OK)
    config = json.loads((bundle.probe_dir / "config.json").read_text(encoding="utf-8"))
    assert config["probe_id"] == bundle.probe_id
    assert config["prefix"] == sys.prefix
    facts = json.loads(
        (bundle.probe_dir / "submit_facts.json").read_text(encoding="utf-8")
    )
    assert facts["submit_context"]["condor_version"] == "$CondorVersion: 24.0.0 $"
    assert facts["summary"]["warnings"] == ["paths.eos"]
    assert not any(call[0] == "condor_submit" for call in runner.calls)


def test_prepare_never_reuses_a_probe_directory(tmp_path: Path) -> None:
    first = prepare_probe(_config(tmp_path), now=NOW, runner=FakeRunner())
    second = prepare_probe(_config(tmp_path), now=NOW, runner=FakeRunner())

    assert second.probe_dir == first.probe_dir.with_name(f"{first.probe_id}-1")


def test_prepare_fails_on_submit_host_failures(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match=r"submit host: python\.imports"):
        prepare_probe(
            _config(tmp_path),
            now=NOW,
            runner=FakeRunner(failed_required=["python.imports"]),
        )


def test_prepare_requires_existing_prefix(tmp_path: Path) -> None:
    config = _config(tmp_path, prefix=str(tmp_path / "missing"))

    with pytest.raises(ProbeError, match="does not exist"):
        prepare_probe(config, now=NOW, runner=FakeRunner())


def test_submit_description_preserves_invariants(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        condor={
            "flavour": "espresso",
            "request_disk": "1GB",
            "count": 3,
            "extra_directives": {"+AccountingGroup": '"group_u_CMS.u_zh"'},
        },
    )

    text = submit.render_submit_description(config, tmp_path / "p")

    lines = text.splitlines()
    assert f"executable = {tmp_path / 'p' / 'probe.sh'}" in lines
    assert f"initialdir = {tmp_path / 'p'}" in lines
    assert "transfer_executable = False" in lines
    assert 'transfer_output_files = ""' in lines
    assert "transfer_input_files = config.json, submit_facts.json, _worker.py" in lines
    assert "output = stdout.$(ProcId).json" in lines
    assert "error = stderr.$(ProcId).log" in lines
    assert "log = condor.log" in lines
    assert "stream_output = True" in lines
    assert "stream_error = True" in lines
    assert "request_disk = 1GB" in lines
    assert '+JobFlavour = "espresso"' in lines
    assert '+AccountingGroup = "group_u_CMS.u_zh"' in lines
    assert lines[-1] == "queue 3"
    assert not any("x509" in line.lower() for line in lines)


def _stage_executable(tmp_path: Path, prefix: str, config: dict[str, Any]) -> Path:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    shutil.copy2(submit.WORKER_SCRIPT, job_dir / submit.WORKER_FILE)
    (job_dir / submit.CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    script = job_dir / submit.EXECUTABLE_FILE
    script.write_text(
        submit.render_executable(prefix=Path(prefix), probe_id="probe-1"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return job_dir


def test_executable_emits_fallback_report_without_interpreter(tmp_path: Path) -> None:
    job_dir = _stage_executable(tmp_path, str(tmp_path / "no env"), {})

    completed = subprocess.run(
        ["sh", "probe.sh", "7", "0"],
        cwd=job_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 127
    report = json.loads(completed.stdout)
    assert report["schema"] == worker.SCHEMA
    assert report["job"] == {"cluster": 7, "process": 0}
    assert report["checks"][0]["name"] == "python.interpreter"
    assert report["checks"][0]["status"] == worker.FAIL
    assert "missing or not executable" in report["checks"][0]["error"]
    assert report["summary"]["verdict"] == worker.FAIL


def test_executable_runs_worker_with_interpreter(tmp_path: Path) -> None:
    job_dir = _stage_executable(
        tmp_path,
        sys.prefix,
        {
            "probe_id": "probe-1",
            "prefix": sys.prefix,
            "imports": {"required": ["json"]},
            "paths": [{"name": "job", "path": str(tmp_path), "required": True}],
        },
    )
    env = {**os.environ, "_CONDOR_SCRATCH_DIR": str(job_dir)}

    completed = subprocess.run(
        ["sh", "probe.sh", "7", "0"],
        cwd=job_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    assert report["mode"] == "worker"
    assert report["job"] == {"cluster": 7, "process": 0}
    assert report["summary"]["verdict"] == worker.PASS
    assert report["host"]["env"]["PYTHONNOUSERSITE"] == "1"


def test_submit_records_cluster(tmp_path: Path) -> None:
    runner = FakeRunner()
    bundle = prepare_probe(_config(tmp_path), now=NOW, runner=runner)

    result = submit_probe(bundle, runner=runner)

    assert (result.cluster, result.count) == (4242, 1)
    record = json.loads((bundle.probe_dir / "submit.json").read_text(encoding="utf-8"))
    assert record["cluster"] == 4242
    assert record["probe_id"] == bundle.probe_id
    with pytest.raises(ProbeError, match="already submitted"):
        submit_probe(bundle, runner=runner)


def test_submit_reports_condor_submit_failure(tmp_path: Path) -> None:
    runner = FakeRunner(submit_stdout="ERROR: no schedd\n", submit_returncode=1)
    bundle = prepare_probe(_config(tmp_path), now=NOW, runner=runner)

    with pytest.raises(ProbeError, match="condor_submit failed"):
        submit_probe(bundle, runner=runner)

    assert not (bundle.probe_dir / "submit.json").exists()
