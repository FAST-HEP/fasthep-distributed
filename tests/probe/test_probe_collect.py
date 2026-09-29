from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fasthep_distributed.probe import collect_probe, format_collection
from fasthep_distributed.probe._report import parse_condor_log

CONDOR_LOG = """\
000 (4242.000.000) 2026-09-29 17:50:01 Job submitted from host: <1.2.3.4:9618>
...
000 (4242.001.000) 2026-09-29 17:50:01 Job submitted from host: <1.2.3.4:9618>
...
001 (4242.000.000) 2026-09-29 17:51:00 Job executing on host: <5.6.7.8:9618>
...
005 (4242.000.000) 2026-09-29 17:52:00 Job terminated.
\t(1) Normal termination (return value 0)
\t\tUsr 0 00:00:01, Sys 0 00:00:00  -  Run Remote Usage
...
012 (4242.001.000) 2026-09-29 17:53:00 Job was held.
\tError from slot1@b9: Failed to execute probe.sh
\tCode 6 Subcode 2
...
005 (9999.000.000) 2026-09-29 17:54:00 Job terminated.
\t(1) Normal termination (return value 3)
...
"""


def _report(verdict: str) -> dict[str, Any]:
    return {
        "schema": "fasthep.distributed.probe/v1",
        "host": {"hostname": "b9g47n1234.cern.ch"},
        "checks": [
            {"name": "python.interpreter", "status": "pass", "error": None},
            {
                "name": "paths.eos",
                "status": "warn",
                "error": "TimeoutError: timeout after 30s",
            },
        ],
        "summary": {"verdict": verdict, "source_drift": []},
    }


def _submitted(tmp_path: Path, count: int) -> Path:
    (tmp_path / "submit.json").write_text(
        json.dumps({"probe_id": "probe-1", "cluster": 4242, "count": count}),
        encoding="utf-8",
    )
    (tmp_path / "condor.log").write_text(CONDOR_LOG, encoding="utf-8")
    return tmp_path


def test_parse_condor_log_tracks_latest_state_per_process(tmp_path: Path) -> None:
    path = tmp_path / "condor.log"
    path.write_text(CONDOR_LOG, encoding="utf-8")

    states = parse_condor_log(path, cluster=4242)

    assert states[0].state == "terminated"
    assert states[0].return_value == 0
    assert states[1].state == "held"
    assert states[1].detail == "Error from slot1@b9: Failed to execute probe.sh"
    assert set(states) == {0, 1}


def test_collect_passing_probe(tmp_path: Path) -> None:
    probe_dir = _submitted(tmp_path, count=1)
    (probe_dir / "stdout.0.json").write_text(json.dumps(_report("pass")), "utf-8")

    collection = collect_probe(probe_dir)

    assert collection.verdict == "pass"
    text = format_collection(collection)
    assert "job 4242.0  condor: terminated (return value 0)" in text
    assert "host: b9g47n1234.cern.ch  verdict: pass" in text
    assert "warn  paths.eos  TimeoutError: timeout after 30s" in text
    assert "source drift: none" in text
    assert text.endswith("verdict: pass\n")


def test_collect_held_job_without_report_fails(tmp_path: Path) -> None:
    probe_dir = _submitted(tmp_path, count=2)
    (probe_dir / "stdout.0.json").write_text(json.dumps(_report("pass")), "utf-8")
    (probe_dir / "stdout.1.json").write_text("", "utf-8")

    collection = collect_probe(probe_dir)

    assert [job.verdict for job in collection.jobs] == ["pass", "fail"]
    assert collection.verdict == "fail"
    text = format_collection(collection)
    assert "condor: held (Error from slot1@b9: Failed to execute probe.sh)" in text
    assert "report: stdout.1.json is empty" in text


def test_collect_running_job_is_pending(tmp_path: Path) -> None:
    (tmp_path / "submit.json").write_text(
        json.dumps({"probe_id": "probe-1", "cluster": 4242, "count": 1}), "utf-8"
    )

    collection = collect_probe(tmp_path)

    assert collection.jobs[0].condor.state == "unknown"
    assert collection.verdict == "pending"


def test_collect_invalid_report(tmp_path: Path) -> None:
    probe_dir = _submitted(tmp_path, count=1)
    (probe_dir / "stdout.0.json").write_text("{not json", "utf-8")

    collection = collect_probe(probe_dir)

    assert collection.verdict == "fail"
    assert "not valid JSON" in (collection.jobs[0].report_error or "")


def test_collect_unsubmitted_probe(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"probe_id": "p"}), "utf-8")

    collection = collect_probe(tmp_path)

    assert collection.submitted is False
    assert collection.probe_id == "p"
    assert collection.verdict == "fail"
    assert "not submitted" in format_collection(collection)
