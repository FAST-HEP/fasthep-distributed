from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from fasthep_distributed.probe import _worker as worker


def test_worker_script_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "worker script must not use relative imports"
            assert node.module is not None
            modules.add(node.module.split(".", maxsplit=1)[0])

    assert modules - {"__future__"} <= set(sys.stdlib_module_names)


def test_check_path_lists_directories(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("x", encoding="utf-8")
    (tmp_path / "b").mkdir()

    status, detail = worker.check_path(str(tmp_path))

    assert status == worker.PASS
    assert detail == {
        "path": str(tmp_path),
        "type": "directory",
        "entries": 2,
        "entries_truncated": False,
    }


def test_check_path_reads_files(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("hello", encoding="utf-8")

    status, detail = worker.check_path(str(path))

    assert status == worker.PASS
    assert detail["type"] == "file"
    assert detail["read_bytes"] == 5


def test_missing_required_path_fails(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing")

    check = worker.run_check("paths.missing", lambda: worker.check_path(missing))

    assert check["status"] == worker.FAIL
    assert check["error"].startswith("FileNotFoundError")


def test_missing_optional_path_warns(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing")

    check = worker.run_check(
        "paths.eos", lambda: worker.check_path(missing), required=False
    )

    assert check["status"] == worker.WARN
    assert check["required"] is False


def test_hung_check_times_out() -> None:
    def hang() -> worker.CheckResult:
        time.sleep(2)
        return worker.PASS, {}

    check = worker.run_check("paths.hung", hang, timeout=0.05)

    assert check["status"] == worker.FAIL
    assert check["error"] == "TimeoutError: timeout after 0.05s"


def test_imports_that_install_signal_handlers_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # dask_jobqueue installs a signal handler at import time, which fails off
    # the main thread.
    (tmp_path / "installs_signal_handler.py").write_text(
        "import signal\nsignal.signal(signal.SIGTERM, signal.getsignal(signal.SIGTERM))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "installs_signal_handler", raising=False)

    check = worker.run_check(
        "python.imports",
        lambda: worker.check_imports(["installs_signal_handler"], []),
        timeout=30,
        main_thread=True,
    )

    assert check["status"] == worker.PASS, check


def test_main_thread_check_times_out_with_alarm() -> None:
    def slow_imports() -> worker.CheckResult:
        time.sleep(2)
        return worker.PASS, {}

    check = worker.run_check(
        "python.imports", slow_imports, timeout=0.05, main_thread=True
    )

    assert check["status"] == worker.FAIL
    assert check["error"] == "TimeoutError: timeout after 0.05s"


def test_detail_error_is_promoted() -> None:
    check = worker.run_check("x", lambda: (worker.FAIL, {"error": "bad", "a": 1}))

    assert check["error"] == "bad"
    assert check["detail"] == {"a": 1}


def test_check_imports_distinguishes_required_and_optional() -> None:
    status, detail = worker.check_imports(["json"], ["not_a_real_module_xyz"])

    assert status == worker.WARN
    assert detail["modules"]["json"]["ok"] is True
    assert detail["modules"]["json"]["required"] is True
    assert detail["modules"]["not_a_real_module_xyz"]["ok"] is False
    assert "optional imports failed" in detail["warning"]

    status, detail = worker.check_imports(["not_a_real_module_xyz"], [])

    assert status == worker.FAIL
    assert "required imports failed" in detail["error"]


def test_check_interpreter_compares_prefix(tmp_path: Path) -> None:
    assert worker.check_interpreter(sys.prefix)[0] == worker.PASS

    status, detail = worker.check_interpreter(str(tmp_path))

    assert status == worker.FAIL
    assert "does not match" in detail["error"]


def test_check_scratch_uses_condor_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("_CONDOR_SCRATCH_DIR", str(tmp_path))

    status, detail = worker.check_scratch()

    assert status == worker.PASS
    assert detail["path"] == str(tmp_path)
    assert detail["from_condor"] is True
    assert detail["free_bytes"] > 0
    assert list(tmp_path.iterdir()) == []


def _editables_check(*items: dict[str, Any]) -> dict[str, Any]:
    return {"name": "python.editables", "detail": {"distributions": list(items)}}


def _editable(name: str, revision: str, **vcs: Any) -> dict[str, Any]:
    return {
        "name": name,
        "version": "1.0",
        "vcs": {"revision": revision, "dirty": False, **vcs},
    }


def test_source_drift_reports_changed_and_missing_editables() -> None:
    submit = {
        "checks": [
            _editables_check(
                _editable("same", "a"),
                _editable("moved", "a"),
                _editable("gone", "a"),
            )
        ]
    }
    running = [
        _editables_check(
            _editable("same", "a"),
            _editable("moved", "b", dirty=True),
        )
    ]

    drift = worker.source_drift(submit, running)

    assert [(item["name"], item["reason"]) for item in drift] == [
        ("gone", "missing on worker"),
        ("moved", "changed: revision, dirty"),
    ]


def test_summary_fails_only_on_required_failures() -> None:
    checks = [
        {"name": "a", "status": worker.PASS, "required": True},
        {"name": "b", "status": worker.WARN, "required": False},
    ]

    assert worker.summarise(checks, [])["verdict"] == worker.PASS

    checks.append({"name": "c", "status": worker.FAIL, "required": True})
    summary = worker.summarise(checks, [])

    assert summary["verdict"] == worker.FAIL
    assert summary["failed_required"] == ["c"]
    assert summary["warnings"] == ["b"]


def test_main_writes_worker_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("_CONDOR_SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "editable_records", lambda prefix, git_timeout: [])
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "probe_id": "probe-1",
                "prefix": sys.prefix,
                "imports": {"required": ["json"], "optional": []},
                "paths": [
                    {"name": "tmp", "path": str(tmp_path), "required": True},
                    {"name": "eos", "path": str(tmp_path / "no"), "required": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    code = worker.main(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--cluster",
            "12",
            "--process",
            "0",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert report["schema"] == worker.SCHEMA
    assert report["mode"] == "worker"
    assert report["job"] == {"cluster": 12, "process": 0}
    assert report["submit"] is None
    assert [check["name"] for check in report["checks"]] == [
        "python.interpreter",
        "python.imports",
        "python.editables",
        "paths.tmp",
        "paths.eos",
        "scratch",
    ]
    assert report["summary"]["verdict"] == worker.PASS
    assert report["summary"]["warnings"] == ["paths.eos"]


def test_submit_mode_skips_scratch(tmp_path: Path) -> None:
    report = worker.build_report(
        {"prefix": sys.prefix, "imports": {"required": ["json"]}},
        mode="submit",
    )

    assert "scratch" not in [check["name"] for check in report["checks"]]
    assert "submit" not in report
