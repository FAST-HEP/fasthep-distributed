"""
Worker-side checks for the HTCondor worker probe.

This file is copied into each probe directory and executed by path with the
interpreter under test::

    $PREFIX/bin/python _worker.py --mode worker --config config.json

It must only import the Python standard library: importing FAST-HEP packages is
one of the checks, so it cannot be a precondition for running the probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

SCHEMA = "fasthep.distributed.probe/v1"

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

DEFAULT_TIMEOUTS = {
    "imports": 300.0,
    "paths": 30.0,
    "git": 60.0,
    "scratch": 60.0,
}

# Environment variables that are safe and useful to record. The environment is
# never dumped wholesale because it may contain credentials or tokens.
RECORDED_ENV_VARS = (
    "_CONDOR_SCRATCH_DIR",
    "TMPDIR",
    "HOME",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "OMP_NUM_THREADS",
)

SCRATCH_TEST_BYTES = 1024 * 1024
MAX_LISTED_ENTRIES = 10_000

CheckResult = tuple[str, dict[str, Any]]


def installed_package_records(prefix: Path) -> list[dict[str, Any]]:
    site_packages = _site_packages(prefix)
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions(path=[str(site_packages)]):
        name = dist.metadata.get("Name")
        if not name:
            continue
        direct_url_text = dist.read_text("direct_url.json")
        editable = False
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError:
                direct_url = {}
            editable = bool(direct_url.get("dir_info", {}).get("editable"))
        records.append(
            {
                "name": name,
                "version": dist.version,
                "editable": editable,
            }
        )
    return sorted(records, key=lambda item: str(item["name"]).lower())


def vcs_state(
    source_path: Path,
    *,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    try:
        root = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"root": root, "revision": None, "dirty": None}
    return {
        "root": root,
        "revision": revision,
        "dirty": bool(status.strip()),
    }


def editable_records(
    prefix: Path,
    *,
    git_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    Describe editable distributions in ``prefix`` with their source revisions.

    ``diff_sha256`` fingerprints uncommitted tracked changes, so that edits made
    between submission and execution are visible even when both sides are
    dirty.
    """

    records: list[dict[str, Any]] = []
    for dist in metadata.distributions(path=[str(_site_packages(prefix))]):
        direct_url_text = dist.read_text("direct_url.json")
        if not direct_url_text:
            continue
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            continue
        if not direct_url.get("dir_info", {}).get("editable"):
            continue
        url = str(direct_url.get("url") or "")
        source_path = url.removeprefix("file://") if url.startswith("file://") else None
        vcs = None
        if source_path is not None:
            vcs = vcs_state(Path(source_path), timeout=git_timeout)
            if vcs is not None and vcs.get("dirty"):
                vcs["diff_sha256"] = _git_diff_sha256(
                    Path(source_path), timeout=git_timeout
                )
        records.append(
            {
                "name": dist.metadata["Name"],
                "version": dist.version,
                "source_path": source_path,
                "vcs": vcs,
            }
        )
    return sorted(records, key=lambda item: str(item["name"]).lower())


def host_facts() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "hostname": socket.getfqdn(),
        "user": _user_name(),
        "os": _os_release(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "libc": {"name": libc_name or None, "version": libc_version or None},
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
    }


def check_interpreter(prefix: str) -> CheckResult:
    expected_prefix = os.path.realpath(prefix)
    actual_prefix = os.path.realpath(sys.prefix)
    detail = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "configured_prefix": prefix,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    if actual_prefix != expected_prefix:
        detail["error"] = (
            f"interpreter prefix {actual_prefix} does not match configured "
            f"prefix {expected_prefix}"
        )
        return FAIL, detail
    return PASS, detail


def check_imports(required: list[str], optional: list[str]) -> CheckResult:
    try:
        distributions = metadata.packages_distributions()
    except Exception:
        distributions = {}
    detail: dict[str, Any] = {}
    failed_required: list[str] = []
    failed_optional: list[str] = []
    for name in [*required, *optional]:
        record = _import_record(name, distributions)
        record["required"] = name in required
        detail[name] = record
        if not record["ok"]:
            (failed_required if name in required else failed_optional).append(name)
    if failed_required:
        return FAIL, {
            "modules": detail,
            "error": f"required imports failed: {', '.join(failed_required)}",
        }
    if failed_optional:
        return WARN, {
            "modules": detail,
            "warning": f"optional imports failed: {', '.join(failed_optional)}",
        }
    return PASS, {"modules": detail}


def check_editables(
    prefix: str,
    *,
    git_timeout: float | None,
) -> CheckResult:
    records = editable_records(Path(prefix), git_timeout=git_timeout)
    missing_vcs = [item["name"] for item in records if item["vcs"] is None]
    detail: dict[str, Any] = {"distributions": records}
    if missing_vcs:
        detail["warning"] = (
            f"no git revision for editable distributions: {', '.join(missing_vcs)}"
        )
        return WARN, detail
    return PASS, detail


def check_path(path: str) -> CheckResult:
    target = Path(path)
    info = target.stat()
    detail: dict[str, Any] = {"path": path}
    if target.is_dir():
        count = 0
        with os.scandir(path) as entries:
            for _ in entries:
                count += 1
                if count >= MAX_LISTED_ENTRIES:
                    break
        detail["type"] = "directory"
        detail["entries"] = count
        detail["entries_truncated"] = count >= MAX_LISTED_ENTRIES
    else:
        with target.open("rb") as stream:
            data = stream.read(4096)
        detail["type"] = "file"
        detail["size_bytes"] = info.st_size
        detail["read_bytes"] = len(data)
    return PASS, detail


def check_scratch() -> CheckResult:
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR") or str(Path.cwd())
    usage = shutil.disk_usage(scratch)
    payload = os.urandom(SCRATCH_TEST_BYTES)
    test_file = Path(scratch) / f".fasthep-probe-scratch-{os.getpid()}"
    start = time.monotonic()
    try:
        test_file.write_bytes(payload)
        read_back = test_file.read_bytes()
    finally:
        test_file.unlink(missing_ok=True)
    elapsed = time.monotonic() - start
    detail = {
        "path": scratch,
        "from_condor": "_CONDOR_SCRATCH_DIR" in os.environ,
        "tmpdir": os.environ.get("TMPDIR"),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "test_bytes": SCRATCH_TEST_BYTES,
        "test_seconds": round(elapsed, 4),
    }
    if read_back != payload:
        detail["error"] = "scratch read-back did not match written data"
        return FAIL, detail
    return PASS, detail


def run_check(
    name: str,
    func: Callable[[], CheckResult],
    *,
    required: bool = True,
    timeout: float | None = None,
    main_thread: bool = False,
) -> dict[str, Any]:
    """
    Run one check and record its outcome.

    Checks run in a watchdog thread by default. ``main_thread=True`` runs the
    check on the calling thread with a ``SIGALRM`` timeout instead; imports
    need this because modules such as ``dask_jobqueue`` install signal
    handlers, which only works on the main thread.
    """

    start = time.monotonic()
    error: str | None = None
    call = _call_with_alarm if main_thread else _call_with_timeout
    try:
        status, detail = call(func, timeout)
    except (Exception, _ProbeTimeout) as exc:
        status, detail = FAIL, {}
        error = _format_error(exc)
    detail = dict(detail)
    if error is None and "error" in detail:
        error = str(detail.pop("error"))
    if status == FAIL and not required:
        status = WARN
    return {
        "name": name,
        "status": status,
        "required": required,
        "duration_s": round(time.monotonic() - start, 4),
        "detail": detail,
        "error": error,
    }


def run_checks(config: Mapping[str, Any], *, mode: str) -> list[dict[str, Any]]:
    timeouts = {**DEFAULT_TIMEOUTS, **config.get("timeouts", {})}
    prefix = str(config["prefix"])
    imports = config.get("imports", {})
    checks = [
        run_check("python.interpreter", lambda: check_interpreter(prefix)),
        run_check(
            "python.imports",
            lambda: check_imports(
                list(imports.get("required", [])),
                list(imports.get("optional", [])),
            ),
            timeout=timeouts["imports"],
            main_thread=True,
        ),
        run_check(
            "python.editables",
            lambda: check_editables(prefix, git_timeout=timeouts["git"]),
            required=False,
        ),
    ]
    for spec in config.get("paths", []):
        path = str(spec["path"])
        checks.append(
            run_check(
                f"paths.{spec['name']}",
                partial(check_path, path),
                required=bool(spec.get("required", True)),
                timeout=timeouts["paths"],
            )
        )
    if mode == "worker":
        checks.append(run_check("scratch", check_scratch, timeout=timeouts["scratch"]))
    return checks


def source_drift(
    submit_facts: Mapping[str, Any] | None,
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not submit_facts:
        return []
    submitted = _editables_by_name(submit_facts.get("checks", []))
    running = _editables_by_name(checks)
    drift: list[dict[str, Any]] = []
    for name in sorted(set(submitted) | set(running)):
        before = submitted.get(name)
        after = running.get(name)
        if before is None or after is None:
            drift.append(
                {
                    "name": name,
                    "reason": "missing on worker"
                    if after is None
                    else "missing at submit",
                }
            )
            continue
        fields = ("revision", "dirty", "diff_sha256")
        before_vcs = before.get("vcs") or {}
        after_vcs = after.get("vcs") or {}
        changed = [
            field for field in fields if before_vcs.get(field) != after_vcs.get(field)
        ]
        if before.get("version") != after.get("version"):
            changed.insert(0, "version")
        if changed:
            drift.append(
                {
                    "name": name,
                    "reason": f"changed: {', '.join(changed)}",
                    "submit": {"version": before.get("version"), "vcs": before_vcs},
                    "worker": {"version": after.get("version"), "vcs": after_vcs},
                }
            )
    return drift


def summarise(
    checks: list[dict[str, Any]],
    drift: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_required = [
        check["name"]
        for check in checks
        if check["status"] == FAIL and check["required"]
    ]
    warnings = [check["name"] for check in checks if check["status"] == WARN]
    return {
        "verdict": FAIL if failed_required else PASS,
        "failed_required": failed_required,
        "warnings": warnings,
        "source_drift": drift,
    }


def build_report(
    config: Mapping[str, Any],
    *,
    mode: str,
    job: Mapping[str, Any] | None = None,
    submit_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    checks = run_checks(config, mode=mode)
    drift = source_drift(submit_facts, checks) if mode == "worker" else []
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "probe_id": config.get("probe_id"),
        "job": dict(job or {}),
        "started": _iso_utc(started),
        "finished": _iso_utc(time.time()),
        "host": host_facts(),
        "checks": checks,
    }
    if mode == "worker":
        report["submit"] = dict(submit_facts) if submit_facts else None
    report["summary"] = summarise(checks, drift)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FAST-HEP worker probe")
    parser.add_argument("--mode", choices=("worker", "submit"), default="worker")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--submit-facts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cluster")
    parser.add_argument("--process")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    submit_facts = None
    if args.submit_facts is not None and args.submit_facts.exists():
        submit_facts = json.loads(args.submit_facts.read_text(encoding="utf-8"))
    job = {
        "cluster": _maybe_int(args.cluster),
        "process": _maybe_int(args.process),
    }
    report = build_report(config, mode=args.mode, job=job, submit_facts=submit_facts)
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.output)
    return 0 if report["summary"]["verdict"] == PASS else 1


class _ProbeTimeout(BaseException):
    """
    Raised by the ``SIGALRM`` handler.

    Derives from ``BaseException`` so that per-module ``except Exception``
    handlers in :func:`_import_record` cannot swallow the timeout.
    """


def _call_with_alarm(
    func: Callable[[], CheckResult],
    timeout: float | None,
) -> CheckResult:
    if (
        timeout is None
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        return func()

    def handler(*_: object) -> None:
        raise _ProbeTimeout(f"timeout after {timeout:g}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _call_with_timeout(
    func: Callable[[], CheckResult],
    timeout: float | None,
) -> CheckResult:
    if timeout is None:
        return func()
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = func()
        except BaseException as exc:
            outcome["error"] = exc

    # A daemon thread is abandoned on timeout. Filesystem calls on a hung AFS,
    # EOS or CVMFS mount cannot be interrupted, so the probe moves on and exits
    # with os._exit instead of waiting for them.
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"timeout after {timeout:g}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]  # type: ignore[no-any-return]


def _import_record(
    name: str,
    distributions: Mapping[str, list[str]],
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        module = import_module(name)
    except (Exception, SystemExit) as exc:
        return {
            "ok": False,
            "error": _format_error(exc),
            "seconds": round(time.monotonic() - start, 4),
        }
    dist_names = list(distributions.get(name.split(".", maxsplit=1)[0], []))
    version = getattr(module, "__version__", None)
    if version is None and dist_names:
        try:
            version = metadata.version(dist_names[0])
        except metadata.PackageNotFoundError:
            version = None
    return {
        "ok": True,
        "version": str(version) if version is not None else None,
        "distributions": dist_names,
        "file": getattr(module, "__file__", None),
        "seconds": round(time.monotonic() - start, 4),
    }


def _editables_by_name(checks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for check in checks:
        if check.get("name") == "python.editables":
            return {
                str(item["name"]): item
                for item in check.get("detail", {}).get("distributions", [])
            }
    return {}


def _git_diff_sha256(source_path: Path, *, timeout: float | None) -> str | None:
    try:
        diff = subprocess.run(
            ["git", "-C", str(source_path), "diff", "HEAD"],
            check=True,
            capture_output=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return hashlib.sha256(diff).hexdigest()


def _site_packages(prefix: Path) -> Path:
    return Path(
        sysconfig.get_path(
            "purelib",
            vars={
                "base": str(prefix),
                "platbase": str(prefix),
            },
        )
    )


def _os_release() -> dict[str, str]:
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return {}
    return {
        key: release[key]
        for key in ("ID", "VERSION_ID", "PRETTY_NAME")
        if key in release
    }


def _user_name() -> str | None:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return None


def _format_error(exc: BaseException) -> str:
    kind = "TimeoutError" if isinstance(exc, _ProbeTimeout) else type(exc).__name__
    return f"{kind}: {exc}"


def _iso_utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter shutdown: abandoned threads may be blocked on a hung
    # network filesystem and would otherwise keep the job alive.
    os._exit(exit_code)
