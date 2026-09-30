from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._config import ProbeConfig
from ._worker import FAIL, SCHEMA

WORKER_SCRIPT = Path(__file__).with_name("_worker.py")

CONFIG_FILE = "config.json"
SUBMIT_FACTS_FILE = "submit_facts.json"
WORKER_FILE = "_worker.py"
EXECUTABLE_FILE = "probe.sh"
SUBMIT_FILE = "probe.sub"
SUBMIT_RECORD_FILE = "submit.json"
CONDOR_LOG_FILE = "condor.log"

SUBMIT_FACTS_TIMEOUT = 900.0
_SUBMITTED_PATTERN = re.compile(r"(\d+) job\(s\) submitted to cluster (\d+)")

Runner = Callable[..., subprocess.CompletedProcess[str]]


class ProbeError(RuntimeError):
    """Raised when a probe cannot be prepared or submitted."""


@dataclass(slots=True, frozen=True)
class ProbeBundle:
    probe_id: str
    probe_dir: Path
    submit_description: Path
    executable: Path
    submit_facts: dict[str, Any]
    count: int


@dataclass(slots=True, frozen=True)
class SubmitResult:
    probe_dir: Path
    cluster: int
    count: int


def prepare_probe(
    config: ProbeConfig,
    *,
    now: datetime | None = None,
    runner: Runner = subprocess.run,
) -> ProbeBundle:
    """
    Write a self-contained probe directory without contacting the schedd.

    The worker script is first run on the submit host with the configured
    interpreter. Its report becomes ``submit_facts.json``, which the worker
    compares against to detect source changes between submission and
    execution.
    """

    _validate_prefix(config)
    probe_id = make_probe_id(now=now, runner=runner)
    probe_dir = _create_probe_dir(config.probe_root, probe_id)

    shutil.copy2(WORKER_SCRIPT, probe_dir / WORKER_FILE)
    _write_json(probe_dir / CONFIG_FILE, config.worker_config(probe_id))

    submit_facts = _collect_submit_facts(config, probe_dir, runner=runner)
    submit_facts["submit_context"] = submit_context(config, runner=runner)
    _write_json(probe_dir / SUBMIT_FACTS_FILE, submit_facts)
    failed = submit_facts.get("summary", {}).get("failed_required", [])
    if failed:
        raise ProbeError(
            f"Required checks failed on the submit host: {', '.join(failed)}. "
            f"See {probe_dir / SUBMIT_FACTS_FILE}"
        )

    executable = probe_dir / EXECUTABLE_FILE
    executable.write_text(
        render_executable(prefix=config.prefix, probe_id=probe_id),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    submit_description = probe_dir / SUBMIT_FILE
    submit_description.write_text(
        render_submit_description(config, probe_dir),
        encoding="utf-8",
    )
    return ProbeBundle(
        probe_id=probe_id,
        probe_dir=probe_dir,
        submit_description=submit_description,
        executable=executable,
        submit_facts=submit_facts,
        count=config.condor.count,
    )


def submit_probe(
    bundle: ProbeBundle,
    *,
    runner: Runner = subprocess.run,
) -> SubmitResult:
    if (bundle.probe_dir / SUBMIT_RECORD_FILE).exists():
        raise ProbeError(
            f"Probe {bundle.probe_dir} was already submitted; prepare a new probe"
        )
    try:
        completed = runner(
            ["condor_submit", SUBMIT_FILE],
            cwd=bundle.probe_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ProbeError(f"Cannot run condor_submit: {exc}") from exc
    output = f"{completed.stdout or ''}{completed.stderr or ''}".strip()
    if completed.returncode != 0:
        raise ProbeError(f"condor_submit failed ({completed.returncode}): {output}")
    match = _SUBMITTED_PATTERN.search(completed.stdout or "")
    if match is None:
        raise ProbeError(f"Cannot parse condor_submit output: {output}")
    result = SubmitResult(
        probe_dir=bundle.probe_dir,
        cluster=int(match.group(2)),
        count=int(match.group(1)),
    )
    _write_json(
        bundle.probe_dir / SUBMIT_RECORD_FILE,
        {
            "schema": SCHEMA,
            "probe_id": bundle.probe_id,
            "cluster": result.cluster,
            "count": result.count,
            "submitted": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "condor_submit_output": output,
        },
    )
    return result


def make_probe_id(
    *,
    now: datetime | None = None,
    cwd: Path | None = None,
    runner: Runner = subprocess.run,
) -> str:
    timestamp = (now or _utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha(cwd or Path.cwd(), runner=runner)
    return f"{timestamp}-{sha}" if sha else timestamp


def render_executable(*, prefix: Path, probe_id: str) -> str:
    return _EXECUTABLE_TEMPLATE.format(
        prefix=shlex.quote(str(prefix)),
        probe_id=shlex.quote(probe_id),
        schema=SCHEMA,
    )


def render_submit_description(config: ProbeConfig, probe_dir: Path) -> str:
    condor = config.condor
    lines = [
        "# Generated by fasthep_distributed.probe; do not edit.",
        "universe = vanilla",
        f"executable = {probe_dir / EXECUTABLE_FILE}",
        "arguments = $(ClusterId) $(ProcId)",
        f"initialdir = {probe_dir}",
        "transfer_executable = False",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        f"transfer_input_files = {CONFIG_FILE}, {SUBMIT_FACTS_FILE}, {WORKER_FILE}",
        'transfer_output_files = ""',
        "output = stdout.$(ProcId).json",
        "error = stderr.$(ProcId).log",
        f"log = {CONDOR_LOG_FILE}",
        # No stream_output/stream_error: CERN's schedds reject them since
        # November 2025. stdout and stderr are transferred back on exit.
        f"request_cpus = {condor.request_cpus}",
        f"request_memory = {condor.request_memory}",
    ]
    if condor.request_disk is not None:
        lines.append(f"request_disk = {condor.request_disk}")
    if condor.flavour is not None:
        lines.append(f'+JobFlavour = "{condor.flavour}"')
    lines.extend(f"{key} = {value}" for key, value in condor.extra_directives.items())
    lines.append(f"queue {condor.count}")
    return "\n".join(lines) + "\n"


def submit_context(
    config: ProbeConfig,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    lockfile: dict[str, Any] | None = None
    if config.lockfile is not None:
        lockfile = {
            "path": str(config.lockfile),
            "sha256": _sha256(config.lockfile) if config.lockfile.is_file() else None,
        }
    return {
        "host": socket.getfqdn(),
        "user": getpass.getuser(),
        "cwd": str(Path.cwd()),
        "probe_root": str(config.probe_root),
        "lockfile": lockfile,
        "condor_version": _first_line(["condor_version"], runner=runner),
        "pixi_version": _pixi_version(runner=runner),
    }


def _collect_submit_facts(
    config: ProbeConfig,
    probe_dir: Path,
    *,
    runner: Runner,
) -> dict[str, Any]:
    output = probe_dir / SUBMIT_FACTS_FILE
    try:
        completed = runner(
            [
                str(config.python),
                WORKER_FILE,
                "--mode",
                "submit",
                "--config",
                CONFIG_FILE,
                "--output",
                SUBMIT_FACTS_FILE,
            ],
            cwd=probe_dir,
            env=_probe_environment(),
            check=False,
            capture_output=True,
            text=True,
            timeout=SUBMIT_FACTS_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(
            f"Cannot run {config.python} on the submit host: {exc}"
        ) from exc
    if not output.exists():
        details = (completed.stderr or completed.stdout or "").strip()
        raise ProbeError(
            f"{config.python} did not produce submit facts "
            f"(exit {completed.returncode}): {details}"
        )
    facts: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    if completed.returncode not in (0, 1) and not facts.get("summary"):
        facts["summary"] = {"verdict": FAIL, "failed_required": ["submit_facts"]}
    return facts


def _probe_environment() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _validate_prefix(config: ProbeConfig) -> None:
    if not config.prefix.is_dir():
        raise ProbeError(f"Probe prefix {config.prefix} does not exist")
    if not os.access(config.python, os.X_OK):
        raise ProbeError(f"Probe interpreter {config.python} is not executable")


def _create_probe_dir(probe_root: Path, probe_id: str) -> Path:
    probe_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        name = probe_id if attempt == 0 else f"{probe_id}-{attempt}"
        candidate = probe_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise ProbeError(f"Cannot create a unique probe directory under {probe_root}")


def _git_short_sha(cwd: Path, *, runner: Runner) -> str | None:
    try:
        completed = runner(
            ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    sha = (completed.stdout or "").strip()
    return sha if completed.returncode == 0 and sha else None


def _pixi_version(*, runner: Runner) -> str | None:
    pixi = shutil.which("pixi")
    if pixi is None:
        default = Path.home() / ".pixi" / "bin" / "pixi"
        pixi = str(default) if default.exists() else None
    if pixi is None:
        return None
    return _first_line([pixi, "--version"], runner=runner)


def _first_line(cmd: list[str], *, runner: Runner) -> str | None:
    try:
        completed = runner(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or "").strip().splitlines()
    return lines[0] if lines else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> datetime:
    return datetime.now(UTC)


_EXECUTABLE_TEMPLATE = """\
#!/bin/sh
# Generated by fasthep_distributed.probe; do not edit.
#
# Runs the worker checks with the interpreter under test. If the interpreter
# cannot run or exits without a report, emit a minimal failure report so the
# job always produces parseable JSON on stdout.
set -u

PROBE_PREFIX={prefix}
PROBE_ID={probe_id}
PROBE_PYTHON="$PROBE_PREFIX/bin/python"
CLUSTER="${{1:-}}"
PROCESS="${{2:-}}"
REPORT="$PWD/probe-report.json"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME

HOST="$(hostname -f 2>/dev/null || hostname)"
echo "[fasthep-probe] probe_id=$PROBE_ID host=$HOST cwd=$PWD" >&2
echo "[fasthep-probe] python=$PROBE_PYTHON" >&2

if [ -x "$PROBE_PYTHON" ]; then
    "$PROBE_PYTHON" _worker.py --mode worker --config config.json \\
        --submit-facts submit_facts.json --output "$REPORT" \\
        --cluster "$CLUSTER" --process "$PROCESS"
    rc=$?
    reason="worker script exited with code $rc without writing a report"
else
    rc=127
    reason="interpreter $PROBE_PYTHON is missing or not executable"
fi

if [ -s "$REPORT" ]; then
    cat "$REPORT"
    exit "$rc"
fi

echo "[fasthep-probe] $reason" >&2
json_escape() {{
    printf '%s' "$1" | sed -e 's/\\\\/\\\\\\\\/g' -e 's/"/\\\\"/g'
}}
printf '{{"schema": "%s", "mode": "worker", "probe_id": "%s", ' \\
    "{schema}" "$(json_escape "$PROBE_ID")"
printf '"job": {{"cluster": %s, "process": %s}}, ' \\
    "${{CLUSTER:-null}}" "${{PROCESS:-null}}"
printf '"host": {{"hostname": "%s", "kernel": "%s"}}, ' \\
    "$(json_escape "$HOST")" "$(json_escape "$(uname -r)")"
printf '"checks": [{{"name": "python.interpreter", "status": "fail", '
printf '"required": true, "duration_s": null, '
printf '"detail": {{"executable": "%s", "exit_code": %s}}, "error": "%s"}}], ' \\
    "$(json_escape "$PROBE_PYTHON")" "$rc" "$(json_escape "$reason")"
printf '"summary": {{"verdict": "fail", "failed_required": ["python.interpreter"], '
printf '"warnings": [], "source_drift": []}}}}\\n'
if [ "$rc" -eq 0 ]; then
    rc=1
fi
exit "$rc"
"""
