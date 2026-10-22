"""
Output delivery for the HTCondor backend.

With ``execution.config.output_destination`` set, workers write their writer
outputs below their job directory and HTCondor sends them to EOS when the worker
job exits. This module derives what to send, tells the workers where to write,
and checks afterwards that every file arrived intact.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from hepflow.build_layout import partition_output_roots
from hepflow.model.io import ArtifactManifest, ArtifactReference
from hepflow.model.plan import ExecutionPlan

# Workers write below their working directory, which is the HTCondor job
# directory, and HTCondor sends files from there.
WORKER_FILES_ROOT = "."
DEFAULT_COMMAND_TIMEOUT = 60.0

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def output_destination(execution: dict[str, Any]) -> str | None:
    destination = dict(execution.get("config") or {}).get("output_destination")
    return str(destination) if destination else None


def execution_with_output_roots(
    plan: ExecutionPlan, execution: dict[str, Any]
) -> dict[str, Any]:
    """
    Fill in ``output_files`` and ``output_directories`` from the plan.

    When an output destination is set and ``output_files`` is not, the files to
    send are the top-level directories that the plan's partition writers create
    (for example one per dataset), and those directories are created in every job
    before the worker starts, so that a worker that handled no partition of a
    dataset still has the directory HTCondor is told to send.
    """

    config = dict(execution.get("config") or {})
    if config.get("output_destination") is None or config.get("output_files") is not None:
        return execution
    roots = partition_output_roots(plan)
    if not roots:
        raise ValueError(
            "execution.config.output_destination is set, but the workflow has no "
            "partition writer that writes below the files root; list "
            "execution.config.output_files explicitly or remove the destination"
        )
    config["output_files"] = roots
    config.setdefault("output_directories", list(roots))
    return {**execution, "config": config}


def partition_task_context(execution: dict[str, Any]) -> dict[str, str]:
    """Context entries for partition tasks, so writers write where HTCondor sends from."""

    destination = output_destination(execution)
    if destination is None:
        return {}
    return {
        "output_files_root": WORKER_FILES_ROOT,
        "output_destination": destination,
    }


def delivered_writer_records(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Manifest records of the files that partition tasks wrote below the files root."""

    records: list[dict[str, Any]] = []
    for item in task_results:
        for value in dict(item.get("value_store") or {}).values():
            for reference in _references(value):
                manifest = reference.metadata.get("writer_manifest")
                if (
                    isinstance(manifest, dict)
                    and manifest.get("path_type") == "relative_to_files_root"
                ):
                    records.append(dict(manifest))
    return records


def verify_delivery(
    records: list[dict[str, Any]],
    destination: str,
    *,
    runner: Runner = subprocess.run,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> dict[str, Any]:
    """
    Check with ``xrdfs`` that each file is at the destination with the right size
    and adler32. Never raises; problems are listed in the returned summary.

    Statuses: ``ok``, ``missing``, ``size_mismatch``, ``checksum_mismatch`` and
    ``error`` (the check itself failed).
    """

    server, base = split_destination(destination)
    problems: list[dict[str, Any]] = []
    delivered = 0
    for record in sorted(records, key=lambda item: str(item.get("path"))):
        relative = str(record["path"])
        status, detail = _check_file(
            server,
            f"{base}/{relative}",
            size=record.get("size_bytes"),
            adler32=record.get("adler32"),
            runner=runner,
            timeout=timeout,
        )
        if status == "ok":
            delivered += 1
        else:
            problems.append({"path": relative, "status": status, **detail})
    return {
        "destination": destination,
        "expected": len(records),
        "delivered": delivered,
        "complete": not problems and bool(records),
        "problems": problems,
    }


def split_destination(destination: str) -> tuple[str, str]:
    """``root://host//eos/x/`` becomes (``root://host``, ``/eos/x``)."""

    parts = urlsplit(destination)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Not a root:// URL: {destination!r}")
    return f"{parts.scheme}://{parts.netloc}", "/" + parts.path.strip("/")


def _check_file(
    server: str,
    path: str,
    *,
    size: Any,
    adler32: Any,
    runner: Runner,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    stat = _xrdfs(runner, server, ["stat", path], timeout)
    if stat.returncode != 0:
        message = (stat.stderr or stat.stdout or "").strip()
        status = "missing" if "No such file" in message else "error"
        return status, {"message": message[-300:]}
    remote_size = _parse_size(stat.stdout)
    if size is not None and remote_size != int(size):
        return "size_mismatch", {"expected": int(size), "found": remote_size}
    if adler32:
        checksum = _xrdfs(runner, server, ["query", "checksum", path], timeout)
        if checksum.returncode != 0:
            message = (checksum.stderr or checksum.stdout or "").strip()
            return "error", {"message": message[-300:]}
        found = _parse_adler32(checksum.stdout)
        if found != str(adler32).lower():
            return "checksum_mismatch", {"expected": str(adler32), "found": found}
    return "ok", {}


def _xrdfs(
    runner: Runner, server: str, arguments: list[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            ["xrdfs", server, *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            ["xrdfs", server, *arguments], 255, "", f"{type(exc).__name__}: {exc}"
        )


def _parse_size(output: str) -> int | None:
    for line in output.splitlines():
        if line.startswith("Size:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _parse_adler32(output: str) -> str | None:
    fields = output.split()
    if len(fields) >= 2 and fields[0] == "adler32":
        return fields[1].lower()
    return None


def _references(value: Any) -> list[ArtifactReference]:
    if isinstance(value, ArtifactManifest):
        return list(value.parts)
    if isinstance(value, ArtifactReference):
        return [value]
    if isinstance(value, list):
        return [reference for item in value for reference in _references(item)]
    return []
