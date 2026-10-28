from __future__ import annotations

import errno
import re
import shlex
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hepflow.build_layout import BuildPaths

from fasthep_distributed._dask._common import compute_with_client
from fasthep_distributed._dask._htcondor_shutdown import (
    DEFAULT_WORKER_SHUTDOWN_GRACE,
    cluster_job_ids,
    shutdown_htcondor_workers,
)
from fasthep_distributed._dask._pooled import DaskPooledHTCondorCluster
from fasthep_distributed._dask._pools import (
    DaskWorkerPool,
    dask_resources_for_resource,
    dask_worker_resource_args,
    resolve_dask_worker_pools,
)
from fasthep_distributed._dask._worker_env import (
    DistributedPreparationProgress,
    StagedExecutionFiles,
    prepare_staged_execution_files,
    prepare_worker_environment,
    validate_transfer_file_basenames,
)

MISSING_DASK_JOBQUEUE_MESSAGE = (
    "Dask HTCondor strategy requires dask-jobqueue. Install the dask HTCondor "
    "extra or add dask-jobqueue to the environment."
)

_HTCONDOR_FIXED_DIRECTIVES = {
    "transfer_executable": "False",
    "transfer_output_files": '""',
}
_HTCONDOR_RUNTIME_DIRECTIVES = {"Output", "Error", "Log"}
_SCHEDULER_OPTION_KEYS = {"host", "port", "dashboard_address", "interface", "protocol"}
# Sentinel resolved on the submit host when the cluster starts, so compiled
# plans do not record a particular interactive node.
SCHEDULER_HOST_FQDN = "fqdn"


def normalize_dask_htcondor_config(execution: dict[str, Any]) -> dict[str, Any]:
    pools = _resolve_htcondor_worker_pools(execution)
    pool_specs: dict[str, dict[str, Any]] = {
        pool.name: {
            "workers": pool.workers or 0,
            "job_kwargs": _htcondor_cluster_options_for_pool(pool),
        }
        for pool in pools
    }
    scale = {pool.name: pool.workers or 0 for pool in pools}
    first_pool = pools[0]
    first_options = dict(pool_specs[first_pool.name]["job_kwargs"])
    config = dict(execution.get("config") or {})

    return {
        "workers": first_pool.workers,
        "batch_name": _batch_name(config),
        "worker_start_timeout": _worker_start_timeout(config),
        "worker_shutdown_grace": _worker_shutdown_grace(config),
        "scheduler_options": _scheduler_options(config),
        "cluster_options": first_options,
        "pool_specs": pool_specs,
        "scale": scale,
        "pools": [_pool_summary(pool) for pool in pools],
    }


def _resolve_htcondor_worker_pools(execution: dict[str, Any]) -> list[DaskWorkerPool]:
    pools = resolve_dask_worker_pools(execution)
    if pools:
        return pools

    config = dict(execution.get("config") or {})
    resources_by_name = dict(execution.get("resources") or {})
    default_resources = dict(resources_by_name.get("default") or {})
    workers = config.get("n_workers", config.get("workers"))
    if workers is not None:
        workers = int(workers)
    dask_resources = (
        dask_resources_for_resource("default", default_resources)
        if "default" in resources_by_name
        else {}
    )
    return [
        DaskWorkerPool(
            name="default",
            resource_name="default",
            workers=workers,
            resources=default_resources,
            dask_resources=dask_resources,
            config=config,
        )
    ]


def compute_with_htcondor(
    tasks: list[Any],
    *,
    execution: dict[str, Any],
    build_paths: BuildPaths,
    progress: Any | None = None,
) -> tuple[list[Any], str | None, dict[str, Any]]:
    with htcondor_client(
        execution=execution,
        build_paths=build_paths,
        progress=progress,
    ) as (client, htcondor_config):
        if progress is not None:
            progress.phase_started("executing")
        results, dashboard_link = compute_with_client(client, tasks)
        return results, dashboard_link, htcondor_config


@contextmanager
def htcondor_client(
    *,
    execution: dict[str, Any],
    build_paths: BuildPaths,
    progress: Any | None = None,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """
    Start HTCondor workers, wait for them, and yield a connected client.

    Shared by plan execution and the Dask worker probe, so that the probe
    exercises the production start-up path.
    """

    try:
        import dask_jobqueue  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(MISSING_DASK_JOBQUEUE_MESSAGE) from exc

    from distributed import Client  # noqa: PLC0415

    htcondor_config = normalize_dask_htcondor_config(execution)
    scheduler_options = resolve_scheduler_options(htcondor_config["scheduler_options"])
    check_scheduler_port(scheduler_options)
    pool_specs = _prepare_htcondor_pool_specs(
        htcondor_config["pool_specs"],
        execution=execution,
        build_paths=build_paths,
        progress=progress,
    )

    batch_name = htcondor_config["batch_name"] or build_paths.root.name
    cluster = DaskPooledHTCondorCluster(
        pools=pool_specs,
        scheduler_options=scheduler_options,
        name_prefix=f"{batch_name}-" if batch_name else "",
    )
    client = Client(cluster)
    try:
        cluster.scale(htcondor_config["scale"])
        _wait_for_htcondor_workers(
            client,
            requested=sum(int(value) for value in htcondor_config["scale"].values()),
            pool_names=list(htcondor_config["scale"]),
            timeout=float(htcondor_config["worker_start_timeout"]),
            build_paths=build_paths,
        )
        yield client, htcondor_config
    finally:
        # Let the workers and their jobs exit on their own before the cluster
        # removes them: removed jobs deliver no output files and no logs.
        htcondor_config["shutdown"] = shutdown_htcondor_workers(
            client,
            logs_dir=_htcondor_execution_paths(build_paths)["logs"],
            reported_job_ids=cluster_job_ids(cluster),
            grace=float(htcondor_config["worker_shutdown_grace"]),
        )
        client.close()
        cluster.close()


def resolve_scheduler_options(options: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(options)
    if resolved.get("host") == SCHEDULER_HOST_FQDN:
        resolved["host"] = socket.getfqdn()
    return resolved


def check_scheduler_port(options: dict[str, Any]) -> None:
    """
    Fail before submitting workers if the fixed scheduler port is taken.

    On shared interactive nodes (e.g. lxplus, port 8786) another user's
    scheduler may already hold the port; ports are per node, not per user.
    """

    port = options.get("port")
    if not port:
        return
    host = str(options.get("host") or "")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Like the scheduler's own listener: connections in TIME_WAIT from a
        # previous run must not count as the port being in use.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, int(port)))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            node = host or socket.getfqdn()
            raise RuntimeError(
                f"Dask scheduler port {port} is already in use on {node}. Only one "
                "scheduler per node can use a fixed port; run from another node or "
                "choose a different execution.config.scheduler_options.port."
            ) from exc


def _prepare_htcondor_pool_specs(
    pool_specs: dict[str, dict[str, Any]],
    *,
    execution: dict[str, Any],
    build_paths: BuildPaths,
    progress: Any | None = None,
) -> dict[str, dict[str, Any]]:
    staging_mode = str(dict(execution.get("staging") or {}).get("mode") or "shared")
    preparation_progress = DistributedPreparationProgress(progress)
    preparation_progress.step(
        "resolving_editable_snapshots",
        staging_mode=staging_mode,
    )
    try:
        return _prepare_htcondor_pool_specs_impl(
            pool_specs,
            execution=execution,
            build_paths=build_paths,
            preparation_progress=preparation_progress,
            staging_mode=staging_mode,
        )
    except BaseException as exc:
        preparation_progress.fail(exc)
        raise


def _prepare_htcondor_pool_specs_impl(
    pool_specs: dict[str, dict[str, Any]],
    *,
    execution: dict[str, Any],
    build_paths: BuildPaths,
    preparation_progress: DistributedPreparationProgress,
    staging_mode: str,
) -> dict[str, dict[str, Any]]:
    prepared = {
        name: {"workers": spec["workers"], "job_kwargs": dict(spec["job_kwargs"])}
        for name, spec in pool_specs.items()
    }
    worker_environment = prepare_worker_environment(
        execution,
        build_paths=build_paths,
        progress=preparation_progress,
    )
    preparation_progress.step(
        "preparing_compilation_and_staging_files",
        staging_mode=staging_mode,
    )
    staged_files = prepare_staged_execution_files(
        worker_environment,
        build_paths=build_paths,
        staging=dict(execution.get("staging") or {}),
    )
    preparation_progress.step(
        "ready_to_submit_workers",
        staging_mode=staging_mode,
        transfer_file_count=len(staged_files.transfer_files) if staged_files else 0,
    )
    paths = _htcondor_execution_paths(build_paths)
    for path in [_dask_htcondor_dir(build_paths, "submit"), *paths.values()]:
        path.mkdir(parents=True, exist_ok=True)

    for name, spec in prepared.items():
        job_kwargs = spec["job_kwargs"]
        log_directory = job_kwargs.get("log_directory")
        if log_directory is not None:
            log_path = Path(str(log_directory))
            if not log_path.is_absolute():
                log_path = build_paths.root / log_path
                job_kwargs["log_directory"] = str(log_path)
            log_path.mkdir(parents=True, exist_ok=True)
        if staged_files:
            job_kwargs = _merge_htcondor_job_kwargs(
                _htcondor_staged_execution_kwargs(staged_files),
                job_kwargs,
            )
            spec["job_kwargs"] = job_kwargs
        job_kwargs["submit_directory"] = str(_dask_htcondor_dir(build_paths, "submit"))
        pool_directives = dict(job_kwargs.get("job_extra_directives") or {})
        bootstrap_directives = _htcondor_bootstrap_directives(paths)
        if _has_output_destination(pool_directives):
            bootstrap_directives.pop("transfer_output_files")
        job_kwargs["job_extra_directives"] = _merge_htcondor_directives(
            pool_directives,
            bootstrap_directives,
            f"HTCondor worker pool {name!r}",
        )
    preparation_progress.complete(
        prepared=worker_environment,
        staged=staged_files,
        staging_mode=staging_mode,
    )
    return prepared


def _htcondor_execution_paths(build_paths: BuildPaths) -> dict[str, Path]:
    return {
        "logs": _dask_htcondor_dir(build_paths, "logs"),
        "out": _dask_htcondor_dir(build_paths, "out"),
        "err": _dask_htcondor_dir(build_paths, "err"),
    }


def _dask_htcondor_dir(build_paths: BuildPaths, kind: str) -> Path:
    return build_paths.execution_backend_dir("dask", "htcondor", kind)


def _htcondor_bootstrap_directives(paths: dict[str, Path]) -> dict[str, str]:
    return {
        **_HTCONDOR_FIXED_DIRECTIVES,
        "Output": str((paths["out"] / "worker-$(ClusterId).$(ProcId).out").resolve()),
        "Error": str((paths["err"] / "worker-$(ClusterId).$(ProcId).err").resolve()),
        "Log": str((paths["logs"] / "worker-$(ClusterId).log").resolve()),
    }


def _htcondor_staged_execution_kwargs(
    staging: StagedExecutionFiles,
) -> dict[str, object]:
    validate_transfer_file_basenames(staging.transfer_files)
    job_directives: dict[str, str] = {}
    if staging.transfer_files:
        transfer_files = ",".join(str(path.resolve()) for path in staging.transfer_files)
        job_directives.update(
            {
                "should_transfer_files": "YES",
                "when_to_transfer_output": "ON_EXIT",
                "transfer_input_files": transfer_files,
            }
        )
    return {
        "python": staging.python,
        "job_extra_directives": job_directives,
        "job_script_prologue": list(staging.bootstrap_commands),
    }


def _merge_htcondor_job_kwargs(
    common: dict[str, object],
    pool: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(common)
    for key, value in pool.items():
        if key == "job_extra_directives":
            current = merged.get(key)
            current_directives = current if isinstance(current, dict) else {}
            pool_directives = value if isinstance(value, dict) else {}
            merged[key] = _merge_htcondor_directives(
                current_directives,
                pool_directives,
                "HTCondor worker environment",
            )
        elif key in {"job_script_prologue", "worker_extra_args"}:
            current = merged.get(key)
            current_items = current if isinstance(current, list) else []
            pool_items = value if isinstance(value, list) else []
            merged[key] = [*current_items, *pool_items]
        else:
            merged[key] = value
    return merged


def _htcondor_cluster_options_for_pool(pool: DaskWorkerPool) -> dict[str, Any]:
    return _htcondor_cluster_options(
        resources=pool.resources,
        config=pool.config,
        dask_resources=pool.dask_resources,
    )


def _htcondor_cluster_options(
    *,
    resources: dict[str, Any],
    config: dict[str, Any],
    dask_resources: dict[str, Any],
) -> dict[str, Any]:
    _validate_htcondor_config_options(config)

    cores = resources.get("cpus", config.get("cores"))
    if cores is not None:
        cores = int(cores)

    log_directory = config.get("log_directory")

    cluster_options: dict[str, Any] = {}
    if cores is not None:
        cluster_options["cores"] = cores
    if resources.get("memory") is not None:
        cluster_options["memory"] = resources["memory"]
    if resources.get("disk") is not None:
        cluster_options["disk"] = resources["disk"]
    if log_directory is not None:
        cluster_options["log_directory"] = log_directory

    nanny = config.get("nanny")
    if nanny is not None:
        if not isinstance(nanny, bool):
            raise ValueError("execution.config.nanny must be a boolean")
        cluster_options["nanny"] = nanny

    submit_command_extra = config.get("submit_command_extra")
    if submit_command_extra is not None:
        if not isinstance(submit_command_extra, list) or not all(
            isinstance(item, str) for item in submit_command_extra
        ):
            raise ValueError(
                "execution.config.submit_command_extra must be a list of strings"
            )
        cluster_options["submit_command_extra"] = list(submit_command_extra)

    job_extra_directives: dict[str, Any] = dict(_HTCONDOR_FIXED_DIRECTIVES)
    if config.get("queue") is not None:
        job_extra_directives["+JobFlavour"] = f'"{config["queue"]}"'
    # Site-specific ClassAds (e.g. CERN's MY.IsDaskWorker) come from site
    # profiles. They are kept apart from job_extra_directives so that a
    # workflow setting its own directives does not drop them, since Flow merges
    # execution.config only one level deep. User directives win on conflicts.
    if config.get("site_directives") is not None:
        site_directives = config["site_directives"]
        if not isinstance(site_directives, dict):
            raise ValueError("execution.config.site_directives must be a mapping")
        job_extra_directives = _merge_htcondor_directives(
            job_extra_directives,
            site_directives,
            "execution.config.site_directives",
        )
    output_directives = _output_transfer_directives(config)
    if output_directives:
        job_extra_directives = _merge_htcondor_directives(
            job_extra_directives,
            output_directives,
            "execution.config.output_destination",
        )
    if config.get("job_extra_directives") is not None:
        raw_directives = config["job_extra_directives"]
        if not isinstance(raw_directives, dict):
            raise ValueError("execution.config.job_extra_directives must be a mapping")
        job_extra_directives = _merge_htcondor_directives(
            job_extra_directives,
            raw_directives,
            "execution.config.job_extra_directives",
        )
    if resources.get("gpus") is not None:
        job_extra_directives.setdefault("request_gpus", resources["gpus"])
    if job_extra_directives:
        cluster_options["job_extra_directives"] = job_extra_directives

    worker_extra_args = _worker_extra_args(config.get("worker_extra_args"))
    worker_extra_args.extend(
        _worker_port_args(config.get("worker_port_range"), nanny=nanny is not False)
    )
    worker_extra_args.extend(dask_worker_resource_args(dask_resources))
    if worker_extra_args:
        cluster_options["worker_extra_args"] = worker_extra_args

    prologue = _output_directory_prologue(config)
    if prologue:
        cluster_options["job_script_prologue"] = prologue

    return cluster_options


def _validate_htcondor_config_options(config: dict[str, Any]) -> None:
    if config.get("walltime") is not None:
        raise ValueError(
            "execution.config.walltime is not supported by the Dask HTCondor "
            "backend. Use execution.config.queue for site job flavours or "
            "execution.config.job_extra_directives for explicit HTCondor ClassAds."
        )


def _batch_name(config: dict[str, Any]) -> str | None:
    raw = config.get("batch_name")
    if raw is None:
        return None
    if not isinstance(raw, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", raw):
        raise ValueError(
            "execution.config.batch_name must be a label of letters, digits, "
            f"'.', '_' and '-', got {raw!r}"
        )
    return raw


def _worker_start_timeout(config: dict[str, Any]) -> float:
    raw = config.get("worker_start_timeout", 120)
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution.config.worker_start_timeout must be numeric") from exc
    if timeout <= 0:
        raise ValueError("execution.config.worker_start_timeout must be positive")
    return timeout


def _worker_shutdown_grace(config: dict[str, Any]) -> float:
    raw = config.get("worker_shutdown_grace", DEFAULT_WORKER_SHUTDOWN_GRACE)
    try:
        grace = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution.config.worker_shutdown_grace must be numeric") from exc
    if grace < 0:
        raise ValueError("execution.config.worker_shutdown_grace must not be negative")
    return grace


def _wait_for_htcondor_workers(
    client: Any,
    *,
    requested: int,
    pool_names: list[str],
    timeout: float,
    build_paths: BuildPaths,
) -> None:
    if requested <= 0:
        return
    try:
        client.wait_for_workers(requested, timeout=timeout)
    except Exception as exc:
        connected = _connected_worker_count(client)
        paths = _htcondor_execution_paths(build_paths)
        logs = ", ".join(
            f"{name}={path.resolve()}" for name, path in sorted(paths.items())
        )
        pools = ", ".join(pool_names) if pool_names else "<none>"
        raise TimeoutError(
            "Timed out waiting for Dask HTCondor workers: "
            f"requested={requested}, connected={connected}, pools={pools}, "
            f"timeout={timeout:g}s, logs=({logs})"
        ) from exc


def _connected_worker_count(client: Any) -> int:
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return 0
    return len(workers)


def _merge_htcondor_directives(
    base: dict[str, Any],
    overlay: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    merged = dict(base)
    fixed_directives = dict(_HTCONDOR_FIXED_DIRECTIVES)
    if _has_output_destination(base) or _has_output_destination(overlay):
        # With an output destination the files to send are chosen by the user.
        fixed_directives.pop("transfer_output_files")
    fixed_by_lower = {key.lower(): key for key in fixed_directives}
    runtime_by_lower = {key.lower(): key for key in _HTCONDOR_RUNTIME_DIRECTIVES}
    for key, value in overlay.items():
        key_lower = str(key).lower()
        fixed_key = fixed_by_lower.get(key_lower)
        if fixed_key is not None:
            expected = fixed_directives[fixed_key]
            if str(value) != expected:
                raise ValueError(
                    f"{source} cannot override HTCondor invariant {fixed_key}={expected}"
                )
        runtime_key = runtime_by_lower.get(key_lower)
        if runtime_key is not None and key in base and str(value) != str(base[key]):
            raise ValueError(
                f"{source} cannot override HTCondor invariant {runtime_key}"
            )
        merged[key] = value
    return merged


def _has_output_destination(directives: dict[str, Any]) -> bool:
    return any(str(key).lower() == "output_destination" for key in directives)


def _output_transfer_directives(config: dict[str, Any]) -> dict[str, str]:
    """
    HTCondor directives for sending job output files to a remote destination.

    ``execution.config.output_destination`` is an XRootD URL; at CERN this uses
    the EOS file-transfer plugin (Kerberos). ``execution.config.output_files``
    lists what to send, relative to the job directory. HTCondor copies these
    when the job exits, so outputs of a long-lived worker arrive only then.
    """

    destination = config.get("output_destination")
    files = config.get("output_files")
    if destination is None:
        if files is not None:
            raise ValueError(
                "execution.config.output_files requires execution.config.output_destination"
            )
        return {}
    if not isinstance(destination, str) or not destination.startswith(
        ("root://", "xroot://")
    ):
        raise ValueError("execution.config.output_destination must be a root:// URL")
    if "/eos/" not in destination:
        raise ValueError(
            "execution.config.output_destination must point into /eos/ "
            "(the CERN file-transfer plugin only supports EOS)"
        )
    if not isinstance(files, list) or not files:
        raise ValueError(
            "execution.config.output_files must be a non-empty list of names "
            "when execution.config.output_destination is set"
        )
    _validate_relative_names(files, "execution.config.output_files")
    return {
        "output_destination": destination,
        "transfer_output_files": ", ".join(files),
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "MY.XRDCP_CREATE_DIR": "True",
    }


def _validate_relative_names(items: list[Any], key: str) -> None:
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} entries must be non-empty strings")
        parts = Path(item).parts
        if (
            Path(item).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or any(ch in item for ch in ", \t\n\"'")
        ):
            raise ValueError(
                f"{key} entry {item!r} must be a simple relative path in the job "
                "directory"
            )


def _output_directory_prologue(config: dict[str, Any]) -> list[str]:
    """
    ``mkdir`` lines creating ``execution.config.output_directories`` in the job
    directory before the worker starts.

    HTCondor is told to send these directories; a worker that never wrote into
    one would otherwise have nothing to send. One command per line: dask-jobqueue
    joins the prologue with ``;``.
    """

    directories = config.get("output_directories")
    if directories is None:
        return []
    if config.get("output_destination") is None:
        raise ValueError(
            "execution.config.output_directories requires "
            "execution.config.output_destination"
        )
    if not isinstance(directories, list):
        raise ValueError("execution.config.output_directories must be a list of names")
    _validate_relative_names(directories, "execution.config.output_directories")
    return [f"mkdir -p -- {shlex.quote(item)}" for item in directories]


def _worker_extra_args(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("execution.config.worker_extra_args must be a list of strings")
    return list(raw)


def _scheduler_options(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("scheduler_options")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("execution.config.scheduler_options must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in _SCHEDULER_OPTION_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported execution.config.scheduler_options: " + ", ".join(unknown)
        )
    port = raw.get("port")
    if port is not None and (
        isinstance(port, bool) or not isinstance(port, int) or not 0 <= port < 65536
    ):
        raise ValueError(
            "execution.config.scheduler_options.port must be a port number"
        )
    return dict(raw)


def _worker_port_args(raw: Any, *, nanny: bool) -> list[str]:
    """
    Render a site's open port range for the worker and, if enabled, its nanny.

    Kept as a dedicated option so that a workflow's own worker_extra_args do not
    replace the range set by a site profile.
    """

    if raw is None:
        return []
    port_range = str(raw)
    bounds = port_range.split(":")
    if (
        isinstance(raw, bool)
        or len(bounds) > 2
        or not all(item.isdigit() for item in bounds)
        or not all(0 < int(item) < 65536 for item in bounds)
        or (len(bounds) == 2 and int(bounds[0]) > int(bounds[1]))
    ):
        raise ValueError(
            "execution.config.worker_port_range must be a port or a range "
            f"'low:high', got {raw!r}"
        )
    args = ["--worker-port", port_range]
    if nanny:
        args.extend(["--nanny-port", port_range])
    return args


def _pool_summary(pool: DaskWorkerPool) -> dict[str, Any]:
    return {
        "name": pool.name,
        "resources": pool.resource_name,
        "workers": pool.workers,
        "dask_resources": pool.dask_resources,
        "cluster_options": _htcondor_cluster_options_for_pool(pool),
    }
