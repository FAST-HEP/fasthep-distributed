from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._worker import DEFAULT_TIMEOUTS

DEFAULT_REQUIRED_IMPORTS = ("distributed", "fasthep_distributed", "hepflow")
DEFAULT_FORBIDDEN_SUBMIT_PREFIXES = ("/eos",)

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TOP_LEVEL_KEYS = {
    "prefix",
    "probe_root",
    "lockfile",
    "imports",
    "paths",
    "condor",
    "submit_dir",
    "timeouts",
}
_CONDOR_KEYS = {
    "flavour",
    "request_cpus",
    "request_memory",
    "request_disk",
    "count",
    "extra_directives",
}


class ProbeConfigError(ValueError):
    """Raised when a probe configuration is invalid."""


@dataclass(slots=True, frozen=True)
class PathSpec:
    name: str
    path: str
    required: bool = True


@dataclass(slots=True, frozen=True)
class CondorSpec:
    flavour: str | None = "espresso"
    request_cpus: int = 1
    request_memory: str = "2GB"
    request_disk: str | None = None
    count: int = 1
    extra_directives: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProbeConfig:
    prefix: Path
    probe_root: Path
    lockfile: Path | None = None
    required_imports: tuple[str, ...] = DEFAULT_REQUIRED_IMPORTS
    optional_imports: tuple[str, ...] = ()
    paths: tuple[PathSpec, ...] = ()
    condor: CondorSpec = field(default_factory=CondorSpec)
    forbidden_submit_prefixes: tuple[str, ...] = DEFAULT_FORBIDDEN_SUBMIT_PREFIXES
    required_submit_prefix: str | None = None
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))

    @property
    def python(self) -> Path:
        return self.prefix / "bin" / "python"

    def worker_config(self, probe_id: str) -> dict[str, Any]:
        """Return the JSON-serialisable configuration passed to ``_worker.py``."""

        return {
            "probe_id": probe_id,
            "prefix": str(self.prefix),
            "imports": {
                "required": list(self.required_imports),
                "optional": list(self.optional_imports),
            },
            "paths": [
                {"name": item.name, "path": item.path, "required": item.required}
                for item in self.paths
            ],
            "timeouts": dict(self.timeouts),
        }


def load_probe_config(path: Path) -> ProbeConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeConfigError(f"Cannot read probe config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProbeConfigError(f"Invalid YAML in probe config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ProbeConfigError(f"Probe config {path} must be a mapping")
    return probe_config_from_mapping(raw, base_dir=path.parent)


def probe_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> ProbeConfig:
    """
    Build a validated :class:`ProbeConfig`.

    Relative paths are resolved against ``base_dir`` (normally the directory of
    the config file). Paths are normalised lexically and never resolved through
    the filesystem, because probing a hung network mount must not hang here.
    """

    _reject_unknown(raw, _TOP_LEVEL_KEYS, "probe config")
    base = Path(os.path.normpath(Path.cwd() / (base_dir or Path())))

    if "prefix" not in raw:
        raise ProbeConfigError("probe config requires 'prefix'")
    if "probe_root" not in raw:
        raise ProbeConfigError("probe config requires 'probe_root'")
    prefix = _absolute(raw["prefix"], base, "prefix")
    probe_root = _absolute(raw["probe_root"], base, "probe_root")
    lockfile = (
        _absolute(raw["lockfile"], base, "lockfile")
        if raw.get("lockfile") is not None
        else None
    )

    imports = raw.get("imports") or {}
    if not isinstance(imports, Mapping):
        raise ProbeConfigError("'imports' must be a mapping")
    _reject_unknown(imports, {"required", "optional"}, "imports")
    required_imports = _string_tuple(
        imports.get("required", DEFAULT_REQUIRED_IMPORTS), "imports.required"
    )
    optional_imports = _string_tuple(imports.get("optional", ()), "imports.optional")
    overlap = sorted(set(required_imports) & set(optional_imports))
    if overlap:
        raise ProbeConfigError(
            f"imports listed as both required and optional: {', '.join(overlap)}"
        )

    submit_dir = raw.get("submit_dir") or {}
    if not isinstance(submit_dir, Mapping):
        raise ProbeConfigError("'submit_dir' must be a mapping")
    _reject_unknown(submit_dir, {"forbidden_prefixes", "required_prefix"}, "submit_dir")
    forbidden = _string_tuple(
        submit_dir.get("forbidden_prefixes", DEFAULT_FORBIDDEN_SUBMIT_PREFIXES),
        "submit_dir.forbidden_prefixes",
    )
    required_prefix = submit_dir.get("required_prefix")
    if required_prefix is not None and not isinstance(required_prefix, str):
        raise ProbeConfigError("'submit_dir.required_prefix' must be a string")
    _validate_submit_dir(probe_root, forbidden, required_prefix)

    return ProbeConfig(
        prefix=prefix,
        probe_root=probe_root,
        lockfile=lockfile,
        required_imports=required_imports,
        optional_imports=optional_imports,
        paths=_path_specs(raw.get("paths") or [], base),
        condor=_condor_spec(raw.get("condor") or {}),
        forbidden_submit_prefixes=forbidden,
        required_submit_prefix=required_prefix,
        timeouts=_timeouts(raw.get("timeouts") or {}),
    )


def _validate_submit_dir(
    probe_root: Path,
    forbidden: tuple[str, ...],
    required_prefix: str | None,
) -> None:
    for prefix in forbidden:
        if _is_under(probe_root, prefix):
            raise ProbeConfigError(
                f"probe_root {probe_root} is under {prefix}; HTCondor submit files "
                "and logs must not live there"
            )
    if required_prefix is not None and not _is_under(probe_root, required_prefix):
        raise ProbeConfigError(
            f"probe_root {probe_root} must be under {required_prefix}"
        )


def _path_specs(raw: Any, base: Path) -> tuple[PathSpec, ...]:
    if not isinstance(raw, list):
        raise ProbeConfigError("'paths' must be a list")
    specs: list[PathSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ProbeConfigError(f"paths[{index}] must be a mapping")
        _reject_unknown(item, {"name", "path", "required"}, f"paths[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            raise ProbeConfigError(
                f"paths[{index}].name must match {_NAME_PATTERN.pattern}"
            )
        if name in seen:
            raise ProbeConfigError(f"duplicate path name {name!r}")
        seen.add(name)
        if "path" not in item:
            raise ProbeConfigError(f"paths[{index}] requires 'path'")
        required = item.get("required", True)
        if not isinstance(required, bool):
            raise ProbeConfigError(f"paths[{index}].required must be a boolean")
        specs.append(
            PathSpec(
                name=name,
                path=str(_absolute(item["path"], base, f"paths[{index}].path")),
                required=required,
            )
        )
    return tuple(specs)


def _condor_spec(raw: Any) -> CondorSpec:
    if not isinstance(raw, Mapping):
        raise ProbeConfigError("'condor' must be a mapping")
    _reject_unknown(raw, _CONDOR_KEYS, "condor")
    defaults = CondorSpec()
    count = raw.get("count", defaults.count)
    cpus = raw.get("request_cpus", defaults.request_cpus)
    for key, value in (("count", count), ("request_cpus", cpus)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ProbeConfigError(f"'condor.{key}' must be a positive integer")
    extra = raw.get("extra_directives") or {}
    if not isinstance(extra, Mapping):
        raise ProbeConfigError("'condor.extra_directives' must be a mapping")
    flavour = raw.get("flavour", defaults.flavour)
    disk = raw.get("request_disk", defaults.request_disk)
    return CondorSpec(
        flavour=str(flavour) if flavour is not None else None,
        request_cpus=cpus,
        request_memory=str(raw.get("request_memory", defaults.request_memory)),
        request_disk=str(disk) if disk is not None else None,
        count=count,
        extra_directives={str(key): str(value) for key, value in extra.items()},
    )


def _timeouts(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ProbeConfigError("'timeouts' must be a mapping")
    _reject_unknown(raw, set(DEFAULT_TIMEOUTS), "timeouts")
    timeouts = dict(DEFAULT_TIMEOUTS)
    for key, value in raw.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ProbeConfigError(f"'timeouts.{key}' must be a positive number")
        timeouts[key] = float(value)
    return timeouts


def _absolute(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str | os.PathLike) or not str(value).strip():
        raise ProbeConfigError(f"'{label}' must be a non-empty path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.normpath(path))


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ProbeConfigError(f"'{label}' must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ProbeConfigError(f"'{label}' must be a list of non-empty strings")
    return tuple(value)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ProbeConfigError(f"unknown keys in {label}: {', '.join(unknown)}")


def _is_under(path: Path, prefix: str) -> bool:
    normalised = Path(os.path.normpath(prefix))
    return path == normalised or normalised in path.parents
