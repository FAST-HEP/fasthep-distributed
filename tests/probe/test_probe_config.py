from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fasthep_distributed.probe import (
    PathSpec,
    ProbeConfigError,
    load_probe_config,
    probe_config_from_mapping,
)


def _raw(**extra: Any) -> dict[str, Any]:
    return {
        "prefix": "/afs/example/env",
        "probe_root": "/afs/example/probes",
        **extra,
    }


def test_minimal_config_uses_defaults() -> None:
    config = probe_config_from_mapping(_raw())

    assert config.prefix == Path("/afs/example/env")
    assert config.python == Path("/afs/example/env/bin/python")
    assert config.required_imports == ("distributed", "fasthep_distributed", "hepflow")
    assert config.optional_imports == ()
    assert config.paths == ()
    assert config.condor.flavour == "espresso"
    assert config.condor.count == 1
    assert config.timeouts["paths"] == 30.0


def test_relative_paths_resolve_against_base_dir(tmp_path: Path) -> None:
    config = probe_config_from_mapping(
        {
            "prefix": "../.pixi/envs/dev",
            "probe_root": "build/probes",
            "lockfile": "../pixi.lock",
            "paths": [{"name": "here", "path": "."}],
        },
        base_dir=tmp_path / "cern-testing",
    )

    assert config.prefix == tmp_path / ".pixi" / "envs" / "dev"
    assert config.probe_root == tmp_path / "cern-testing" / "build" / "probes"
    assert config.lockfile == tmp_path / "pixi.lock"
    assert config.paths == (PathSpec(name="here", path=str(tmp_path / "cern-testing")),)


def test_probe_root_on_eos_is_rejected() -> None:
    with pytest.raises(ProbeConfigError, match="must not live there"):
        probe_config_from_mapping(_raw(probe_root="/eos/user/u/user/probes"))


def test_eos_paths_may_be_probed() -> None:
    config = probe_config_from_mapping(
        _raw(paths=[{"name": "eos", "path": "/eos/user/u/user", "required": False}])
    )

    assert config.paths == (PathSpec("eos", "/eos/user/u/user", required=False),)


def test_required_submit_prefix_is_enforced() -> None:
    with pytest.raises(ProbeConfigError, match="must be under /afs"):
        probe_config_from_mapping(
            _raw(probe_root="/tmp/probes", submit_dir={"required_prefix": "/afs"})
        )


def test_prefix_match_is_by_path_component() -> None:
    config = probe_config_from_mapping(_raw(probe_root="/eosfoo/probes"))

    assert config.probe_root == Path("/eosfoo/probes")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_raw(unexpected=True), "unknown keys in probe config: unexpected"),
        ({"probe_root": "/afs/x"}, "requires 'prefix'"),
        (_raw(imports={"required": "hepflow"}), "list of strings"),
        (
            _raw(imports={"required": ["hepflow"], "optional": ["hepflow"]}),
            "both required and optional",
        ),
        (
            _raw(paths=[{"name": "a", "path": "/a"}, {"name": "a", "path": "/b"}]),
            "duplicate path name",
        ),
        (_raw(paths=[{"name": "a b", "path": "/a"}]), "must match"),
        (_raw(paths=[{"name": "a", "path": "/a", "required": "no"}]), "boolean"),
        (_raw(condor={"count": 0}), "positive integer"),
        (_raw(condor={"queue": "workday"}), "unknown keys in condor"),
        (_raw(timeouts={"paths": -1}), "positive number"),
        (_raw(timeouts={"network": 5}), "unknown keys in timeouts"),
    ],
)
def test_invalid_configs_are_rejected(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(ProbeConfigError, match=message):
        probe_config_from_mapping(raw)


def test_worker_config_is_json_ready() -> None:
    config = probe_config_from_mapping(
        _raw(
            imports={"required": ["hepflow"], "optional": ["fasthep"]},
            paths=[{"name": "work", "path": "/afs/work"}],
            timeouts={"paths": 5},
        )
    )

    assert config.worker_config("probe-1") == {
        "probe_id": "probe-1",
        "prefix": "/afs/example/env",
        "imports": {"required": ["hepflow"], "optional": ["fasthep"]},
        "paths": [{"name": "work", "path": "/afs/work", "required": True}],
        "timeouts": {"imports": 300.0, "paths": 5.0, "git": 60.0, "scratch": 60.0},
    }


def test_load_probe_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "probe.yaml"
    path.write_text(
        "prefix: env\nprobe_root: probes\ncondor:\n  flavour: microcentury\n",
        encoding="utf-8",
    )

    config = load_probe_config(path)

    assert config.prefix == tmp_path / "env"
    assert config.condor.flavour == "microcentury"


def test_load_probe_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "probe.yaml"
    path.write_text("- a\n", encoding="utf-8")

    with pytest.raises(ProbeConfigError, match="must be a mapping"):
        load_probe_config(path)
