from __future__ import annotations

import json
import tarfile
import types
from pathlib import Path
from typing import Any, cast

import pytest
from hepflow.build_layout import BuildPaths
from hepflow.model.plan import ExecutionPartition, ExecutionPlan

from fasthep_distributed._dask._common import DaskBackend
from fasthep_distributed._dask._worker_env import (
    PreparedWorkerEnvironment,
    build_compile_bundle,
    prepare_staged_execution_files,
)


def _names(archive: Path) -> set[str]:
    with tarfile.open(archive) as handle:
        return set(handle.getnames())


def _per_dataset_compile(compile_dir: Path, *units: str) -> None:
    compile_dir.mkdir(parents=True)
    for name in ["plan.yaml", "deps.yaml", "worker_environment.json"]:
        (compile_dir / name).write_text(name, encoding="utf-8")
    for unit in units:
        directory = compile_dir / "datasets" / unit
        directory.mkdir(parents=True)
        for name in ["manifest.yaml", "normalized.yaml", "dataset_entries.json"]:
            (directory / name).write_text(f"{unit}:{name}", encoding="utf-8")
        (directory / "dataset_metadata.json").write_text("{}", encoding="utf-8")


def test_flat_compile_directory_keeps_the_fixed_bundle(tmp_path: Path) -> None:
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir()
    for name in ["plan.yaml", "normalized.yaml", "deps.yaml", "unlisted.txt"]:
        (compile_dir / name).write_text(name, encoding="utf-8")

    archive = build_compile_bundle(compile_dir, tmp_path / "out" / "compile.tar.gz")

    assert _names(archive) == {"plan.yaml", "normalized.yaml", "deps.yaml"}


def test_per_dataset_bundle_holds_the_shared_files_and_exactly_the_named_units(
    tmp_path: Path,
) -> None:
    compile_dir = tmp_path / "compile"
    _per_dataset_compile(compile_dir, "alpha", "beta")

    archive = build_compile_bundle(compile_dir, tmp_path / "compile.tar.gz", datasets=["alpha"])

    assert _names(archive) == {
        "plan.yaml",
        "deps.yaml",
        "worker_environment.json",
        "datasets/alpha",
        "datasets/alpha/manifest.yaml",
        "datasets/alpha/normalized.yaml",
        "datasets/alpha/dataset_entries.json",
        "datasets/alpha/dataset_metadata.json",
    }


def test_several_datasets_carry_each_unit_once(tmp_path: Path) -> None:
    compile_dir = tmp_path / "compile"
    _per_dataset_compile(compile_dir, "alpha", "beta")

    archive = build_compile_bundle(
        compile_dir, tmp_path / "compile.tar.gz", datasets=["beta", "alpha", "beta"]
    )

    names = _names(archive)
    assert {"datasets/alpha/manifest.yaml", "datasets/beta/manifest.yaml"} <= names


def test_a_missing_or_incomplete_unit_is_an_error(tmp_path: Path) -> None:
    compile_dir = tmp_path / "compile"
    _per_dataset_compile(compile_dir, "alpha")
    (compile_dir / "datasets" / "alpha" / "manifest.yaml").unlink()

    with pytest.raises(FileNotFoundError, match=r"manifest.yaml is missing.*--dataset alpha"):
        build_compile_bundle(compile_dir, tmp_path / "a.tar.gz", datasets=["alpha"])
    with pytest.raises(FileNotFoundError, match="--dataset gamma"):
        build_compile_bundle(compile_dir, tmp_path / "b.tar.gz", datasets=["gamma"])


@pytest.mark.parametrize("name", ["", "..", "a/b"])
def test_a_dataset_name_cannot_leave_the_datasets_directory(tmp_path: Path, name: str) -> None:
    _per_dataset_compile(tmp_path / "compile", "alpha")

    with pytest.raises(ValueError, match="cannot name a unit directory"):
        build_compile_bundle(tmp_path / "compile", tmp_path / "c.tar.gz", datasets=[name])


def test_transfer_staging_carries_the_unit_and_lists_the_archive(tmp_path: Path) -> None:
    build_paths = BuildPaths(root=tmp_path / "build")
    _per_dataset_compile(build_paths.compile_dir(), "alpha", "beta")
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    archive, bootstrap, manifest = (
        env_dir / "prefix.tar.gz",
        env_dir / "bootstrap.sh",
        env_dir / "manifest.json",
    )
    for path in (archive, bootstrap, manifest):
        path.write_text("x", encoding="utf-8")
    prepared = PreparedWorkerEnvironment(
        python="./worker-env/bin/python",
        bootstrap_commands=[],
        transfer_files=[],
        env={},
        environment_manifest=manifest,
        environment_archive=archive,
        bootstrap_script=bootstrap,
    )

    staged = prepare_staged_execution_files(
        prepared,
        build_paths=build_paths,
        staging={"mode": "transfer"},
        datasets=["alpha"],
    )

    assert staged is not None
    bundle = next(p for p in staged.transfer_files if p.name == "compile.tar.gz")
    names = _names(bundle)
    assert "datasets/alpha/manifest.yaml" in names
    assert not any(name.startswith("datasets/beta") for name in names)
    assert json.loads(staged.manifest.read_text(encoding="utf-8"))["compile"]["sha256"]


def _run_dispatch(plan: ExecutionPlan, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_compute(tasks: list[Any], **kwargs: Any) -> tuple[list[Any], None, dict[str, Any]]:
        calls.update(kwargs)
        return [], None, {"workers": 1, "cluster_options": {}}

    monkeypatch.setattr("fasthep_distributed._dask._htcondor.compute_with_htcondor", fake_compute)
    dask = types.ModuleType("dask")
    cast(Any, dask).compute = lambda *tasks, **_kwargs: tasks
    cast(Any, dask).delayed = lambda func: func
    monkeypatch.setitem(__import__("sys").modules, "dask", dask)
    DaskBackend().run(plan)
    return calls


def _partition(dataset: str, number: int) -> ExecutionPartition:
    return ExecutionPartition(
        id=f"s__{dataset}__{number}",
        dataset=dataset,
        file="f",
        source="s",
        part=str(number),
    )


def test_a_per_dataset_plan_passes_its_datasets_to_the_htcondor_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = ExecutionPlan(
        execution={
            "backend": "dask",
            "strategy": "htcondor",
            "partitioning": {"manifest": "datasets/{dataset}/manifest.yaml"},
            "config": {"workers": 1},
        },
        context={"outdir": "."},
    )
    plan.partitions = [_partition("beta", 1), _partition("alpha", 1), _partition("alpha", 2)]

    assert _run_dispatch(plan, monkeypatch)["datasets"] == ["alpha", "beta"]


def test_a_flat_plan_passes_no_datasets_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = ExecutionPlan(
        execution={"backend": "dask", "strategy": "htcondor", "config": {"workers": 1}},
        context={"outdir": "."},
    )

    assert "datasets" not in _run_dispatch(plan, monkeypatch)
