from __future__ import annotations

import subprocess
from typing import Any

import pytest
from hepflow.model.io import ArtifactManifest, OutputResult
from hepflow.model.plan import ExecutionNode, ExecutionPartition, ExecutionPlan

from fasthep_distributed._dask._htcondor import normalize_dask_htcondor_config
from fasthep_distributed._dask._output_delivery import (
    delivered_writer_records,
    execution_with_output_roots,
    output_destination,
    partition_task_context,
    split_destination,
    verify_delivery,
)

DESTINATION = "root://eosuser.cern.ch//eos/user/x/y/skim/"


def _plan(path: str = "{dataset}/{part}.root", datasets: tuple[str, ...] = ("a", "b")):
    plan = ExecutionPlan()
    plan.add_node(
        ExecutionNode(
            id="write.Selected",
            graph_node_id="write.Selected",
            role="sink",
            impl="root_tree",
            params={"path": path},
            outputs={"artifact": "artifact"},
        )
    )
    plan.partitions = [
        ExecutionPartition(
            id=f"events__{name}__0", dataset=name, file="f.root", source="events", part="0_0"
        )
        for name in datasets
    ]
    return plan


def _execution(**config: Any) -> dict[str, Any]:
    return {
        "backend": "dask",
        "strategy": "htcondor",
        "resources": {"default": {"cpus": 1}},
        "config": {"workers": 1, **config},
    }


def test_output_roots_are_derived_from_the_plan() -> None:
    execution = execution_with_output_roots(
        _plan(), _execution(output_destination=DESTINATION)
    )

    assert execution["config"]["output_files"] == ["a", "b"]
    assert execution["config"]["output_directories"] == ["a", "b"]


def test_explicit_output_files_are_kept_and_not_created_as_directories() -> None:
    original = _execution(output_destination=DESTINATION, output_files=["manifest.json"])

    execution = execution_with_output_roots(_plan(), original)

    assert execution is original
    assert "output_directories" not in execution["config"]


def test_without_destination_the_execution_is_untouched() -> None:
    original = _execution()

    assert execution_with_output_roots(_plan(), original) is original


def test_destination_without_partition_writers_is_an_error() -> None:
    plan = _plan(path="/data/out/{part}.root")

    with pytest.raises(ValueError, match="no partition writer"):
        execution_with_output_roots(plan, _execution(output_destination=DESTINATION))


def test_partition_task_context_points_writers_at_the_job_directory() -> None:
    execution = _execution(output_destination=DESTINATION)

    assert partition_task_context(execution) == {
        "output_files_root": ".",
        "output_destination": DESTINATION,
    }
    assert output_destination(execution) == DESTINATION
    assert partition_task_context(_execution()) == {}


def test_derived_output_directories_become_job_prologue_commands() -> None:
    execution = execution_with_output_roots(
        _plan(), _execution(output_destination=DESTINATION)
    )

    options = normalize_dask_htcondor_config(execution)["pool_specs"]["default"][
        "job_kwargs"
    ]

    assert options["job_script_prologue"] == ["mkdir -p -- a", "mkdir -p -- b"]
    assert options["job_extra_directives"]["transfer_output_files"] == "a, b"


def test_output_directory_names_are_shell_quoted() -> None:
    options = normalize_dask_htcondor_config(
        _execution(
            output_destination=DESTINATION,
            output_files=["a"],
            output_directories=["a;b"],
        )
    )["pool_specs"]["default"]["job_kwargs"]

    assert options["job_script_prologue"] == ["mkdir -p -- 'a;b'"]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"output_directories": ["a"]}, "requires execution.config.output_destination"),
        (
            {"output_destination": DESTINATION, "output_files": ["a"], "output_directories": "a"},
            "must be a list",
        ),
        (
            {"output_destination": DESTINATION, "output_files": ["a"], "output_directories": ["../a"]},
            "simple relative path",
        ),
    ],
)
def test_output_directories_are_validated(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_dask_htcondor_config(_execution(**config))


def _record(path: str, *, size: int = 7, adler32: str | None = "0a0b0c0d") -> OutputResult:
    manifest = {
        "kind": "root_tree",
        "path": path,
        "path_type": "relative_to_files_root",
        "size_bytes": size,
        "dataset": path.split("/", maxsplit=1)[0],
        "partition": 0,
    }
    if adler32 is not None:
        manifest["adler32"] = adler32
    return OutputResult(
        kind="artifact",
        path=path,
        metadata={"writer_manifest": manifest},
        dataset_name=path.split("/", maxsplit=1)[0],
    )


def test_delivered_writer_records_collect_files_below_the_files_root() -> None:
    local = OutputResult(
        kind="artifact",
        path="x",
        metadata={"writer_manifest": {"path": "x", "path_type": "relative_to_outdir"}},
    )
    manifest = ArtifactManifest(
        product_kind="artifact",
        format="root",
        producer_node="write.Selected",
        output_name="artifact",
        dataset_name="b",
        parts=(_record("b/0_0.root"), _record("b/1_0.root")),
    )
    results: list[dict[str, Any]] = [
        {"value_store": {("write.Selected", "artifact"): _record("a/0_0.root")}},
        {"value_store": {("write.Selected", "artifact"): manifest}},
        {"value_store": {("write.Selected", "artifact"): local}},
        {"value_store": {}},
    ]

    records = delivered_writer_records(results)

    assert sorted(item["path"] for item in records) == [
        "a/0_0.root",
        "b/0_0.root",
        "b/1_0.root",
    ]


@pytest.mark.parametrize(
    ("destination", "expected"),
    [
        (DESTINATION, ("root://eosuser.cern.ch", "/eos/user/x/y/skim")),
        ("root://eosuser.cern.ch//eos/user/x", ("root://eosuser.cern.ch", "/eos/user/x")),
    ],
)
def test_split_destination(destination: str, expected: tuple[str, str]) -> None:
    assert split_destination(destination) == expected
    with pytest.raises(ValueError, match="Not a root:// URL"):
        split_destination("/eos/user/x")


class FakeXrdfs:
    """Answers xrdfs stat and query checksum from a dict of path to (size, adler32)."""

    def __init__(self, files: dict[str, tuple[int, str]]) -> None:
        self.files = files
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        operation, path = command[2], command[-1]
        if path not in self.files:
            return subprocess.CompletedProcess(
                command, 54, "", f"[ERROR] Server responded with an error: No such file: {path}"
            )
        size, adler32 = self.files[path]
        if operation == "stat":
            return subprocess.CompletedProcess(
                command, 0, f"Path:   {path}\nSize:   {size}\n", ""
            )
        return subprocess.CompletedProcess(command, 0, f"adler32 {adler32}\n", "")


def _records(*paths: str) -> list[dict[str, Any]]:
    return [dict(_record(path).metadata["writer_manifest"]) for path in paths]


def test_verify_delivery_accepts_complete_output() -> None:
    xrdfs = FakeXrdfs(
        {
            "/eos/user/x/y/skim/a/0_0.root": (7, "0a0b0c0d"),
            "/eos/user/x/y/skim/b/0_0.root": (7, "0a0b0c0d"),
        }
    )

    summary = verify_delivery(
        _records("a/0_0.root", "b/0_0.root"), DESTINATION, runner=xrdfs
    )

    assert summary == {
        "destination": DESTINATION,
        "expected": 2,
        "delivered": 2,
        "complete": True,
        "problems": [],
    }
    assert xrdfs.calls[0][:2] == ["xrdfs", "root://eosuser.cern.ch"]


def test_verify_delivery_lists_missing_and_mismatching_files() -> None:
    xrdfs = FakeXrdfs(
        {
            "/eos/user/x/y/skim/a/0_0.root": (7, "0a0b0c0d"),
            "/eos/user/x/y/skim/b/0_0.root": (9, "0a0b0c0d"),
            "/eos/user/x/y/skim/c/0_0.root": (7, "ffffffff"),
        }
    )

    summary = verify_delivery(
        _records("a/0_0.root", "b/0_0.root", "c/0_0.root", "d/0_0.root"),
        DESTINATION,
        runner=xrdfs,
    )

    assert summary["delivered"] == 1
    assert summary["complete"] is False
    statuses = {item["path"]: item["status"] for item in summary["problems"]}
    assert statuses == {
        "b/0_0.root": "size_mismatch",
        "c/0_0.root": "checksum_mismatch",
        "d/0_0.root": "missing",
    }
    mismatch = next(i for i in summary["problems"] if i["path"] == "b/0_0.root")
    assert mismatch["expected"] == 7
    assert mismatch["found"] == 9


def test_verify_delivery_without_checksum_checks_size_only() -> None:
    record = dict(_record("a/0_0.root", adler32=None).metadata["writer_manifest"])
    xrdfs = FakeXrdfs({"/eos/user/x/y/skim/a/0_0.root": (7, "00000000")})

    summary = verify_delivery([record], DESTINATION, runner=xrdfs)

    assert summary["complete"] is True
    assert all(call[2] == "stat" for call in xrdfs.calls)


def test_verify_delivery_reports_xrdfs_that_cannot_run() -> None:
    def broken(command: list[str], **_: Any) -> Any:
        raise FileNotFoundError("xrdfs")

    summary = verify_delivery(_records("a/0_0.root"), DESTINATION, runner=broken)

    assert summary["complete"] is False
    assert summary["problems"][0]["status"] == "error"
    assert "FileNotFoundError" in summary["problems"][0]["message"]


def test_verify_delivery_of_nothing_is_not_complete() -> None:
    summary = verify_delivery([], DESTINATION, runner=FakeXrdfs({}))

    assert summary["expected"] == 0
    assert summary["complete"] is False


def _run_backend_with_fake_cluster(
    monkeypatch: pytest.MonkeyPatch,
    *,
    delivery: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from hepflow.backends.model import BackendResult  # noqa: PLC0415

    from fasthep_distributed._dask._common import DaskBackend  # noqa: PLC0415

    seen: dict[str, Any] = {}
    reference = _record("ds/0_0.root")

    def fake_graph(plan: Any, **kwargs: Any) -> list[Any]:
        seen["partition_ctx"] = kwargs["partition_ctx"]
        return []

    def fake_compute(
        tasks: list[Any], *, execution: dict[str, Any], **_: Any
    ) -> tuple[list[Any], str | None, dict[str, Any]]:
        seen["execution"] = execution
        return [{"value_store": {("write.Selected", "artifact"): reference}}], None, {}

    def fake_verify(records: list[dict[str, Any]], destination: str) -> dict[str, Any]:
        seen["records"] = records
        seen["destination"] = destination
        return delivery

    def fake_result(plan: Any, **_: Any) -> BackendResult:
        return BackendResult(
            backend="dask",
            strategy="htcondor",
            success=True,
            summary={"warnings": [], "backend": {}},
        )

    prefix = "fasthep_distributed._dask"
    monkeypatch.setattr(f"{prefix}._common.build_dask_graph", fake_graph)
    monkeypatch.setattr(f"{prefix}._common.verify_delivery", fake_verify)
    monkeypatch.setattr(f"{prefix}._common.build_dask_backend_result", fake_result)
    monkeypatch.setattr(f"{prefix}._htcondor.compute_with_htcondor", fake_compute)
    plan = _plan(datasets=("ds",))
    plan.execution = _execution(output_destination=DESTINATION)
    plan.context = {"outdir": "."}

    return DaskBackend().run(plan), seen


def test_backend_delivers_to_the_destination_and_verifies_the_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = {
        "destination": DESTINATION,
        "expected": 1,
        "delivered": 1,
        "complete": True,
        "problems": [],
    }

    result, seen = _run_backend_with_fake_cluster(monkeypatch, delivery=complete)

    assert seen["partition_ctx"] == {
        "output_files_root": ".",
        "output_destination": DESTINATION,
    }
    assert seen["execution"]["config"]["output_files"] == ["ds"]
    assert seen["destination"] == DESTINATION
    assert [item["path"] for item in seen["records"]] == ["ds/0_0.root"]
    assert result.success is True
    assert result.summary["delivery"] == complete
    assert result.summary["backend"]["delivery"] == complete


def test_backend_fails_the_run_when_a_file_did_not_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = {
        "destination": DESTINATION,
        "expected": 1,
        "delivered": 0,
        "complete": False,
        "problems": [{"path": "ds/0_0.root", "status": "missing", "message": "gone"}],
    }

    result, _ = _run_backend_with_fake_cluster(monkeypatch, delivery=incomplete)

    assert result.success is False
    assert result.summary["delivery"]["problems"][0]["status"] == "missing"
    [warning] = result.summary["warnings"]
    assert warning["kind"] == "output_delivery"
    assert warning["message"] == "ds/0_0.root: missing"
    assert warning["detail"] == {"message": "gone"}
