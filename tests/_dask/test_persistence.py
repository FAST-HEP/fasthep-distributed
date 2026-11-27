from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hepflow.model.plan import ExecutionNode, ExecutionPartition, ExecutionPlan

from fasthep_distributed._dask import _common
from fasthep_distributed._dask._common import DaskBackend, _execute_partition_task
from fasthep_distributed._dask._persistence import (
    persistence_settings,
    validate_persistence,
)

REGISTRY: dict[str, Any] = {
    "product_handlers": {
        "histogram": {
            "boundary": {"retain": True, "representation": "value"},
            "combine": "toy_persistence:combine_counts",
            "merge": "toy_persistence:merge_counts",
            "persistence": {
                "formats": {
                    "toy.counts": {
                        "version": "1",
                        "container": "json",
                        "payload_roles": ["data"],
                        "fidelity": "exact",
                        "save": "toy_persistence:save_toy",
                        "load": "toy_persistence:load_toy",
                    }
                }
            },
        }
    }
}


# -- settings and validation (Task 4) -------------------------------------------


def _execution(strategy: str = "local", **persistence: Any) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "backend": "dask",
        "strategy": strategy,
        "config": {"scheduler": "synchronous"},
    }
    if persistence:
        execution["persistence"] = persistence
    return execution


@pytest.mark.parametrize("strategy", ["local", "htcondor", "slurm"])
def test_off_is_always_valid(strategy: str) -> None:
    validate_persistence(_execution(strategy), strategy)
    assert persistence_settings(_execution(strategy)).mode == "off"


def test_local_default_root_is_valid() -> None:
    validate_persistence(_execution(mode="all"), "local")


@pytest.mark.parametrize("strategy", ["htcondor", "slurm"])
def test_batch_strategy_without_a_root_raises_before_submit(strategy: str) -> None:
    with pytest.raises(ValueError, match="absolute root"):
        validate_persistence(_execution(strategy, mode="all"), strategy)


@pytest.mark.parametrize("root", ["relative/dir", "file://host/x"])
def test_batch_strategy_with_a_job_relative_root_raises(root: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        validate_persistence(_execution("htcondor", mode="all", root=root), "htcondor")


def test_htcondor_with_an_absolute_root_is_valid() -> None:
    validate_persistence(
        _execution("htcondor", mode="all", root="/shared/products"), "htcondor"
    )


def test_remote_root_raises() -> None:
    with pytest.raises(NotImplementedError, match=r"root://"):
        validate_persistence(
            _execution(mode="all", root="root://eos.example//p"), "local"
        )


# -- the Dask path (Task 5) -------------------------------------------------------


def _plan(tmp_path: Path, **persistence: Any) -> ExecutionPlan:
    plan = ExecutionPlan(
        execution=_execution("local", **persistence) if persistence else _execution()
    )
    plan.registry = REGISTRY
    plan.context = {"outdir": str(tmp_path)}
    plan.add_node(
        ExecutionNode(
            id="stage.Hist",
            graph_node_id="stage.Hist",
            role="transform",
            impl="toy.hist",
            outputs={"hist": "histogram"},
            input_scope="partition",
            output_scope="partition",
        )
    )
    plan.partitions = [
        ExecutionPartition(
            id=f"events__toy__{index}",
            dataset="toy",
            file="toy://events",
            source="events",
            part=f"0_{index}",
            start=index,
            stop=index + 1,
        )
        for index in range(2)
    ]
    return plan


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the partition work with a counter; returns the ids it ran."""
    ran: list[str] = []

    def fake(plan: Any, *, ctx: dict[str, Any], **_: Any) -> dict[tuple[str, str], Any]:
        partition_id = ctx["partition"]["id"]
        ran.append(partition_id)
        return {("stage.Hist", "hist"): {"entries": int(partition_id[-1]) + 1}}

    monkeypatch.setattr(_common, "execute_plan_partition", fake)
    return ran


def _partition_values(result: Any) -> list[Any]:
    return [
        store[("stage.Hist", "hist")]
        for store in result.outputs["partition_value_stores"]
    ]


def test_partition_task_persists_selected_products(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all")

    result = _execute_partition_task(
        plan,
        plan.partitions[1],
        base_ctx={"outdir": str(tmp_path)},
        registry_cfg=REGISTRY,
    )

    assert result["value_store"][("stage.Hist", "hist")] == {"entries": 2}
    manifests = list((tmp_path / "artifacts" / "products").rglob("manifest.json"))
    assert len(manifests) == 1
    assert executed == ["events__toy__1"]


def test_partition_task_stamps_the_hash_the_client_computed(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all")

    _execute_partition_task(
        plan,
        plan.partitions[0],
        base_ctx={"outdir": str(tmp_path)},
        registry_cfg=REGISTRY,
        plan_hash="client-hash",
    )

    manifest = next((tmp_path / "artifacts" / "products").rglob("manifest.json"))
    assert json.loads(manifest.read_text())["plan_hash"] == "client-hash"


def test_mode_off_result_is_unchanged(tmp_path: Path, executed: list[str]) -> None:
    plan = _plan(tmp_path)

    result = DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})

    assert "persistence" not in result.summary
    assert not (tmp_path / "artifacts" / "products").exists()
    assert _partition_values(result) == [{"entries": 1}, {"entries": 2}]


def test_reusable_partitions_are_not_submitted(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all", reuse=True)
    DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})
    assert len(executed) == 2
    executed.clear()

    result = DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})

    assert executed == []
    assert _partition_values(result) == [{"entries": 1}, {"entries": 2}]


def test_merged_partition_order_is_preserved_with_partial_reuse(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all", reuse=True)
    DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})
    first = next(
        (tmp_path / "artifacts" / "products").rglob("events__toy__0/manifest.json")
    )
    first.unlink()
    executed.clear()

    result = DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})

    assert executed == ["events__toy__0"]
    assert _partition_values(result) == [{"entries": 1}, {"entries": 2}]


def test_summary_lists_persisted_and_consumed(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all", reuse=True)
    first = DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})
    second = DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})

    assert [p["capability"] for p in first.summary["persistence"]["persisted"]] == [
        "toy.counts"
    ]
    assert first.summary["persistence"].get("consumed") is None
    assert len(second.summary["persistence"]["consumed"]) == 2


def test_unfinished_set_has_no_manifest(
    tmp_path: Path, executed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = {
        "product_handlers": {
            "histogram": {
                **REGISTRY["product_handlers"]["histogram"],
                "persistence": {
                    "formats": {
                        "toy.counts": {
                            **REGISTRY["product_handlers"]["histogram"]["persistence"][
                                "formats"
                            ]["toy.counts"],
                            "save": "toy_persistence:save_then_crash",
                        }
                    }
                },
            }
        }
    }
    plan = _plan(tmp_path, mode="all")
    plan.registry = registry

    with pytest.raises(RuntimeError, match="killed"):
        _execute_partition_task(
            plan,
            plan.partitions[0],
            base_ctx={"outdir": str(tmp_path)},
            registry_cfg=registry,
        )

    assert not list((tmp_path / "artifacts" / "products").rglob("manifest.json"))


# -- review fixes ----------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["htcondor", "slurm"])
@pytest.mark.parametrize("root", ["/tmp/products", "/var/tmp/p", "/dev/shm/p"])
def test_a_node_local_root_is_refused_on_batch_strategies(
    strategy: str, root: str
) -> None:
    with pytest.raises(ValueError, match="node-local"):
        validate_persistence(_execution(strategy, mode="all", root=root), strategy)


def test_file_localhost_is_a_valid_absolute_root() -> None:
    validate_persistence(
        _execution("htcondor", mode="all", root="file://localhost/shared/p"), "htcondor"
    )


def test_mode_off_does_not_build_the_persistence_machinery(
    tmp_path: Path, executed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("persistence off must not build ClientPersistence")

    monkeypatch.setattr(_common, "ClientPersistence", boom)

    DaskBackend().run(_plan(tmp_path), ctx={"outdir": str(tmp_path)})


def test_a_bad_product_key_fails_before_any_work(
    tmp_path: Path, executed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("global side products ran before the setting was checked")

    monkeypatch.setattr(_common, "execute_global_side_product_nodes", boom)
    plan = _plan(tmp_path, mode="selected", products={"stage.Typo:hist": True})

    with pytest.raises(ValueError, match=r"stage\.Typo:hist"):
        DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})


class _Reporter:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.run_id = "r"

        class _Counts:
            def to_dict(self) -> dict[str, int]:
                return {}

        self.counts = _Counts()

    def run_started(self, **_: Any) -> None:
        self.events.append("run_started")

    def run_failed(self, exc: BaseException) -> None:
        self.events.append("run_failed")

    def phase_started(self, *_: Any, **__: Any) -> None:
        self.events.append("phase_started")

    def phase_completed(self, *_: Any, **__: Any) -> None:
        self.events.append("phase_completed")

    def completed(self, *_: Any, **__: Any) -> None:
        self.events.append("completed")

    def close(self) -> list[Any]:
        self.events.append("close")
        return []


def test_a_failure_while_reusing_is_reported_and_closes_the_reporter(
    tmp_path: Path, executed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: Any, partition: Any) -> None:
        raise RuntimeError("reuse failed")

    monkeypatch.setattr(_common.ClientPersistence, "reuse", explode)
    reporter = _Reporter()

    with pytest.raises(RuntimeError, match="reuse failed"):
        DaskBackend().run(
            _plan(tmp_path, mode="all", reuse=True),
            ctx={"outdir": str(tmp_path)},
            progress=reporter,  # type: ignore[arg-type]
        )

    assert reporter.events[0] == "run_started"
    assert "run_failed" in reporter.events
    assert reporter.events[-1] == "close"


def test_a_compute_function_returning_the_wrong_count_is_a_clear_error(
    tmp_path: Path, executed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, mode="all", reuse=True)
    DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})
    first = next(
        (tmp_path / "artifacts" / "products").rglob("events__toy__0/manifest.json")
    )
    first.unlink()  # one partition left to run
    monkeypatch.setattr(
        "fasthep_distributed._dask._local.compute_with_local_strategy",
        lambda tasks, **_: ([], None),
    )

    with pytest.raises(RuntimeError, match="returned 0 results for 1"):
        DaskBackend().run(plan, ctx={"outdir": str(tmp_path)})


def test_global_side_products_are_not_persisted_per_partition(
    tmp_path: Path, executed: list[str]
) -> None:
    plan = _plan(tmp_path, mode="all")
    plan.nodes[0].input_scope = "global"
    plan.nodes[0].output_scope = "global"

    result = _execute_partition_task(
        plan,
        plan.partitions[0],
        base_ctx={"outdir": str(tmp_path)},
        registry_cfg=REGISTRY,
    )

    assert result["value_store"][("stage.Hist", "hist")] == {"entries": 1}
    assert not (tmp_path / "artifacts" / "products").exists()
    summary = _common.ClientPersistence(plan, outdir=str(tmp_path)).summary()
    assert summary["not_persisted"][0]["reason"] == "global side product"
