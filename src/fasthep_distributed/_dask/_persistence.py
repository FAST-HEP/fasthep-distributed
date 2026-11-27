"""Product persistence on the Dask backend (fasthep-flow ``persistence.md``).

Workers save the boundary products of their partition with Flow's
``persist_boundary_products``; the client reuses published sets before it submits tasks.
The persistence root must be on a filesystem the workers and the client share: batch
strategies need an absolute ``root``, remote roots are refused, and delivery of persisted
sets to EOS is not part of this stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from hepflow.compiler.input_paths import is_uri
from hepflow.model.persistence import PersistenceSettings
from hepflow.model.plan import ExecutionPartition, ExecutionPlan
from hepflow.registry.loaders import runtime_registry_from_config
from hepflow.runtime.boundary import plan_partition_boundary
from hepflow.runtime.persistence.evidence import EvidenceLedger
from hepflow.runtime.persistence.reuse import ReuseReport, reuse_partition_products
from hepflow.runtime.persistence.save import persistence_root, provisional_plan_hash
from hepflow.runtime.persistence.select import (
    PersistenceSelection,
    global_side_product_nodes,
    resolve_persistence,
)
from hepflow.runtime.provenance import ProvenanceRecorder

from fasthep_distributed._dask._worker_env import _NODE_LOCAL_ROOTS

_BATCH_STRATEGIES = {"htcondor", "slurm"}


def persistence_settings(execution: Mapping[str, Any] | None) -> PersistenceSettings:
    """``execution.persistence`` as typed settings (``off`` when absent)."""
    return PersistenceSettings.from_execution(execution)


def validate_persistence(execution: Mapping[str, Any], strategy: str) -> None:
    """Refuse a persistence setting the backend cannot honour, before any cluster starts."""
    settings = persistence_settings(execution)
    if settings.mode == "off":
        return
    persistence_root(settings, ".")  # remote URIs raise NotImplementedError here
    if strategy not in _BATCH_STRATEGIES:
        return
    root = settings.root
    if root is None:
        raise ValueError(
            f"execution.persistence on the {strategy} strategy needs an absolute root "
            "on a filesystem shared with the client (execution.persistence.root)"
        )
    parsed = urlparse(root) if is_uri(root) else None
    path = unquote(parsed.path) if parsed else root
    if not Path(path).is_absolute() or (
        parsed is not None and parsed.netloc not in {"", "localhost"}
    ):
        raise ValueError(
            f"execution.persistence.root {root!r} must be an absolute path (or "
            f"file:///absolute/path) on the {strategy} strategy"
        )
    resolved = Path(path).resolve()
    for local in _NODE_LOCAL_ROOTS:
        if resolved == local or local in resolved.parents:
            raise ValueError(
                f"execution.persistence.root {root!r} is under {local}, which is "
                f"node-local on the {strategy} strategy: workers would each publish "
                "to their own copy. Use a path on a shared filesystem."
            )


class ClientPersistence:
    """Client-side view of persistence for one run: selection, reuse and the summary."""

    def __init__(self, plan: ExecutionPlan, *, outdir: str) -> None:
        self.plan = plan
        self.outdir = outdir
        self.settings = persistence_settings(plan.execution)
        self.registry = runtime_registry_from_config(plan.registry)
        self.boundary = plan_partition_boundary(plan, runtime_registry=self.registry)
        self.selection: PersistenceSelection = resolve_persistence(
            self.boundary,
            self.registry,
            self.settings,
            global_nodes=global_side_product_nodes(plan),
        )
        self.ledger = EvidenceLedger()
        self.reuse_report = ReuseReport()
        # once, before any partition runs; workers stamp this, they do not recompute it
        self.plan_hash = provisional_plan_hash(plan)

    @property
    def enabled(self) -> bool:
        return self.settings.mode != "off"

    def reuse(self, partition: ExecutionPartition) -> dict[str, Any] | None:
        """A task result for a partition whose products are published, else ``None``."""
        scratch = ProvenanceRecorder()
        products = reuse_partition_products(
            self.plan,
            partition,
            boundary=self.boundary,
            selection=self.selection,
            settings=self.settings,
            registry=self.registry,
            outdir=self.outdir,
            plan_hash=self.plan_hash,
            report=self.reuse_report,
            ledger=self.ledger,
            store=scratch.store,
        )
        if products is None:
            return None
        from hepflow.runtime.boundary import boundary_products_to_value_store  # noqa: PLC0415

        return {
            "value_store": boundary_products_to_value_store(products),
            "warnings": [],
            "hooks": {"enabled": []},
            "provenance_operations": scratch.operation_records(),
            "provenance_resources": scratch.serialise_resources(),
        }

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = self.selection.to_dict()
        if self.ledger.consumptions:
            summary["consumed"] = list(self.ledger.consumptions)
        if self.settings.reuse:
            summary["reuse"] = self.reuse_report.to_dict()
        return summary
