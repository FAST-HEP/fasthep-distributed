"""A toy histogram kind with a JSON persistence capability, for the persistence tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hepflow.model.persistence import SavedPayload, SaveOutcome


def save_toy(
    value: Any, *, root: Path, params: dict[str, Any], identity: dict[str, Any]
) -> SaveOutcome:
    root.mkdir(parents=True, exist_ok=True)
    (root / "counts.json").write_text(json.dumps(value), encoding="utf-8")
    return SaveOutcome(payloads=(SavedPayload("counts.json", role="data"),))


def load_toy(
    root: Path,
    payloads: Sequence[Path],
    *,
    params: dict[str, Any],
    variant: str | None,
    locator: str | None,
) -> Any:
    return json.loads(Path(payloads[0]).read_text(encoding="utf-8"))


def combine_counts(
    left: dict[str, Any], right: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    return {"entries": left["entries"] + right["entries"]}


def merge_counts(values: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    total = {"entries": 0}
    for value in values:
        total = combine_counts(total, value)
    return total


def save_then_crash(
    value: Any, *, root: Path, params: dict[str, Any], identity: dict[str, Any]
) -> SaveOutcome:
    save_toy(value, root=root, params=params, identity=identity)
    raise RuntimeError("killed")
