from __future__ import annotations

from fasthep_distributed._dask._common import _merge_hook_summaries


def _item(partition_id: str, calls: int = 2) -> dict:
    return {
        "kind": "hep.resource_usage",
        "events": ["partition_start", "partition_end"],
        "source": "execution_hook",
        "calls": calls,
        "results": {"resource_usage": [{"partition_id": partition_id}]},
    }


def test_merge_hook_summaries_concatenates_results_across_partitions() -> None:
    merged = _merge_hook_summaries(
        [
            {"enabled": [_item("p0")]},
            {"enabled": [_item("p1")]},
            {"enabled": [{"kind": "hep.warning_capture", "events": [], "calls": 1}]},
        ]
    )

    usage, warnings = merged["enabled"]
    assert usage["calls"] == 4
    assert [r["partition_id"] for r in usage["results"]["resource_usage"]] == [
        "p0",
        "p1",
    ]
    assert "results" not in warnings
