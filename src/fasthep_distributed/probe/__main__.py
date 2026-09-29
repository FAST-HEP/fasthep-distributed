"""Command line: ``python -m fasthep_distributed.probe {prepare,submit,collect}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._config import ProbeConfigError, load_probe_config
from ._report import collect_probe, format_collection
from ._submit import ProbeBundle, ProbeError, prepare_probe, submit_probe
from ._worker import PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fasthep_distributed.probe",
        description="Standalone HTCondor worker probe (no Dask).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare", "write a probe directory without submitting"),
        ("submit", "prepare a probe directory and submit it with condor_submit"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
    collect = commands.add_parser("collect", help="summarise a submitted probe")
    collect.add_argument("probe_dir", type=Path)
    collect.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "collect":
            collection = collect_probe(args.probe_dir)
            if args.json:
                sys.stdout.write(json.dumps(collection.to_dict(), indent=2) + "\n")
            else:
                sys.stdout.write(format_collection(collection))
            return 0 if collection.verdict == PASS else 1

        bundle = prepare_probe(load_probe_config(args.config))
        _write_prepared(bundle)
        if args.command == "submit":
            result = submit_probe(bundle)
            sys.stdout.write(
                f"submitted {result.count} job(s) to cluster {result.cluster}\n"
                f"collect with: python -m fasthep_distributed.probe collect "
                f"{result.probe_dir}\n"
            )
    except (ProbeConfigError, ProbeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return 0


def _write_prepared(bundle: ProbeBundle) -> None:
    summary = bundle.submit_facts.get("summary", {})
    sys.stdout.write(f"prepared probe {bundle.probe_id}\n  {bundle.probe_dir}\n")
    for name in summary.get("warnings", []):
        sys.stdout.write(f"  submit-host warning: {name}\n")


if __name__ == "__main__":
    sys.exit(main())
