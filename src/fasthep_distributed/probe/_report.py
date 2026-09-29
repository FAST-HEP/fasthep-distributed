from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._submit import CONDOR_LOG_FILE, SUBMIT_RECORD_FILE
from ._worker import FAIL, PASS

_EVENT_PATTERN = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\)")
_RETURN_VALUE_PATTERN = re.compile(r"return value (-?\d+)")
_SIGNAL_PATTERN = re.compile(r"signal (\d+)")

_EVENT_STATES = {
    "000": "submitted",
    "001": "running",
    "004": "evicted",
    "005": "terminated",
    "009": "aborted",
    "012": "held",
    "013": "released",
}


@dataclass(slots=True)
class CondorJobState:
    state: str = "unknown"
    return_value: int | None = None
    detail: str | None = None


@dataclass(slots=True)
class ProbeJobResult:
    process: int
    condor: CondorJobState
    report: dict[str, Any] | None = None
    report_error: str | None = None

    @property
    def verdict(self) -> str:
        if self.report is None:
            return "pending" if self.condor.state in _ACTIVE_STATES else FAIL
        return str(self.report.get("summary", {}).get("verdict", FAIL))


@dataclass(slots=True)
class ProbeCollection:
    probe_dir: Path
    probe_id: str | None
    cluster: int | None
    submitted: bool
    jobs: list[ProbeJobResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.submitted or not self.jobs:
            return FAIL
        verdicts = {job.verdict for job in self.jobs}
        if verdicts == {PASS}:
            return PASS
        if "pending" in verdicts and FAIL not in verdicts:
            return "pending"
        return FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_dir": str(self.probe_dir),
            "probe_id": self.probe_id,
            "cluster": self.cluster,
            "submitted": self.submitted,
            "verdict": self.verdict,
            "jobs": [
                {
                    "process": job.process,
                    "condor": {
                        "state": job.condor.state,
                        "return_value": job.condor.return_value,
                        "detail": job.condor.detail,
                    },
                    "verdict": job.verdict,
                    "report_error": job.report_error,
                    "report": job.report,
                }
                for job in self.jobs
            ],
        }


_ACTIVE_STATES = {"unknown", "submitted", "running", "evicted", "released"}


def collect_probe(probe_dir: Path) -> ProbeCollection:
    probe_dir = Path(probe_dir)
    record_path = probe_dir / SUBMIT_RECORD_FILE
    if not record_path.exists():
        return ProbeCollection(
            probe_dir=probe_dir,
            probe_id=_probe_id_from_config(probe_dir),
            cluster=None,
            submitted=False,
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cluster = int(record["cluster"])
    states = parse_condor_log(probe_dir / CONDOR_LOG_FILE, cluster=cluster)
    jobs = []
    for process in range(int(record.get("count", 1))):
        report, error = _read_report(probe_dir / f"stdout.{process}.json")
        jobs.append(
            ProbeJobResult(
                process=process,
                condor=states.get(process, CondorJobState()),
                report=report,
                report_error=error,
            )
        )
    return ProbeCollection(
        probe_dir=probe_dir,
        probe_id=record.get("probe_id"),
        cluster=cluster,
        submitted=True,
        jobs=jobs,
    )


def parse_condor_log(path: Path, *, cluster: int) -> dict[int, CondorJobState]:
    """Return the latest Condor state for each process of ``cluster``."""

    if not path.exists():
        return {}
    states: dict[int, CondorJobState] = {}
    current: CondorJobState | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EVENT_PATTERN.match(line)
        if match is not None:
            current = None
            code, event_cluster, process = match.groups()
            if int(event_cluster) != cluster or code not in _EVENT_STATES:
                continue
            current = CondorJobState(state=_EVENT_STATES[code])
            states[int(process)] = current
            continue
        if current is None or line.strip() in {"", "..."}:
            continue
        text = line.strip()
        if current.state == "terminated" and current.detail is None:
            current.detail = text
            value = _RETURN_VALUE_PATTERN.search(text)
            if value is not None:
                current.return_value = int(value.group(1))
            elif _SIGNAL_PATTERN.search(text):
                current.return_value = None
        elif current.state in {"held", "aborted"} and current.detail is None:
            current.detail = text
    return states


def format_collection(collection: ProbeCollection) -> str:
    lines = [f"probe {collection.probe_id or '?'}  {collection.probe_dir}"]
    if not collection.submitted:
        lines.append("  not submitted (no submit.json); run 'submit' first")
        return "\n".join(lines) + "\n"
    lines[0] += f"  cluster {collection.cluster}"
    for job in collection.jobs:
        condor = job.condor.state
        if job.condor.return_value is not None:
            condor += f" (return value {job.condor.return_value})"
        elif job.condor.detail and job.condor.state != "terminated":
            condor += f" ({job.condor.detail})"
        host = (job.report or {}).get("host", {}).get("hostname", "?")
        lines.append(
            f"job {collection.cluster}.{job.process}  condor: {condor}  "
            f"host: {host}  verdict: {job.verdict}"
        )
        if job.report is None:
            if job.report_error:
                lines.append(f"  report: {job.report_error}")
            continue
        for check in job.report.get("checks", []):
            message = check.get("error") or _detail_message(check.get("detail") or {})
            suffix = f"  {message}" if message else ""
            lines.append(f"  {check.get('status', '?'):<5} {check.get('name')}{suffix}")
        drift = job.report.get("summary", {}).get("source_drift", [])
        if drift:
            for item in drift:
                lines.append(f"  drift {item.get('name')}: {item.get('reason')}")
        else:
            lines.append("  source drift: none")
    lines.append(f"verdict: {collection.verdict}")
    return "\n".join(lines) + "\n"


def _detail_message(detail: dict[str, Any]) -> str | None:
    warning = detail.get("warning")
    return str(warning) if warning else None


def _read_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"{path.name} not found"
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None, f"{path.name} is empty"
    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(report, dict):
        return None, f"{path.name} does not contain a JSON object"
    return report, None


def _probe_id_from_config(probe_dir: Path) -> str | None:
    config = probe_dir / "config.json"
    if not config.exists():
        return None
    try:
        return json.loads(config.read_text(encoding="utf-8")).get("probe_id")
    except json.JSONDecodeError:
        return None
