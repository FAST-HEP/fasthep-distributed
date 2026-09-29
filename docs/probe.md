# HTCondor worker probe

The worker probe submits a plain HTCondor job, without Dask, that reports
whether a worker node can use a shared Python environment and read the paths
a production needs. Run it before starting Dask workers on a new site: if the
probe passes and a Dask cluster then fails, the problem is the network or
Dask, not storage or the environment.

The probe is a development tool. Site-specific paths belong in the probe
configuration, not in this package.

## Commands

```bash
python -m fasthep_distributed.probe prepare --config probe.yaml
python -m fasthep_distributed.probe submit  --config probe.yaml
python -m fasthep_distributed.probe collect <probe_dir> [--json]
```

- `prepare` writes a new probe directory and runs the same checks on the submit
  host, without contacting the schedd. It fails early if a required check
  already fails there.
- `submit` runs `prepare` and then `condor_submit`.
- `collect` reads the Condor event log and the job reports, prints a summary
  and exits non-zero unless every job passed.

## Configuration

```yaml
prefix: /shared/path/.pixi/envs/dev    # environment used on workers
probe_root: /shared/path/build/probes  # submit files and logs go here
lockfile: /shared/path/pixi.lock       # optional; recorded by sha256
imports:
  required: [hepflow, fasthep_distributed, distributed]
  optional: [fasthep]
paths:
  - {name: work, path: /shared/path}
  - {name: data, path: /data/user, required: false}
condor:
  flavour: espresso        # +JobFlavour; set to null if the site has none
  request_cpus: 1
  request_memory: 2GB
  request_disk: null
  count: 1
  extra_directives: {}
submit_dir:
  forbidden_prefixes: [/eos]  # default
  required_prefix: null
timeouts:                     # seconds
  imports: 300
  paths: 30
  git: 60
  scratch: 60
```

Relative paths are resolved against the directory containing the
configuration file.

## What the job checks

| Check | Meaning |
|---|---|
| `python.interpreter` | the configured interpreter runs and `sys.prefix` matches `prefix` |
| `python.imports` | required and optional imports, versions and source files |
| `python.editables` | editable distributions with git revision, dirty state and a diff hash |
| `paths.<name>` | `stat` and a directory listing, or a short file read |
| `scratch` | worker scratch location, free space and a small write/read test |

Every filesystem check has a timeout, so a hung network mount is reported
instead of stalling the job. Paths with `required: false` produce warnings
rather than failures.

The worker compares its editable source revisions with those recorded on the
submit host and lists differences as `source_drift`.

The probe does not yet test credentials or XRootD access.

## Probe directory

```text
<probe_root>/<UTC timestamp>-<git short sha>/
  config.json  _worker.py  submit_facts.json   # transferred to the job
  probe.sh  probe.sub                          # executable and submit file
  submit.json                                  # written by submit
  condor.log  stdout.<proc>.json  stderr.<proc>.log
```

`_worker.py` uses only the standard library and is run by path with the
interpreter under test, so importing FAST-HEP is a check rather than a
precondition. If the interpreter cannot start, `probe.sh` still writes a
minimal failure report.

Each job writes one JSON report to stdout with schema
`fasthep.distributed.probe/v1`. The report contains `host` facts, the list of
`checks`, the `submit` facts and a `summary` with `verdict`, `failed_required`,
`warnings` and `source_drift`. Environment variables are recorded only from a
short allowlist.
