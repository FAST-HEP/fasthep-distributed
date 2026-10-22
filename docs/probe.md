# HTCondor worker probe

The worker probe reports whether a worker node can use a shared Python
environment, read the paths a production needs, read remote data over XRootD
and publish files to EOS. It has two modes. The plain mode submits an HTCondor
job without Dask; run it before starting Dask workers on a new site: if the
probe passes and a Dask cluster then fails, the problem is the network or
Dask, not storage or the environment. The Dask mode starts workers through the
production HTCondor path and runs the same checks as Dask tasks.

The probe is a development tool. Site-specific paths belong in the probe
configuration, not in this package.

## Commands

```bash
python -m fasthep_distributed.probe prepare --config probe.yaml
python -m fasthep_distributed.probe submit  --config probe.yaml
python -m fasthep_distributed.probe collect <probe_dir> [--json]
python -m fasthep_distributed.probe dask    --config dask-probe.yaml
```

- `prepare` writes a new probe directory and runs the same checks on the submit
  host, without contacting the schedd. It fails early if a required check
  already fails there.
- `submit` runs `prepare` and then `condor_submit`.
- `collect` reads the Condor event log and the job reports, prints a summary
  and exits non-zero unless every job passed.
- `dask` starts the configured number of HTCondor Dask workers, runs the worker
  checks once on each, shuts the workers down and prints a report (see
  the Dask workers section below). It needs a `dask:` section in the
  configuration.

## Configuration

```yaml
prefix: /shared/path/.pixi/envs/dev    # environment used on workers
probe_root: /shared/path/build/probes  # submit files and logs go here
lockfile: /shared/path/pixi.lock       # optional; recorded by sha256
imports:
  required: [hepflow, fasthep_distributed, distributed]
  optional: [uproot]
paths:
  - {name: work, path: /shared/path}
  - {name: data, path: /data/user, required: false}
xrootd:                                # optional; remote read test
  - name: nanoaod_test_file
    url: root://cms-xrd-global.cern.ch//store/mc/.../file.root
    expect_size: 3912802               # optional, bytes
eos_write:                             # optional; publish test
  - name: eos_user_testing
    url: root://eosuser.cern.ch//eos/user/u/user/testing/probe
    size_bytes: 1048576                # default 1 MiB
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
  xrootd: 300
  eos_write: 300
  output_files: 60
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
| `xrootd.<name>` | copy a file with `xrdcp` into scratch and compare its size; records the X509 proxy path and state |
| `eos_write.<name>` | publish a file with `xrdcp` to a temporary name, `xrdfs mv` it, check the size and remove it; records the available credentials |
| `output_files.<name>` | worker only: write partition-like files in the job directory (see Output transfer below) |
| `scratch` | worker scratch location, free space and a small write/read test |

Every filesystem check has a timeout, so a hung network mount is reported
instead of stalling the job. Paths with `required: false` produce warnings
rather than failures.

The worker compares its editable source revisions with those recorded on the
submit host and lists differences as `source_drift`.

Credentials are never read or logged. The `xrootd` and `eos_write` checks record
only whether `X509_USER_PROXY` is set, the path it points to, whether the file
exists and its mode. The `eos_write` check also records whether `KRB5CCNAME` is
set and which entries the HTCondor credentials directory holds, by name. Both
checks also run on the submit host before anything is submitted, and a required check
that fails there stops the probe, so a missing proxy is reported at once. A
check that needs a proxy therefore needs `X509_USER_PROXY` set in the
submitting shell.

The checks cannot say which credential EOS accepted when both a proxy and a
Kerberos ticket are available.

## Probe directory

```text
<probe_root>/<UTC timestamp>-<git short sha>/
  config.json  _worker.py  submit_facts.json   # transferred to the job
  probe.sh  probe.sub                          # executable and submit file
  submit.json                                  # written by submit
  condor.log  stdout.<proc>.json  stderr.<proc>.log
  execution.json  dask-report.json  execution/ # written by the dask mode
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

## Dask workers

The `dask` mode uses a `dask:` section. `use` and `execution` have the meaning
they have in a workflow and are resolved by Flow, so site profiles such as
`fasthep_distributed:dask_htcondor_cern` can be used; the probe goes through the
same start-up path as plan execution.

```yaml
dask:
  retire_grace: 120      # seconds to wait for the worker jobs to exit
  task_timeout: 900      # seconds for the checks on each worker
  kill_workers: 0        # remove this many workers with condor_rm first
  use:
    profiles: [fasthep_distributed:dask_htcondor_cern]
  execution:
    backend: dask
    strategy: htcondor
    environment: {type: shared-prefix, prefix: /shared/path/.pixi/envs/dev}
    staging: {mode: shared}
    resources: {default: {cpus: 1, memory: 2GB, disk: 1GB}}
    config:
      workers: 2
      queue: espresso
```

The probe waits for the workers, runs the worker checks on each worker as a
Dask task, releases the results, then shuts the workers down with the same
function the backend uses (`shutdown_htcondor_workers`): workers are retired
and the jobs given `retire_grace` seconds to exit on their own. The report
records the port each worker uses against the configured range, the final state
of every Condor job (including whether its output transfer finished), whether
the worker log came back, the timings, and the shutdown outcome. It is written
to `dask-report.json` in a new probe directory, also when start-up fails.

`kill_workers` removes the first N workers (by job id) with `condor_rm` after
their checks and before the others are retired, to see what a worker that does
not exit cleanly leaves behind. A removed job returns no log, which the report
shows as a warning.

### Output transfer

With `execution.config.output_destination` and `output_files` set (see the
HTCondor strategy in `AGENTS.md`), HTCondor sends the listed files from the job
directory to EOS when the job exits. The `output_files` check supports testing
this: each worker writes `<subdir>/<host>-<token>_0.root`, optionally
`<subdir>/<shared_name>` with different content in every job, and a top-level
decoy `flat-<token>.bin` that a `transfer_output_files` list naming only
`<subdir>` should not send. Names, sizes and sha256 values are recorded in the
worker report for comparison with what arrives on EOS.

```yaml
output_files:
  - {name: partition_like, subdir: my_dataset, size_bytes: 65536, shared_name: same_0.root}
```

Names never contain the process id: process ids are not unique across jobs.
