# fasthep-distributed Agent Instructions

These instructions apply when working in the standalone `fasthep-distributed`
repository. The Python package is `fasthep_distributed`.

## Ownership

`fasthep-distributed` owns reusable distributed execution implementations for
FAST-HEP:

- Dask backend implementation for Flow execution plans;
- local Dask execution strategy and Dask `LocalCluster` configuration;
- HTCondor worker submission through `dask-jobqueue`;
- pooled Dask worker templates for heterogeneous resources;
- worker environment packing, staging and transfer preparation;
- distributed resource mapping, scheduler integration and backend registry
  entries.

It does not own backend-neutral workflow semantics:

- `fasthep-flow` owns workflow YAML, normalisation, graph/data-flow analysis,
  execution plan models, runtime orchestration and backend interfaces.
  This package implements those backend contracts; it should not redefine them.
- HEP analysis operations, ROOT/awkward IO, histogram filling and cutflows
  belong in `fasthep-carpenter`.
- Metadata inspection, diagnostics and package-level provenance features belong
  in `fasthep-curator`.
- Plots, reports and visual outputs belong in `fasthep-render`.
- Workshop demonstrations belong in `fasthep-workshop`; cross-package smoke and
  release checks belong in `fasthep-dev`.

## Backend Contracts

- `src/fasthep_distributed/_dask/_common.py` provides `DaskBackend`, builds
  partition-granular Dask graphs from `hepflow.model.plan.ExecutionPlan`, and
  returns `hepflow.backends.model.BackendResult`.
- `src/fasthep_distributed/_dask/_spec.py` exposes `DASK_BACKEND_SPEC` for Flow
  registry validation and backend build directories.
- Keep Flow imports at the public contract level where possible:
  `hepflow.backends.model`, `hepflow.model.*`, `hepflow.runtime.*` and
  `hepflow.build_layout`.
- Do not add workflow-language, compiler or runtime orchestration behavior here
  when it belongs in `fasthep-flow`.

## Dask Strategies

- `strategy: local` and `strategy: default` use local Dask execution. Plain
  Dask schedulers support `threads`, `processes` and `synchronous`; local
  cluster mode uses `distributed.LocalCluster`.
- `strategy: htcondor` uses `dask-jobqueue` HTCondOR workers, pooled worker
  specs, worker start timeout handling, scheduler dashboard propagation and
  Flow build directories under `execution/dask/htcondor/`.
- Heterogeneous worker pools are supported for jobqueue strategies. Local Dask
  currently supports only a single default pool.
- Resource classes in Flow execution config map to Dask worker resources such as
  `resource.<name>` and `GPU`; node `execution.require` metadata drives Dask
  task annotations.
- A SLURM module and unit tests exist, but do not document or promise SLURM as a
  supported user-facing strategy unless the package docs/profiles and site
  validation are updated with it.

## Worker Environments and Staging

- `src/fasthep_distributed/_dask/_worker_env.py` handles packed Pixi worker
  environments, editable-package snapshots, compile bundles, bootstrap scripts,
  staging manifests and X509 proxy transfer metadata.
- Packed Pixi prefix mode currently resolves `source='current'`, packs the
  active prefix with `conda-pack`, validates imports before submit, and records
  manifests under Flow build paths.
- Staging mode `shared` assumes workers can see shared paths. Staging mode
  `transfer` creates transfer inputs such as `compile.tar.gz`, `prefix.tar.gz`,
  `bootstrap.sh` and optional `editable-snapshot.tar.gz`.
- Never put credential contents in plans, manifests or logs. X509 proxy support
  should pass file paths to scheduler transfer and set worker environment
  variables in the job prologue.
- HTCondOR transfer file basenames must be unique. Preserve invariants such as
  `transfer_executable=False`, empty `transfer_output_files`, and
  Flow-controlled `Output`, `Error` and `Log` paths.
- Do not stream HTCondor stdout/stderr by default. dask-jobqueue adds
  `Stream_Output`/`Stream_Error` when `log_directory` is set; the FAST-HEP job
  class removes them unless requested in `job_extra_directives`. CERN schedds
  reject streaming since November 2025.
- Scheduler-specific options are site-sensitive. Validate queue/flavour,
  walltime, disk, memory, GPU and log-directory behavior against the scheduler
  adapter rather than assuming another scheduler's vocabulary.

## Registry and Profiles

- Built-in registry: `src/fasthep_distributed/profiles/registry.yaml`
- Local Dask profile: `src/fasthep_distributed/profiles/dask_local.yaml`
- The registry contributes backend `dask` with both spec and implementation.
  Preserve this extension mechanism instead of hard-coding registration in
  Flow.

## Important Locations

- Public package: `src/fasthep_distributed/`
- Dask backend modules: `src/fasthep_distributed/_dask/`
- Profiles and registry YAML: `src/fasthep_distributed/profiles/`
- Dask tests: `tests/_dask/`
- Import smoke test: `tests/test_import.py`
- Documentation source: `docs/`

## Development Commands

Use Pixi from the repository root.

- Install/update the development environment: `pixi install`
- Format: `pixi run format`
- Lint: `pixi run lint`
- Autofix lint: `pixi run lint-fix`
- Type-check: `pixi run typecheck`
- Test: `pixi run test`
- Package build: `pixi run build`
- Build docs: `pixi run docs-build`
- Clean built docs: `pixi run docs-clean`
- Serve docs locally: `pixi run docs-serve`
- Standard validation: `pixi run check`
- Release-style validation: `pixi run ci`
- Distribution check after building: `pixi run check-dist`

`check` runs lint, type-check and tests. `ci` runs `check` and then builds the
package. Pixi environments include `py311`, `py312`, `py313`, `py314` and
`docs`; use environment-qualified runs when validating Python-version-specific
or docs-environment behavior.

## Public API and Compatibility

- Preserve the `fasthep_distributed` import namespace and
  `src/fasthep_distributed/py.typed`.
- Treat registry entries, backend specs, strategy names, worker-pool config,
  staging config and scheduler option normalization as extension surface area.
- Keep `_dask` modules internal unless an API is deliberately exported from
  `src/fasthep_distributed/_dask/__init__.py`.
- Preserve compatibility with `fasthep-flow >=2026.8.4.dev0` unless the package
  metadata, tests and downstream documentation are updated together.

## Generated Files and Special Care

- Do not edit `src/fasthep_distributed/_version.py` by hand; it is generated by
  `hatch-vcs`. Keep `src/fasthep_distributed/_version.pyi` aligned if public
  typing changes.
- Build-path artifacts for packed worker environments, staging bundles,
  scheduler submit files, logs and docs output are generated. Do not check them
  in unless they are intentional fixtures.
- The README and docs may lag implementation details; update active docs when
  changing supported distributed behavior.

## Testing Expectations

- Dask graph, strategy normalization, backend result and Flow contract changes
  need focused coverage in `tests/_dask/test_common.py`.
- Local strategy changes belong in `tests/_dask/test_local.py`.
- HTCondOR config, pooling, staging, timeout and directive changes belong in
  `tests/_dask/test_htcondor.py` and `tests/_dask/test_pooled.py`.
- Worker pool resource mapping belongs in `tests/_dask/test_pools.py`.
- Worker environment changes should test both `shared` and `transfer` staging,
  editable snapshots, packed-prefix validation and credential path handling.
- The default `pixi run test` suite uses mocked scheduler/client behavior and
  should run without a configured batch submit node. Live HTCondOR or SLURM
  submission checks require a site-configured submit node and should be treated
  as manual or integration validation outside the default unit suite.
- For broad backend-contract changes, run `pixi run check`; for narrow Python
  changes, run the smallest relevant tests plus `pixi run lint`.
