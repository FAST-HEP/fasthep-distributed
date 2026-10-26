# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project uses
calendar versioning (`YYYY.MM.N`) rather than semantic versioning.

Add new entries under `## [Unreleased]`, in the category they belong to. At release time, rename
`## [Unreleased]` to `## [<version>] - <YYYY-MM-DD>` and start a fresh `## [Unreleased]` above it.

## [Unreleased]

### Added

- A probe tool to validate distributed-execution readiness at a site before submitting real jobs,
  covering CERN-specific HTCondor recommendations, XRootD reads, EOS writes, and output transfer
  verification.
- HTCondor site options and a CERN profile, plus a client context manager for the HTCondor
  scheduler.
- A `shared-prefix` worker environment mode: workers run directly from an existing Pixi prefix on a
  shared filesystem instead of packing and transferring one.
- Delivery of writer outputs directly to EOS from HTCondor workers (`output_destination` /
  `output_files`), verified against the writer manifests after the job exits.
- Graceful HTCondor worker shutdown: workers are retired and given time to finish in-flight jobs
  before the cluster closes.

### Changed

- Probe and worker execution now resolve through Flow's `CompileContext`, aligning with Flow's
  compile-time path resolution.
- Dask partition tasks now merge hook results and return only boundary products across the
  scheduler boundary.

### Fixed

- Workers holding task results now release them before being retired, preventing a shutdown hang.
- The shared-prefix bootstrap script now joins correctly when commands are combined with `;`.
- Worker probe reporting now reflects per-process job state instead of a single aggregate result.

## [2026.08.1] - 2026-08-28

### Added

- Initial Dask backend implementation for Flow execution plans, with local, pooled, and HTCondor
  worker strategies.
- A `dask` backend registry entry with a local-execution profile.

### Changed

- Backend policy moved into registry specifications rather than being hardcoded per backend.

### Fixed

- Packed worker environment extraction now filters correctly.
