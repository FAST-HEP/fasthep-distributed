# FAST-HEP Distributed

`fasthep-distributed` is the planned home for reusable distributed execution
support in FAST-HEP.

This repository currently contains package scaffolding only. Distributed
backends and runtime integrations are not implemented here yet.

## Development

Install the local Pixi environment:

```bash
pixi install
```

Run the baseline checks:

```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run docs-build
```
