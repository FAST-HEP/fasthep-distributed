# Contributing

Thank you for contributing to FAST-HEP Distributed.

This repository is currently a package skeleton. Keep changes small and avoid
adding distributed backend behavior until the package boundary and public API
are explicitly defined.

Run the local checks before opening a pull request:

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run docs-build
```

Do not commit generated caches, local environments, or build outputs.
