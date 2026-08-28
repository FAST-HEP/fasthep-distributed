# Development

Install the project environment from the repository root:

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

Build and inspect distributions:

```bash
pixi run build
pixi run check-dist
```
