## Installation Instructions

This project uses [pixi](https://pixi.sh) for environment and build management. MLIR, LLVM tools, `lit`, and the MLIR Python bindings are resolved from conda-forge through pixi; no extra PyPI package index is required.

1. Install dependencies:

```bash
pixi install
```

2. Configure CMake:

```bash
pixi run configure
```

3. Build:

```bash
pixi run build
```

4. Run tests:

```bash
pixi run test
```

To clean build artifacts:

```bash
pixi run clean
```

## Installing Python Package

The `tiny` Python DSL package is automatically installed as an editable dependency by `pixi install`. To verify the package and MLIR Python bindings:

```bash
pixi run python -c "from mlir.ir import Context; import tiny; print('tiny package and MLIR bindings loaded')"
```

## Running Python Examples

1. Set the `TUTORIAL_OPT` environment variable to point to the built `tutorial-opt` binary. Pixi sets this automatically for `pixi run` commands; for manual shells, use:

```bash
export TUTORIAL_OPT=$PWD/build/tutorial/tutorial-opt
```

2. Run an example to verify:

```bash
pixi run python tutorial/ch1-cpu-vector-dsl/square.py
```

The examples will print the generated MLIR at each lowering stage.

## Doing tutorial exercises

Each tutorial chapter has a README.md file explaining how to start with the exercise and some basic explanation of the exercise. For example, check out tutorial/ch1-cpu-vector-dsl/README.md
