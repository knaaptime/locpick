# Installation

locpick supports Python >= 3.12. We recommend using [miniforge] or [pixi].

## Installing a released version

`locpick` is available on PyPI and can be installed with:

```bash
pip install locpick
```

For optional dependencies:

```bash
pip install locpick[jax]        # JAX backend for autodiff
pip install locpick[optax]      # Optax solver backend
pip install locpick[spatial]    # Spatial distance utilities
pip install locpick[all]        # All optional dependencies
```

## Installing from source

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/oturns/locpick.git
cd locpick
conda env create -f environment.yml
conda activate locpick
pip install -e . --no-deps
```

## Verifying the installation

```python
import locpick
print(locpick.__version__)
```

[miniforge]: https://github.com/conda-forge/miniforge
[pixi]: https://pixi.sh