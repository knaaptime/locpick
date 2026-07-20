# Installation

locpick supports Python >= 3.12. We recommend using [miniforge] or [pixi].

## Installing a released version

`locpick` is available on PyPI and can be installed with:

```bash
pip install locpick
```

JAX is a core dependency and is always installed. Optional extras add
alternative solvers, spatial utilities, and faster sparse solves:

```bash
pip install locpick[optax]       # Optax solver backend
pip install locpick[optimagic]   # optimagic solver backend
pip install locpick[optimistix]  # Optimistix (pure-JAX) solver backend
pip install locpick[spatial]     # Spatial graph / distance utilities
pip install locpick[sparse]      # CHOLMOD-accelerated sparse SAR solves
pip install locpick[all]         # All of the above
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