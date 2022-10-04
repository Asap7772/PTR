[![CircleCI](https://circleci.com/gh/ikostrikov/jaxrl2/tree/main.svg?style=svg&circle-token=c9b870c5ff1765e4c43ee08a190e8147f67d98b9)](https://circleci.com/gh/ikostrikov/jaxrl2/tree/main)
[![codecov](https://codecov.io/gh/ikostrikov/jaxrl2/branch/main/graph/badge.svg?token=8cE47NwU7g)](https://codecov.io/gh/ikostrikov/jaxrl2)
# jaxrl2

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

## Examples

[Here.](examples/)

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```

