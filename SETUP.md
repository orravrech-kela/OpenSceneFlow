# Environment setup

This repo can be set up via either `uv` (faster, modern) or `conda` (matches the
upstream `environment.yaml`). The `uv` path is what we actively use; the `conda`
path is here for reproducibility against upstream.

## Prerequisites

- **GPU**: a CUDA-capable device. We currently develop on an NVIDIA L4 with the
  CUDA 12.8 system toolkit (driver 570). Any modern CUDA ≥ 12.1 host works.
- **`nvcc`** on `PATH` (needed to build the in-repo CUDA extensions). On this
  host: `/usr/local/cuda-12.8/bin/nvcc`.
- For `uv`: `uv ≥ 0.8` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Recommended: uv path

This is what gets you to a working training stack from scratch in a single
session.

```bash
cd /home/ubuntu/orr/dev/forks/OpenSceneFlow

# 1. Create the venv (Python 3.10 — pinned upstream).
uv venv --python 3.10

# 2. Install pinned training stack (torch cu117 baseline + lightning + spconv + …).
#    requirements-train.txt transitively includes requirements-extract.txt.
uv pip install -r requirements-train.txt

# 3. Override the stale torch/spconv pins so they match the host CUDA toolkit.
#    cu117 wheels can't compile against nvcc 12.x; we promote to cu121.
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.1.2' 'torchvision==0.16.2'
uv pip uninstall spconv-cu117
uv pip install 'numpy<2' 'spconv-cu121'

# 4. Restore modern pydantic/fastapi (lightning==2.0.1 pulled in fastapi==0.1.17
#    which is incompatible with anything else).
uv pip install 'pydantic>=2' 'fastapi>=0.115'

# 5. Build the two in-repo CUDA extensions against the now-installed torch.
#    TORCH_CUDA_ARCH_LIST: pick architectures present on your GPU(s).
#    8.9 = L4 / RTX 4090.  Add others (e.g. "7.5;8.0;8.6;8.9") for a multi-arch
#    wheel if you train on heterogeneous GPUs.
TORCH_CUDA_ARCH_LIST="8.9" \
  uv pip install --no-build-isolation ./assets/cuda/chamfer3D
TORCH_CUDA_ARCH_LIST="8.9" \
  uv pip install --no-build-isolation ./assets/cuda/mmcv
```

### Verify

```bash
.venv/bin/python -c "
import torch, lightning, pytorch_lightning, spconv, h5py, linefit
from assets.cuda.chamfer3D import nnChamferDis
from assets.cuda.mmcv import Voxelization
print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
"
```

Expected output:

```
torch: 2.1.2+cu121 cuda? True
device: NVIDIA L4
```

### Why the manual steps after `requirements-train.txt`?

`requirements-train.txt` was written for the upstream `cu117` baseline and pulls
in `lightning==2.0.1`, whose `lightning.app` subpackage transitively depends on
`fastapi==0.1.17` (pre-pydantic-1.0 API). Modern resolvers install
`pydantic 2.x`, which breaks the chain at import time. We sidestep by:

- Using `pytorch_lightning` directly in our code (`import pytorch_lightning as pl`) — the
  metapackage `lightning` is no longer imported, so its `app` subpackage is dormant.
- Upgrading `pydantic`/`fastapi` to modern versions for everything else
  (`wandb`, etc.).

The torch upgrade to `cu121` is so that `nvcc 12.x` on the host can compile the
two in-repo CUDA extensions against torch's bundled CUDA libs. Compiling with a
mismatched `nvcc` is rejected by `torch.utils.cpp_extension._check_cuda_version`.

## Alternative: conda path

Mirrors the upstream `environment.yaml` (cu117 baseline). Use this only if your
host already has CUDA 11.7 + nvcc 11.7, e.g. via `nvidia/label/cuda-11.7.0::cuda`.
Otherwise the extension build will fail with the same `nvcc` mismatch you'd hit
under `uv`, and you'd end up doing the same overrides.

```bash
conda env create -f environment.yaml      # creates env "opensf"
conda activate opensf
cd assets/cuda/chamfer3D && python setup.py install && cd -
cd assets/cuda/mmcv && python setup.py install && cd -
```

### Is the conda path easier?

**No, on this host.** The conda recipe is pinned to CUDA 11.7, but the system
nvcc is 12.8. To make the conda env actually compile the CUDA extensions you'd
need to additionally install the 11.7 toolkit (`nvidia/label/cuda-11.7.0::cuda`,
~3 GB), and you'd still hit the same `lightning.app` vs `pydantic 2.x` import
trap unless you replicate the `lightning.pytorch` → `pytorch_lightning` swap.

The `uv` path is faster (10s of seconds for `uv pip install` vs. several
minutes for `conda env create`), lighter (no extra 11.7 toolkit needed), and
already known-good.

**Use conda if**: you need the exact upstream environment for paper-reproducibility
debugging, or you're on a host that already has CUDA 11.7 installed.

**Use uv otherwise.**

## Common pitfalls

- **`ModuleNotFoundError: No module named 'torch'` while building chamfer3D / mmcv**:
  you forgot `--no-build-isolation`. The extensions' `setup.py` imports torch at
  build time and assumes torch is already in the active environment.

- **`RuntimeError: The detected CUDA version (12.8) mismatches the version that
  was used to compile PyTorch (11.7)`**: you skipped step 3 above and are still
  on torch cu117. Upgrade torch to cu121.

- **`ImportError: cannot import name 'Schema' from 'pydantic'`**: lightning's app
  subpackage is being imported and hit pre-pydantic-1.0 API. We use
  `pytorch_lightning` directly to avoid this; do not switch back to
  `import lightning.pytorch as pl` in our code.

- **`ImportError: Could not load mmcv extension with functions: [...]`**: the
  `assets/cuda/mmcv` build failed silently. Re-run step 5 and look for compiler
  errors. The most common cause is a mismatched `TORCH_CUDA_ARCH_LIST` for your
  GPU.

- **Slow first iteration during training**: cumm/spconv JIT-compiles kernels on
  first use. The next iteration will be ~10× faster.
