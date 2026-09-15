import logging
from pathlib import Path

from text_embeddings_server.utils.device import is_rocm

logger = logging.getLogger(__name__)

_LOCKFILE = Path(__file__).resolve().parents[2] / "kernels.lock"
_KERNEL_REPO = "kernels-community/triton-layer-norm"
_KERNEL_REVISION = "v0.1.0"  # keep in sync with kernels.lock

_triton_layer_norm = None
if is_rocm():
    try:
        # Prefer the pre-downloaded, locked kernel from the local cache
        # (`load_kernel` uses `local_files_only=True`). In the container the
        # kernel is baked in at build time via `kernels download` (see
        # Dockerfile-rocm), so this path never reaches out to the Hub.
        from kernels import load_kernel

        _triton_layer_norm = load_kernel(_KERNEL_REPO, lockfile=_LOCKFILE)
    except Exception as e:
        # Not pre-downloaded (e.g. a bare clone run outside the container).
        # Fall back to fetching the locked revision from the Hub at runtime.
        logger.info(
            f"{_KERNEL_REPO} not found in local cache ({e}); "
            f"downloading it from the Hub."
        )
        try:
            from kernels import get_kernel

            _triton_layer_norm = get_kernel(_KERNEL_REPO, revision=_KERNEL_REVISION)
        except Exception as e:
            logger.warning(
                f"Could not load the {_KERNEL_REPO} kernel, falling back to "
                f"torch layer norm: {e}"
            )


def get_triton_layer_norm():
    """Return the pre-downloaded triton-layer-norm kernel, or None if unavailable."""
    return _triton_layer_norm
