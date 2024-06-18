from text_embeddings_server.utils.import_utils import SYSTEM
import os

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")
if SYSTEM == "cuda":
    from .cuda import attention
elif SYSTEM == "rocm":
    from .rocm import attention
else:
    raise ImportError(f"System {SYSTEM} doesn't support flash/paged attention")
