from text_embeddings_server.utils.import_utils import SYSTEM
import os

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    class Attention:
        def __getattr__(self, name):
            raise RuntimeError(f"TEI is used with USE_FLASH_ATTENTION=false, accessing `attention` is prohibited")
    attention = Attention()
if SYSTEM == "cuda":
    from .cuda import attention
elif SYSTEM == "rocm":
    from .rocm import attention
else:
    raise ImportError(f"System {SYSTEM} doesn't support flash/paged attention")
