import importlib
import sys

import text_embeddings_server.utils.device as device


def test_rocm_varlen_flag_is_defined_on_non_rocm_backends(monkeypatch):
    """Decoder models import ROCM_HAS_FA_VARLEN at module import time.

    IPEX and HPU take the early branch and never entered the ROCm block, so the
    name was missing and `import text_embeddings_server.models` failed.
    """
    monkeypatch.setattr(device, "use_ipex", lambda: True)
    monkeypatch.setattr(device, "is_hpu", lambda: False)
    monkeypatch.setattr(device, "is_rocm", lambda: False)
    monkeypatch.delenv("USE_FLASH_ATTENTION", raising=False)
    sys.modules.pop("text_embeddings_server.utils.flash_attn", None)

    flash_attn = importlib.import_module("text_embeddings_server.utils.flash_attn")

    assert flash_attn.ROCM_HAS_FA_VARLEN is False
    assert flash_attn.HAS_FLASH_ATTN_V2 is True
