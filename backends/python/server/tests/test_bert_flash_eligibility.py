import torch

from text_embeddings_server.utils.flash_eligibility import bert_flash_eligible


def test_ipex_does_not_skip_dtype_and_position_checks():
    assert (
        bert_flash_eligible(True, "cpu", "absolute", torch.float32, True) is False
    )
    assert (
        bert_flash_eligible(True, "xpu", "relative", torch.float16, True) is False
    )
    assert bert_flash_eligible(True, "xpu", "absolute", torch.float16, False) is False


def test_absolute_half_precision_still_uses_flash_bert():
    for device in ("cuda", "hpu", "cpu"):
        assert (
            bert_flash_eligible(True, device, "absolute", torch.bfloat16, True) is True
        )
    assert bert_flash_eligible(False, "cuda", "absolute", torch.float16, True) is True
    assert bert_flash_eligible(False, "cpu", "absolute", torch.float16, True) is False
