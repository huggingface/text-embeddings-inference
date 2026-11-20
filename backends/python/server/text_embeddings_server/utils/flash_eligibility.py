import torch


def bert_flash_eligible(ipex, device_type, position_embedding_type, datatype, flash_attention):
    """FlashBert is only valid for absolute positions in fp16 or bf16.

    `and` binds tighter than `or`, so `ipex or cuda and dtype and flash` lets every
    IPEX model into the flash branch, including float32 and relative positions.
    """
    return (
        (ipex or device_type in ["cuda", "hpu"])
        and position_embedding_type == "absolute"
        and datatype in (torch.float16, torch.bfloat16)
        and flash_attention
    )
