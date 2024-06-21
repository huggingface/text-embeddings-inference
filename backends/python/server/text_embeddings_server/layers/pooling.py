import torch
from flash_attn.bert_padding import pad_input

from loguru import logger

def mean_pooling(embedding, cu_seqlens, max_s):
    # Ideally, rust would pass `indices` to the FlashBatch.
    seqlens = cu_seqlens[1:].clone()
    seqlens[0] = cu_seqlens[1]
    seqlens[1:] -= cu_seqlens[1:-1]
    batch_size = len(seqlens)

    # Example: indices = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13]
    mask = torch.zeros(batch_size, max_s, dtype=torch.int32, device=cu_seqlens.device)
    mask[torch.arange(max_s) < seqlens[:, None].cpu()] = 1
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

    embedding_padded = pad_input(embedding, indices, batch_size, max_s)

    sum_embeddings = torch.sum(embedding_padded, 1)

    return sum_embeddings / seqlens[:, None]
