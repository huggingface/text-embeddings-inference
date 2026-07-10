# AGENTS.md

Notes for coding agents (and humans) deploying or benchmarking Text Embeddings Inference (TEI)
on GPU hardware, especially NVIDIA Turing (T4, RTX 2000 series, ...).

## Turing (T4) GPUs: avoid the `cuda-1.9` quickstart image without an explicit Flash Attention override

The README's headline Docker quickstart uses the multi-architecture
`ghcr.io/huggingface/text-embeddings-inference:cuda-1.9` image. On Turing GPUs this image
correctly auto-detects the architecture and dispatches the Turing-compatible binary, but it still
starts with `USE_FLASH_ATTENTION=True` by default — that default is baked in at build time and is
not adjusted per-architecture the way the binary selection is.

This has been observed on real T4 hardware to go beyond the "precision issues" mentioned in the
README's Docker Images table: the server starts cleanly, responds HTTP 200 with `Success` logged
for every request, and returns embedding vectors that are **entirely NaN** — with no warning or
error surfaced anywhere. This was reproduced across two unrelated model architectures
(`Qwen/Qwen3-Embedding-0.6B` and `BAAI/bge-small-en-v1.5`), 3+ runs each, 100% reproduction rate.

**If you are deploying on a Turing GPU:**
- Prefer the dedicated `turing-1.9` image, which defaults to `USE_FLASH_ATTENTION=False`, or
- If you must use `cuda-1.9`, explicitly pass `-e USE_FLASH_ATTENTION=False`.

Do not rely on the absence of errors/warnings as a signal that flash attention is safe on Turing —
in the broken configuration there is no signal at all; the only symptom is corrupted output.

## Turning Flash Attention off can trigger an OOM at the default `--max-batch-tokens`

Once Flash Attention is disabled (either via `turing-1.9` or `USE_FLASH_ATTENTION=False`), some
models can fail to start with:

```
Error: Model backend is not healthy

Caused by:
    DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

This was reproduced with `Qwen/Qwen3-Embedding-0.6B` at the default `--max-batch-tokens=16384` on
a 16GB T4 (the non-flash attention path appears to require materializing larger intermediate
tensors during warmup). `BAAI/bge-small-en-v1.5`, whose default `--max-batch-tokens` is much
smaller (512), did not hit this.

If you hit an OOM after disabling Flash Attention, lower `--max-batch-tokens` (e.g. `2048`) rather
than assuming the GPU is undersized — peak steady-state VRAM for these models is well under 2GB on
a 16GB T4.

## dtype defaults are already safe on this GPU family

TEI's GPU default dtype is `float16`; there is no `bfloat16` code path in the candle backend, so
the common "silent bf16 fallback" trap seen in other inference servers does not apply here.
`--dtype float32` is available if needed, but on T4 `float16` measured ~18% faster and ~23% lower
VRAM than `float32` for the same model.

## `--quantize` is not a real TEI CLI flag

`int8` quantization and CUDA graphs have no implementation in this codebase (not "supported but
broken" — simply absent). One doc page previously referenced a `--quantize` CLI flag that does not
exist in `docs/source/en/cli_arguments.md` or in `router/src/main.rs`; that reference has been
removed. Don't add code assuming `--quantize` exists without first adding the feature itself.
