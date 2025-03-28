# Candle Cuda Layer Norm

Layer Norm fused operation for the Candle ML framework.

This Layer was adapted from https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm.

It implements fused dropout + residual + LayerNorm, building on Apex's FastLayerNorm.

Major changes:

- Add residual.
- Make it work for both pre-norm and post-norm architecture.
- Support more hidden dimensions (all dimensions divisible by 8, up to 8192).
- Implement RMSNorm as an option.
