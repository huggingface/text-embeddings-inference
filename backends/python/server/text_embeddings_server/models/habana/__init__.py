import os

DISABLE_TENSOR_CACHE = os.getenv("DISABLE_TENSOR_CACHE", "false").lower() in ["true", "1"]


def wrap_model_if_hpu(model_handle, device):
    """Wrap the model in HPU graph if the device is HPU."""
    if device.type == "hpu":
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        model_handle.model = wrap_in_hpu_graph(
            model_handle.model, disable_tensor_cache=DISABLE_TENSOR_CACHE
        )
    return model_handle
