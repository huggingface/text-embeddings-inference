import os
from loguru import logger
import importlib
from packaging import version
import torch

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True

def use_ipex() :
    value = os.environ.get("USE_IPEX", "True")
    if value in ["True", "true", "1"] and is_ipex_available():
        return True
    else:
        return False

def get_device() :
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_ipex_available():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

