import os
from loguru import logger
import importlib.metadata
import importlib.util
from packaging import version
import torch
import subprocess


def _is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return (
            str(version.parse(full_version).major)
            + "."
            + str(version.parse(full_version).minor)
        )

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


def is_hpu() -> bool:
    is_hpu_available = True
    try:
        subprocess.run(["hl-smi"], capture_output=True, check=True)
    except:
        is_hpu_available = False
    return is_hpu_available


def use_ipex() -> bool:
    value = os.environ.get("USE_IPEX", "True").lower()
    return value in ["true", "1"] and _is_ipex_available()


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif is_hpu():
        import habana_frameworks.torch.core as htcore

        if hasattr(torch, "hpu") and torch.hpu.is_available():  # type: ignore
            device = torch.device("hpu")
    elif use_ipex():
        import intel_extension_for_pytorch as ipex

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")

    return device
