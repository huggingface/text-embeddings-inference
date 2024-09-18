import torch
from loguru import logger

SYSTEM = None
if torch.version.hip is not None:
    SYSTEM = "rocm"
elif torch.version.cuda is not None and torch.cuda.is_available():
    SYSTEM = "cuda"
else:
    SYSTEM = "cpu"

logger.info(f"Python backend: detected system {SYSTEM}")
