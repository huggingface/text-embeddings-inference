import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


@pytest.mark.parametrize(
    "ipex,device_type,position,dtype,flash,expected",
    [
        (True, "cpu", "absolute", "float32", True, "default"),
        (True, "xpu", "relative_key", "float16", True, "default"),
        (True, "xpu", "absolute", "float16", False, "default"),
        (True, "xpu", "absolute", "float16", True, "flash"),
        (True, "cpu", "absolute", "bfloat16", True, "flash"),
        (False, "cuda", "absolute", "float16", True, "flash"),
        (False, "hpu", "absolute", "bfloat16", True, "flash"),
        (False, "cuda", "relative_key", "float16", True, "default"),
        (False, "cuda", "absolute", "float32", True, "default"),
        (False, "cpu", "absolute", "float16", True, "default"),
    ],
)
def test_bert_model_selection(ipex, device_type, position, dtype, flash, expected):
    # Execute the real factory with model loading stubbed, without importing GPU kernels.
    source = Path(__file__).parents[1] / "text_embeddings_server/models/__init__.py"
    factory = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_model"
    )
    config = SimpleNamespace(
        model_type="bert", position_embedding_type=position, architectures=["BertModel"]
    )
    namespace = {
        "Path": Path, "Optional": Optional, "torch": torch,
        "AutoConfig": SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
        "get_device": lambda: SimpleNamespace(type=device_type),
        "use_ipex": lambda: ipex, "FLASH_ATTENTION": flash, "TRUST_REMOTE_CODE": False,
        "logger": SimpleNamespace(info=lambda *args: None),
        "FlashBert": "flash", "DefaultModel": "default",
        "create_model": lambda model, *args: model,
    }
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["get_model"](Path("unused"), dtype, "cls") == expected
