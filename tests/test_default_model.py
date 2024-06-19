import pytest
import requests
import json
import torch

@pytest.fixture(scope="module")
def default_model_handle(launcher):
    with launcher("sentence-transformers/all-MiniLM-L6-v2", use_flash_attention=False) as handle:
        yield handle

@pytest.fixture(scope="module")
async def default_model(default_model_handle):
    default_model_handle.health(300)
    return default_model_handle

@pytest.mark.asyncio
@pytest.mark.private
async def test_single_query(default_model):
    url = f"http://0.0.0.0:{default_model.port}/embed"
    data = {"inputs": "What is Deep Learning?"}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=data, headers=headers)

    embedding = torch.Tensor(json.loads(response.text))
    reference_embedding = torch.load("./tests/assets/sentence-transformers-all-MiniLM-L6-v2_inp1_no_flash.pt")

    assert torch.allclose(embedding, reference_embedding, atol=1e-3, rtol=1e-3)