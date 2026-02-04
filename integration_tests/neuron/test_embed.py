from typing import Any, Dict, Generator
from _pytest.fixtures import SubRequest

import pytest
import pytest_asyncio
import numpy as np


# Test configurations for Neuron backend
TEST_CONFIGS = {
    # # On-the-fly Neuron compilation
    # "sentence-transformers/all-MiniLM-L6-v2": {
    #     "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    #     "input": "What is Deep Learning?",
    #     "batch_inputs": [
    #         "What is Deep Learning?",
    #         "How does machine learning work?",
    #         "Tell me about neural networks.",
    #     ],
    #     "expected_output_prefix": None,
    #     "args": [
    #         "--dtype", "float32",
    #         "--max-batch-requests", "1",
    #     ],
    #     "env_config": {
    #         "MAX_WARMUP_SEQUENCE_LENGTH": "512",
    #     },
    # },
    # "BAAI/bge-base-en-v1.5": {
    #     "model_id": "BAAI/bge-base-en-v1.5",
    #     "input": "What is Deep Learning?",
    #     "batch_inputs": [
    #         "What is Deep Learning?",
    #         "How does machine learning work?",
    #         "Tell me about neural networks.",
    #     ],
    #     "expected_output_prefix": None,
    #     "args": [
    #         "--dtype", "float32",
    #         "--max-batch-requests", "1",
    #     ],
    #     "env_config": {
    #         "MAX_WARMUP_SEQUENCE_LENGTH": "512",
    #     },
    # },
    # Pre-compiled Neuron model
    "optimum/bge-base-en-v1.5-neuronx": {
        "model_id": "optimum/bge-base-en-v1.5-neuronx",
        "input": "What is Deep Learning?",
        "batch_inputs": [
            "What is Deep Learning?",
            "How does machine learning work?",
            "Tell me about neural networks.",
        ],
        "expected_output_prefix": None,
        "args": [
            "--dtype", "float32",
            "--max-batch-requests", "1",
        ],
        "env_config": {
            "MAX_WARMUP_SEQUENCE_LENGTH": "512",
        },
    },
}


@pytest.fixture(scope="module", params=TEST_CONFIGS.keys())
def test_config(request: SubRequest) -> Dict[str, Any]:
    """Fixture that provides model configurations for testing."""
    model_name = request.param
    test_config = TEST_CONFIGS[model_name].copy()
    test_config["test_name"] = model_name
    return test_config


@pytest.fixture(scope="module")
def model_id(test_config: Dict[str, Any]) -> Generator[str, None, None]:
    yield test_config["model_id"]


@pytest.fixture(scope="module")
def test_name(test_config: Dict[str, Any]) -> Generator[str, None, None]:
    yield test_config["test_name"]


@pytest.fixture(scope="module")
def input_text(test_config: Dict[str, Any]) -> str:
    return test_config["input"]


@pytest.fixture(scope="module")
def batch_inputs(test_config: Dict[str, Any]) -> list:
    return test_config.get("batch_inputs", [test_config["input"]])


@pytest.fixture(scope="module")
def expected_outputs(test_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "expected_output_prefix": test_config.get("expected_output_prefix"),
    }


@pytest.fixture(scope="function")
def tei_service(neuron_launcher, model_id: str, test_name: str):
    with neuron_launcher(model_id, test_name) as tei_service:
        yield tei_service


@pytest_asyncio.fixture(scope="function")
async def tei_client(tei_service):
    # Neuron models may take longer to load due to compilation
    await tei_service.health(600)  # 10 minute timeout for Neuron compilation
    return tei_service


@pytest.mark.asyncio
async def test_model_single_request(
    tei_client, expected_outputs: Dict[str, Any], input_text: str
):
    """Test single embedding request."""
    response = await tei_client.embed(input_text)

    # Verify response structure
    assert isinstance(response, list), f"Expected list, got {type(response)}"
    assert len(response) > 0, "Embedding should not be empty"

    response_array = np.array(response)

    # Check that values are numeric
    assert response_array.dtype in [np.float32, np.float64, np.float16], \
        f"Expected float array, got {response_array.dtype}"

    # If expected output is provided, validate against it
    expected_prefix = expected_outputs.get("expected_output_prefix")
    if expected_prefix is not None:
        expected_array = np.array(eval(expected_prefix) if isinstance(expected_prefix, str) else expected_prefix)
        prefix_len = len(expected_array.flatten())
        response_flat = response_array.flatten()[:prefix_len]

        if not np.allclose(response_flat, expected_array.flatten(), rtol=1e-4, atol=1e-4):
            print("\nExpected output (prefix):")
            print(f"{expected_array.tolist()}")
            print("\nReceived output (prefix):")
            print(f"{response_flat.tolist()}")
            raise AssertionError("Response array does not match expected array within tolerance")

    # Check embedding dimensions are reasonable (typically 384, 768, 1024, etc.)
    embedding_dim = response_array.shape[-1] if response_array.ndim > 1 else len(response_array)
    assert embedding_dim > 0, "Embedding dimension should be positive"

    print(f"Single request embedding shape: {response_array.shape}")
    print(f"Embedding dimension: {embedding_dim}")


@pytest.mark.asyncio
async def test_model_batch_request(tei_client, batch_inputs: list):
    """Test batch embedding request."""
    response = await tei_client.embed_batch(batch_inputs)

    # Verify response is a list of embeddings
    assert isinstance(response, list), f"Expected list, got {type(response)}"
    assert len(response) == len(batch_inputs), \
        f"Expected {len(batch_inputs)} embeddings, got {len(response)}"

    response_array = np.array(response)
    print(f"Batch request response shape: {response_array.shape}")

    # Check each embedding
    for i, embedding in enumerate(response):
        assert isinstance(embedding, list), f"Embedding {i} should be a list"
        assert len(embedding) > 0, f"Embedding {i} should not be empty"


@pytest.mark.asyncio
async def test_model_embedding_consistency(tei_client, input_text: str):
    """Test that the same input produces consistent embeddings."""
    response1 = await tei_client.embed(input_text)
    response2 = await tei_client.embed(input_text)

    array1 = np.array(response1)
    array2 = np.array(response2)

    # Embeddings for the same input should be identical (or very close)
    assert np.allclose(array1, array2, rtol=1e-4, atol=1e-4), \
        "Same input should produce consistent embeddings"

