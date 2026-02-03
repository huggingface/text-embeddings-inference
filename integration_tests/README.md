# Integration Tests

This directory contains integration tests for the project. This starts the TEI server and runs an /embed request to it while checking the output is as expected.

## How Tests Work

The tests use pytest fixtures to:
1. Start a Docker container with the TEI server
2. Wait for the server to become healthy
3. Send embedding requests and validate responses
4. Stop and remove the container after tests complete

The Docker image must be built before running tests. The `uv run pytest` command will start containers automatically using the pre-built image.

## Running the tests for HPU (Habana Gaudi)

First you have to build the docker image.
```bash
platform="hpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_hpu
```

Then you can run the tests.
```bash
cd integration_tests/gaudi
uv run pytest --durations=0 -sv .
```

### Environment Variables (HPU)

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_IMAGE` | Docker image to use | `tei_hpu` |
| `DOCKER_VOLUME` | Volume for model cache (recommended) | None |
| `HF_TOKEN` | HuggingFace token for gated models | None |
| `LOG_LEVEL` | Server log level | `info` |

## Running the tests for Neuron (AWS Inferentia/Trainium)

### Prerequisites

1. **AWS Neuron instance**: Tests must run on an EC2 instance with Neuron devices (inf1, inf2, or trn1)
2. **Neuron drivers**: Ensure Neuron drivers are installed and `/dev/neuron*` devices are available
3. **Pre-compiled models**: Neuron requires models to be pre-compiled to `.neuron` format

### Building the Docker Image

```bash
docker build . -f Dockerfile-neuron -t tei-neuron
```

### Running the Tests

```bash
cd integration_tests/neuron
uv run pytest --durations=0 -sv .
```

### Environment Variables (Neuron)

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_IMAGE` | Docker image to use | `tei-neuron` |
| `DOCKER_VOLUME` | Volume for model cache (recommended) | None |
| `HF_TOKEN` | HuggingFace token for gated models | None |
| `LOG_LEVEL` | Server log level | `info` |
| `NEURON_RT_NUM_CORES` | Number of Neuron cores to use | `1` |
| `NEURON_RT_VISIBLE_CORES` | Which Neuron cores are visible | `0` |

### Using Pre-compiled Neuron Models

Neuron models must be pre-compiled before use. You have two options:

1. **Use models with pre-compiled Neuron artifacts**: Some models on HuggingFace Hub have `.neuron` files available

2. **Compile models yourself**: Follow the [Optimum Neuron guide](https://huggingface.co/docs/optimum-neuron/en/model_doc/sentence_transformers/overview) to compile your models

Example compilation:
```python
from optimum.neuron import NeuronModelForSentenceTransformers

# Compile and save
model = NeuronModelForSentenceTransformers.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True,
    batch_size=1,
    sequence_length=512,
)
model.save_pretrained("./all-MiniLM-L6-v2-neuron")
model.push_to_hub("your-username/all-MiniLM-L6-v2-neuron")
```

### Troubleshooting Neuron Tests

**Container exits immediately**:
- Check if Neuron devices are available: `ls /dev/neuron*`
- Check container logs for "Neuron model files not found" - model needs compilation
- Ensure the Docker image was built with Neuron support

**Long startup times**:
- Neuron models may take several minutes to load due to compilation
- The test timeout is set to 600 seconds (10 minutes) by default

**Permission errors**:
- Ensure Docker has access to Neuron devices
- The tests add `IPC_LOCK` capability and mount `/dev/neuron*` devices

## Adding New Test Models

To add a new model to test, update the `TEST_CONFIGS` dictionary in `test_embed.py`:

```python
TEST_CONFIGS = {
    "your-model/name": {
        "model_id": "your-model/name",
        "input": "Test input text",
        "batch_inputs": ["Text 1", "Text 2"],
        "args": ["--dtype", "float32"],
        "env_config": {
            "MAX_WARMUP_SEQUENCE_LENGTH": "512",
        },
    },
}
```

For Habana tests, you can also add `expected_output` to validate exact embedding values.
