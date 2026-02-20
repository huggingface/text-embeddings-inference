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

## Running the tests for Neuron (AWS Inferentia/Trainium)

### Prerequisites

1. **AWS Neuron instance**: Tests must run on an EC2 instance with Neuron devices (inf2, trn1 or trn2)
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

