# Integration Tests

This directory contains integration tests for the project. This starts the TEI server and run an /embed request to it while checking the output is as expected.

## Running the tests for HPU

First you have to build the docker image.
```bash
platform="hpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_hpu
```

Then you can run the tests.
```bash
uv run pytest --durations=0 -sv .
```
