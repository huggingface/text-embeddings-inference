## Testing

To run the tests, install from within docker with `--entrypoint "/bin/bash"` the requirements
```
pip install -r requirements.txt
```

and mounting a volume for the tests, they can be run from within the container with
```
pytest tests/ -s -vvvvv
```

## Reference outputs

For example, collecting the reference on an RTX 4090 on Candle backend:
```
docker run --rm -it --gpus all --net host --entrypoint "/bin/bash" -v $(pwd):/tei ghcr.io/huggingface/text-embeddings-inference:89-1.2.3
```
and
```
text-embeddings-router --model-id sentence-transformers/all-MiniLM-L6-v2
```

and then
```
python collect.py --model-id sentence-transformers/all-MiniLM-L6-v2 --n_inp 1 --flash
python collect.py --model-id sentence-transformers/all-MiniLM-L6-v2 --n_inp 3 --flash
```

Restart server with `USE_FLASH_ATTENTION=0`, and
```
python collect.py --model-id sentence-transformers/all-MiniLM-L6-v2 --n_inp 1
python collect.py --model-id sentence-transformers/all-MiniLM-L6-v2 --n_inp 3
```