## Testing

To run the tests, install from within docker with `--entrypoint "/bin/bash"` the requirements
```
pip install -r requirements.txt
```

and mounting a volume for the tests, they can be run from within the container with
```
pytest tests/ -s -vvvvv
```