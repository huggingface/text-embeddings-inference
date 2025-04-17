include Makefile-flash-att
include Makefile-flash-att-v2

unit-tests:
	pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 'types-protobuf' --no-cache-dir
	mkdir text_embeddings_server/pb || true
	python -m grpc_tools.protoc -I../../proto --python_out=text_embeddings_server/pb \
		--grpc_python_out=text_embeddings_server/pb --mypy_out=text_embeddings_server/pb ../../proto/embed.proto
	find text_embeddings_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_embeddings_server/pb/__init__.py

install: gen-server
	pip install pip --upgrade
	pip install --no-deps -r requirements.txt
	pip install -e .

run-dev:
	python text_embeddings_server/cli.py serve BAAI/bge-small-en

export-requirements:
	poetry export -o requirements.txt --without-hashes
