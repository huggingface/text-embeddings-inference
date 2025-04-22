run-integration-tests-hpu:
	DOCKER_VOLUME=/home/sdp/git/text-embeddings-inference/data \
	HF_TOKEN=`cat ${HOME}/.cache/huggingface/token` \
	uv run pytest --durations=0 -sv .


