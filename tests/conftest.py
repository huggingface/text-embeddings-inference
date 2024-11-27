import pytest
import asyncio
import contextlib
import random
import os
import tempfile
import subprocess
import shutil
import sys
from typing import Optional
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
import requests
import time
from requests.exceptions import ConnectionError as RequestsConnectionError

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

class ProcessLauncherHandle:
    def __init__(self, process, port: int):
        self.port = port
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None

    def health(self, timeout: int = 60):
        assert timeout > 0
        for _ in range(timeout):
            if not self._inner_health():
                raise RuntimeError("Launcher crashed")

            try:
                url = f"http://0.0.0.0:{self.port}/health"
                headers = {"Content-Type": "application/json"}

                response = requests.post(url, headers=headers)
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError, RequestsConnectionError) as e:
                print("Connecting")
                time.sleep(1)
        raise RuntimeError("Health check failed")

@pytest.fixture(scope="module")
def launcher(event_loop):
    @contextlib.contextmanager
    def local_launcher(
        model_id: str,
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
        dtype: Optional[str] = None,
        revision: Optional[str] = None,
        pooling: Optional[str] = None,
    ):
        port = random.randint(8000, 10_000)
        shard_uds_path = (
            f"/tmp/tei-tests-{model_id.split('/')[-1]}-server"
        )

        args = [
            "text-embeddings-router",
            "--model-id",
            model_id,
            "--port",
            str(port),
            "--uds-path",
            shard_uds_path,
        ]

        env = os.environ

        if dtype is not None:
            args.append("--dtype")
            args.append(dtype)
        if revision is not None:
            args.append("--revision")
            args.append(revision)
        if trust_remote_code:
            args.append("--trust-remote-code")
        if pooling:
            args.append("--pooling")
            args.append(str(max_input_length))

        env["LOG_LEVEL"] = "debug"

        if not use_flash_attention:
            env["USE_FLASH_ATTENTION"] = "false"

        with tempfile.TemporaryFile("w+") as tmp:
            # We'll output stdout/stderr to a temporary file. Using a pipe
            # cause the process to block until stdout is read.
            print("call subprocess.Popen, with args", args)
            with subprocess.Popen(
                args,
                stdout=tmp,
                stderr=subprocess.STDOUT,
                env=env,
            ) as process:
                yield ProcessLauncherHandle(process, port)

                process.terminate()
                process.wait(60)

                tmp.seek(0)
                shutil.copyfileobj(tmp, sys.stderr)

        if not use_flash_attention:
            del env["USE_FLASH_ATTENTION"]

    return local_launcher
