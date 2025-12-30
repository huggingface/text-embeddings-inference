import sys
import typer

from pathlib import Path
from loguru import logger
from typing import Optional
from enum import Enum

app = typer.Typer()


class Dtype(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bloat16 = "bfloat16"


@app.command()
def serve(
    model_path: Path,
    dtype: Dtype = "float32",
    uds_path: Path = "/tmp/text-embeddings-server",
    logger_level: str = "INFO",
    json_output: bool = False,
    otlp_endpoint: Optional[str] = None,
    otlp_service_name: str = "text-embeddings-inference.server",
    pool: str = "cls",
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_embeddings_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    # Import here after the logger is added to log potential import exceptions
    from text_embeddings_server import server
    from text_embeddings_server.utils.tracing import setup_tracing

    # Setup OpenTelemetry distributed tracing
    if otlp_endpoint is not None:
        setup_tracing(otlp_endpoint=otlp_endpoint, otlp_service_name=otlp_service_name)

    # Downgrade enum into str for easier management later on
    dtype = None if dtype is None else dtype.value
    server.serve(model_path, dtype, uds_path, pool)


if __name__ == "__main__":
    app()
