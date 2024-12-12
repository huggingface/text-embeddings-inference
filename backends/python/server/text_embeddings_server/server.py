import asyncio
import torch
from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import Optional

from text_embeddings_server.models import Model, get_model
from text_embeddings_server.pb import embed_pb2_grpc, embed_pb2
from text_embeddings_server.utils.tracing import UDSOpenTelemetryAioServerInterceptor
from text_embeddings_server.utils.interceptor import ExceptionInterceptor


class EmbeddingService(embed_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, model: Model):
        self.model = model
        # Force inference mode for the lifetime of EmbeddingService
        self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2), device="cuda")
        return embed_pb2.HealthResponse()

    async def Embed(self, request, context):
        batch = self.model.batch_type.from_pb(request, self.model.device)

        embeddings = self.model.embed(batch)

        return embed_pb2.EmbedResponse(embeddings=embeddings)


def serve(
    model_path: Path,
    dtype: Optional[str],
    uds_path: Path,
    pool: str,
):
    async def serve_inner(
        model_path: Path,
        dtype: Optional[str] = None,
    ):
        unix_socket = f"unix://{uds_path}"

        try:
            model = get_model(model_path, dtype, pool)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        embed_pb2_grpc.add_EmbeddingServiceServicer_to_server(
            EmbeddingService(model), server
        )
        SERVICE_NAMES = (
            embed_pb2.DESCRIPTOR.services_by_name["EmbeddingService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(unix_socket)

        await server.start()

        logger.info(f"Server started at {unix_socket}")

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(model_path, dtype))
