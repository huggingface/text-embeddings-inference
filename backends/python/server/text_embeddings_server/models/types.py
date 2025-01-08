import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from opentelemetry import trace

from text_embeddings_server.pb import embed_pb2
from text_embeddings_server.pb.embed_pb2 import Embedding

tracer = trace.get_tracer(__name__)


class Batch(ABC):
    @classmethod
    @abstractmethod
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class PaddedBatch(Batch):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "PaddedBatch":
        # Allocate padded tensors all at once
        all_tensors = torch.zeros(
            [4, len(pb.cu_seq_lengths) - 1, pb.max_length], dtype=torch.int32
        )

        for i, start_index in enumerate(pb.cu_seq_lengths[:-1]):
            end_index = pb.cu_seq_lengths[i + 1]
            input_length = end_index - start_index

            all_tensors[0, i, :input_length] = torch.tensor(
                pb.input_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[1, i, :input_length] = torch.tensor(
                pb.token_type_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[2, i, :input_length] = torch.tensor(
                pb.position_ids[start_index:end_index], dtype=torch.int32
            )
            all_tensors[3, i, :input_length] = 1

        # Move padded tensors all at once
        all_tensors = all_tensors.to(device)

        return PaddedBatch(
            input_ids=all_tensors[0],
            token_type_ids=all_tensors[1],
            position_ids=all_tensors[2],
            attention_mask=all_tensors[3],
        )

    def __len__(self):
        return len(self.input_ids)


@dataclass
class FlashBatch(Batch):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor

    cu_seqlens: torch.Tensor
    max_s: int
    size: int

    @classmethod
    @tracer.start_as_current_span("from_pb")
    def from_pb(cls, pb: embed_pb2.EmbedRequest, device: torch.device) -> "FlashBatch":
        batch_input_ids = torch.tensor(pb.input_ids, dtype=torch.int32, device=device)
        batch_token_type_ids = torch.tensor(
            pb.token_type_ids, dtype=torch.int32, device=device
        )
        batch_position_ids = torch.tensor(
            pb.position_ids, dtype=torch.int32, device=device
        )

        cu_seqlens = torch.tensor(pb.cu_seq_lengths, dtype=torch.int32, device=device)

        return FlashBatch(
            input_ids=batch_input_ids,
            token_type_ids=batch_token_type_ids,
            position_ids=batch_position_ids,
            cu_seqlens=cu_seqlens,
            max_s=pb.max_length,
            size=len(cu_seqlens) - 1,
        )

    def __len__(self):
        return self.size
