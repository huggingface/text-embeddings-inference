import torch

from abc import ABC, abstractmethod
from typing import List, TypeVar, Type

from text_embeddings_server.models.types import Batch, Embedding, Score

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(
        self,
        model,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.model = model
        self.dtype = dtype
        self.device = device

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, batch: B) -> List[Embedding]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: B) -> List[Score]:
        raise NotImplementedError
