import torch
import tempfile
import os

from typing import Tuple
from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin

from .utils import PretrainedMixin
from ..common.utils import read_from_url
from .dit.na import flatten


class PrecomputedEmbeddings(PretrainedMixin, FlashPackDiffusersModelMixin):
    def __init__(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("positive", positive)
        self.register_buffer("negative", negative if negative is not None else torch.zeros_like(positive))

    def get(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the precomputed embeddings.
        """
        return flatten([self.positive]), flatten([self.negative])

    @classmethod
    def default(
        cls,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "PrecomputedEmbeddings":
        """
        Load the default precomputed embeddings from the Hugging Face Hub.
        """
        from seedvr.data import NEG_EMB_PATH, POS_EMB_PATH
        positive = torch.load(POS_EMB_PATH)
        negative = torch.load(NEG_EMB_PATH)
        embeddings = cls(positive, negative)
        return embeddings.to(device=device, dtype=dtype)