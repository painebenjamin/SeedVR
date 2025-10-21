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
        with tempfile.TemporaryDirectory() as temp_dir:
            neg_embeds_path = os.path.join(temp_dir, "neg_emb.pt")
            pos_embeds_path = os.path.join(temp_dir, "pos_emb.pt")

            with open(neg_embeds_path, "wb") as f:
                f.write(read_from_url("https://github.com/ByteDance-Seed/SeedVR/raw/refs/heads/main/neg_emb.pt"))

            with open(pos_embeds_path, "wb") as f:
                f.write(read_from_url("https://github.com/ByteDance-Seed/SeedVR/raw/refs/heads/main/pos_emb.pt"))

            positive = torch.load(pos_embeds_path)
            negative = torch.load(neg_embeds_path)
            embeddings = cls(positive, negative)
            return embeddings.to(device=device, dtype=dtype)