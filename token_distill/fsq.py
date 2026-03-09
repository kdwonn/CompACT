import logging
from vector_quantize_pytorch import FSQ
import torch

logger = logging.getLogger(__name__)


class FSQWrapper(FSQ):
    def __init__(self, *args, **kwargs):
        _ = kwargs.pop("codebook_size", None)
        super().__init__(*args, **kwargs)
        logger.info(f"FSQWrapper: codebook_size = {self.codebook_size}")

    def forward(self, x):
        out, indices = super().forward(x)
        return out, indices, torch.zeros(1).to(x.device)
