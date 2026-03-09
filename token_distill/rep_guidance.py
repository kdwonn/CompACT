import os
from einops import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepGuidancePredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.predictor = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, out_dim),
        )

    def forward(self, x):
        return self.predictor(x)


class RepGuidanceLoss(nn.Module):
    def __init__(
        self, in_dim, repa_target_enc, use_mask=True, dinov3_weights_path=None
    ):
        super().__init__()

        # Check if this is a DINOv3 model
        if repa_target_enc.startswith("dinov3_"):
            # DINOv3 loading with local weights
            if dinov3_weights_path is None:
                raise ValueError(
                    f"dinov3_weights_path required for DINOv3 model: {repa_target_enc}"
                )
            # Get absolute path to dinov3 directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")

            self.repa_target_enc = torch.hub.load(
                repo_or_dir=dinov3_path,
                model=repa_target_enc,
                source="local",
                weights=dinov3_weights_path,
            )
        else:
            # DINOv2 loading (existing code)
            self.repa_target_enc = torch.hub.load(
                "facebookresearch/dinov2", repa_target_enc
            )

        for param in self.repa_target_enc.parameters():
            param.requires_grad = False

        self.repa_target_embed_dim = self.repa_target_enc.embed_dim
        self.predictor = RepGuidancePredictor(in_dim, self.repa_target_embed_dim)
        self.use_mask = use_mask

    def forward(self, pred, image, masks):
        with torch.no_grad():
            target = self.repa_target_enc.get_intermediate_layers(image, n=1)[0]
        pred = self.predictor(pred)
        loss = -F.cosine_similarity(pred, target.detach(), dim=-1)

        if not self.use_mask or masks is None:
            return loss.mean()
        else:
            masks = masks.to(pred.device)
            loss = (loss * masks).sum() / (masks.sum() + 1e-8)
            return loss


class RepSelfSimGuidanceLoss(RepGuidanceLoss):
    def forward(self, pred, image, masks):
        with torch.no_grad():
            target = self.repa_target_enc.get_intermediate_layers(image, n=1)[0]
        pred = self.predictor(pred)

        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        pred_self_sim = einsum(pred_norm, pred_norm, "b n1 d, b n2 d -> b n1 n2")
        target_self_sim = einsum(target_norm, target_norm, "b n1 d, b n2 d -> b n1 n2")

        loss = F.mse_loss(pred_self_sim, target_self_sim)

        return loss.mean()


class DummyRepGuidanceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.repa_target_enc = nn.Identity()

    def forward(self, pred, image, masks):
        return torch.zeros(1).to(pred.device)
