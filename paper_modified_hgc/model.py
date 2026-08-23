"""Independent paper-modified 3D-FPN HGC model and prediction heads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ThreeDFPNEncoder
from .pose import ORIENTATION_BINS


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = -1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Torch-only equivalent of the reference torchvision focal objective."""

    if logits.shape != targets.shape:
        raise ValueError(f"focal logits/targets must match, got {logits.shape} and {targets.shape}")
    targets = targets.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = cross_entropy * (1.0 - p_t).pow(float(gamma))
    if alpha >= 0.0:
        alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


class MLPBlock(nn.Module):
    """Reference paper-modified two-layer point MLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dims), nn.Linear(hidden_dims, out_dim)]
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        output = self.activation(self.layers[0](feature))
        return self.activation(self.layers[1](output))


@dataclass
class PaperModifiedPrediction:
    """Arm-specific output boundary used by trainer and server.

    ``quality_logits`` is dense over the complete 64^3 volume.  The remaining
    tensors are evaluated at the selected feature rows supplied to
    :meth:`PaperModifiedHGC.forward_batch`.
    """

    quality_logits: torch.Tensor
    orientation_logits: torch.Tensor
    residual: torch.Tensor
    q_contact: torch.Tensor

    def __iter__(self):
        # A small convenience for callers that prefer tuple unpacking while
        # retaining named fields for the paper-specific runtime.
        yield self.quality_logits
        yield self.orientation_logits
        yield self.residual
        yield self.q_contact

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.quality_logits, self.orientation_logits, self.residual, self.q_contact)[index]


class PaperModifiedHGCHead(nn.Module):
    """Dense quality plus orientation/residual/contact heads from the reference."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.grasp_qual_head = nn.Sequential(
            nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.feature_dim, 1, kernel_size=1),
        )
        self.grasp_bin_pred_head = MLPBlock(
            in_dim=self.feature_dim,
            out_dim=ORIENTATION_BINS,
            hidden_dims=self.feature_dim,
        )
        self.grasp_reg_head = MLPBlock(
            in_dim=self.feature_dim,
            out_dim=4,
            hidden_dims=self.feature_dim,
        )
        self.q_contact_head = MLPBlock(
            in_dim=self.feature_dim,
            out_dim=12,
            hidden_dims=self.feature_dim,
        )

    def forward_dense(self, feature_grid: torch.Tensor) -> torch.Tensor:
        if feature_grid.ndim != 5 or feature_grid.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected feature grid Bx{self.feature_dim}xDxHxW, got {tuple(feature_grid.shape)}"
            )
        return self.grasp_qual_head(feature_grid)

    def forward_selected(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict orientation, four-vector residual and learned q_contact."""

        if feature.ndim != 3 or feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected selected feature BxNx{self.feature_dim}, got {tuple(feature.shape)}"
            )
        return (
            self.grasp_bin_pred_head(feature),
            self.grasp_reg_head(feature),
            self.q_contact_head(feature),
        )

    def forward(
        self, feature_grid: torch.Tensor, selected_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quality = self.forward_dense(feature_grid)
        orientation, residual, q_contact = self.forward_selected(selected_feature)
        return quality, orientation, residual, q_contact

    @staticmethod
    def _unpack_prediction(
        prediction: PaperModifiedPrediction | tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prediction, PaperModifiedPrediction):
            return (
                prediction.quality_logits,
                prediction.orientation_logits,
                prediction.residual,
                prediction.q_contact,
            )
        if len(prediction) != 4:
            raise ValueError("paper-modified prediction must contain four tensors")
        return tuple(prediction)  # type: ignore[return-value]

    def loss(
        self,
        prediction: PaperModifiedPrediction | tuple[torch.Tensor, ...],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the reference dense focal + pose + contact objective."""

        quality_logits, orientation_logits, residual, q_contact_pred = self._unpack_prediction(prediction)
        quality_target = batch.get("quality_target")
        if quality_target is None:
            raise KeyError("paper-modified batch must contain quality_target")
        if quality_logits.shape != quality_target.shape:
            raise ValueError(
                f"quality logits/target shape mismatch: {quality_logits.shape} vs {quality_target.shape}"
            )
        seg_loss = sigmoid_focal_loss(quality_logits, quality_target)

        positive_mask = batch.get("positive_mask")
        if positive_mask is None:
            labels = batch.get("template_graspable")
            if labels is None:
                raise KeyError("paper-modified batch needs positive_mask or template_graspable")
            positive_mask = labels.max(dim=-1).values > 0
        positive_mask = positive_mask.bool()
        if positive_mask.shape != orientation_logits.shape[:2]:
            raise ValueError(
                f"positive_mask shape {positive_mask.shape} does not match selected rows {orientation_logits.shape[:2]}"
            )
        if not torch.any(positive_mask):
            raise ValueError("paper-modified batch has no positive pose rows")

        orientation_target = batch["orientation_bin"].long()
        pose_target = batch["pose_target"].to(dtype=residual.dtype)
        if orientation_target.shape != orientation_logits.shape[:2] or pose_target.shape != (*orientation_logits.shape[:2], 4):
            raise ValueError("paper pose target shapes do not match selected rows")
        cls_loss = F.cross_entropy(
            orientation_logits[positive_mask], orientation_target[positive_mask]
        )
        reg_loss = F.smooth_l1_loss(residual[positive_mask], pose_target[positive_mask])

        q_contact = batch["q_contact"].to(dtype=q_contact_pred.dtype)
        if q_contact.shape != q_contact_pred.shape:
            raise ValueError(
                f"q_contact target shape {q_contact.shape} does not match output {q_contact_pred.shape}"
            )
        contact_loss = F.smooth_l1_loss(q_contact_pred[positive_mask], q_contact[positive_mask])
        total_loss = 5.0 * seg_loss + cls_loss + reg_loss + contact_loss
        return {
            "seg_loss": seg_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "contact_loss": contact_loss,
            "finger_loss": contact_loss,
            "total_loss": total_loss,
        }


class PaperModifiedHGC(nn.Module):
    """Full-resolution voxel HGC with an independent paper-modified head."""

    paper_modified = True

    def __init__(
        self,
        *,
        in_channels: int = 2,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        encoder_scales: Sequence[int] = (2, 2, 2, 4),
        encoder_out_channels: int = 128,
        align_corners: bool = True,
        num_templates: int | None = None,
    ) -> None:
        super().__init__()
        if num_templates not in (None, 3):
            raise ValueError("paper-modified HGC has no template axis; num_templates must be omitted or 3")
        self.encoder = ThreeDFPNEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            scales=encoder_scales,
            out_channels=encoder_out_channels,
            align_corners=align_corners,
        )
        self.head = PaperModifiedHGCHead(feature_dim=int(encoder_out_channels))

    def encode(self, input_grid: torch.Tensor) -> torch.Tensor:
        """Return the complete ``B,C,64,64,64`` FPN feature grid."""

        return self.encoder(input_grid.float())

    def _select(self, feature_grid: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.encoder.select(feature_grid, indices)

    def forward_grid(self, input_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_grid = self.encode(input_grid)
        return feature_grid, self.head.forward_dense(feature_grid)

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> PaperModifiedPrediction:
        if "input_grid" not in batch:
            raise KeyError("paper-modified batch must contain input_grid")
        if "feature_indices" not in batch:
            raise KeyError("paper-modified batch must contain feature_indices")
        feature_grid = self.encode(batch["input_grid"])
        selected = self._select(feature_grid, batch["feature_indices"])
        quality, orientation, residual, q_contact = self.head(feature_grid, selected)
        return PaperModifiedPrediction(quality, orientation, residual, q_contact)

    def forward(
        self,
        input_grid: torch.Tensor | dict[str, torch.Tensor],
        feature_indices: torch.Tensor | None = None,
    ) -> PaperModifiedPrediction:
        if isinstance(input_grid, dict):
            if feature_indices is not None:
                raise ValueError("feature_indices must be omitted with a batch dictionary")
            return self.forward_batch(input_grid)
        if feature_indices is None:
            raise ValueError("feature_indices is required for tensor forward")
        return self.forward_batch({"input_grid": input_grid, "feature_indices": feature_indices})


__all__ = [
    "MLPBlock",
    "PaperModifiedHGC",
    "PaperModifiedHGCHead",
    "PaperModifiedPrediction",
    "sigmoid_focal_loss",
]
