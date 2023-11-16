from typing import Any, List

import torch
import torch.nn as nn
from yacs.config import CfgNode


class MaskedLoss(nn.Module):
    """
    A loss module that applies a mask to the loss calculation, allowing for selective
    evaluation of the loss function over specified elements.
    """

    def __init__(self, loss: nn.Module, cfg: CfgNode, lambda_val: float = 1.0):
        """
        Initializes the MaskedLoss module.

        Args:
            loss: The loss function to be masked.
            cfg: Configuration object with various settings.
            lambda_val: A weighting factor for the loss (default is 1.0).
        """
        super().__init__()
        self.loss = loss
        self.lambda_val = lambda_val
        self.cfg = cfg

    def forward(self, head_fields: torch.Tensor, head_targets: List[torch.Tensor], shapes: Any) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            head_fields: A tensor containing the predictions from the model.
            head_targets: A list of tensors containing the ground-truth targets.
            shapes: The original shapes of the input images for scaling the loss if needed.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss = self._forward_impl(head_fields, head_targets, shapes)
        return loss

    def _forward_impl(self, predictions: torch.Tensor, targets: List[torch.Tensor], shapes: Any) -> torch.Tensor:
        """
        Implementation of the masked loss calculation.

        Args:
            predictions: The predictions from the segmentation head of the model.
            targets: The ground-truth segmentation targets.
            shapes: The original shapes of the input images for scaling the loss if needed.

        Returns:
            torch.Tensor: The masked loss value.
        """
        BCEseg = self.loss
        drive_area_seg_predicts = predictions.view(-1)
        drive_area_seg_targets = targets[1].view(-1)

        if self.cfg.LOSS.MASKED:
            mask = targets[2].view(-1).clone()  # mask of points
            bool_mask = torch.gt(mask, 0)
            drive_area_seg_predicts = drive_area_seg_predicts[bool_mask]
            drive_area_seg_targets = drive_area_seg_targets[bool_mask]

        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
        lseg_da *= self.cfg.LOSS.DA_SEG_GAIN * self.lambda_val
        return lseg_da


def get_loss(cfg: CfgNode, device: torch.device) -> MaskedLoss:
    """
    Constructs and returns the MaskedLoss module.

    Args:
        cfg: Configuration object with various settings, including the loss configuration.
        device: The device to which the loss module should be assigned.

    Returns:
        MaskedLoss: The constructed loss module with the specified configuration.
    """
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    loss = MaskedLoss(BCEseg, cfg=cfg, lambda_val=cfg.LOSS.LAMBDA)
    return loss
