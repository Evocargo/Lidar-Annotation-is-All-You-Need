import torch.nn as nn
import torch

class MaskedLoss(nn.Module):
    """
    Collect all the loss we need
    """
    def __init__(self, loss, cfg, lambda_val=1.0):
        """
        Inputs:
            losses: (list)[nn.Module, nn.Module, ...]
            cfg: config
            lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        
        self.loss = loss
        self.lambda_val = lambda_val
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes):
        """
        Inputs:
            head_fields: (list) output from each task head
            head_targets: (list) ground-truth for each task head

        Returns:
            loss: masked loss

        """
        loss = self._forward_impl(head_fields, head_targets, shapes)

        return loss

    def _forward_impl(self, predictions, targets, shapes):
        """
        Args:
            predictions: predicts of drive_area_seg
            targets: gts segment_targets

        Returns:
            loss: masked loss value
        """
        cfg = self.cfg
        BCEseg = self.loss

        # Calculate Losses
        drive_area_seg_predicts = predictions.view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        if self.cfg.LOSS.MASKED:
            mask = targets[2].view(-1).clone() # mask of points
            bool_mask = torch.gt(mask, 0)
            drive_area_seg_predicts = drive_area_seg_predicts[bool_mask]
            drive_area_seg_targets = drive_area_seg_targets[bool_mask]
        
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)
        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambda_val
        return lseg_da


def get_loss(cfg, device):
    """
    Get loss

    Inputs:
        cfg: config
        device: cpu or gpu device

    Returns:
        loss: (MaskedLoss)

    """
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    loss = MaskedLoss(BCEseg, cfg=cfg, lambda_val=cfg.LOSS.LAMBDA)
    return loss
