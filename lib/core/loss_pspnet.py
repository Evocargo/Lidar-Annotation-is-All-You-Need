import torch.nn as nn
import torch

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg

        _BCEcls, _BCEobj, BCEseg = self.losses

        # Calculate Losses
        drive_area_seg_predicts = predictions.view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        if self.cfg.LOSS.MASKED:
            mask = targets[2].view(-1).clone() # mask of points
            bool_mask = torch.gt(mask, 0)
            drive_area_seg_predicts = drive_area_seg_predicts[bool_mask]
            drive_area_seg_targets = drive_area_seg_targets[bool_mask]
        
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        loss = lseg_da
        return loss, lseg_da.item()


def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss
