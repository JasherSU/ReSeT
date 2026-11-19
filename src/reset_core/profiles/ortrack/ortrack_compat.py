"""Compatibility helpers for the pinned public ORTrack box head."""

from __future__ import annotations


def cal_bbox_topk(box_head, score_map_ctr, size_map, offset_map, k=5):
    """Decode top-k boxes with either the extended or public ORTrack head.

    Newer ORTrack worktrees provide ``CenterPredictor.cal_bbox_topk``. The
    pinned public revision provides only ``cal_bbox``; the fallback below uses
    the same tensor operations as the extended method.
    """
    native = getattr(box_head, "cal_bbox_topk", None)
    if native is not None:
        return native(score_map_ctr, size_map, offset_map, k=k)

    import torch

    flattened = score_map_ctr.flatten(1)
    k = min(int(k), int(flattened.shape[1]))
    topk_scores, topk_idx = torch.topk(flattened, k, dim=1)
    idx_y = topk_idx // box_head.feat_sz
    idx_x = topk_idx % box_head.feat_sz
    gather_idx = topk_idx.unsqueeze(1).expand(-1, 2, -1)
    size = size_map.flatten(2).gather(dim=2, index=gather_idx)
    offset = offset_map.flatten(2).gather(dim=2, index=gather_idx)
    cx = (idx_x.to(torch.float) + offset[:, 0, :]) / box_head.feat_sz
    cy = (idx_y.to(torch.float) + offset[:, 1, :]) / box_head.feat_sz
    boxes = torch.stack([cx, cy, size[:, 0, :], size[:, 1, :]], dim=-1)
    return boxes, topk_scores
