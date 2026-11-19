ACTION_NORMAL = "normal"
ACTION_RECOVERY = "recovery"
ACTION_SEVERE = "severe"


def action_params(action, base_sf=4.0, recovery_sf=5.0, severe_sf=6.0):
    if action == ACTION_SEVERE:
        return severe_sf, False
    if action == ACTION_RECOVERY:
        return recovery_sf, False
    return base_sf, True


def choose_score_severity_action(prev_score, severe_thresh=0.20, recovery_thresh=0.35):
    if prev_score < severe_thresh:
        return ACTION_SEVERE
    if prev_score < recovery_thresh:
        return ACTION_RECOVERY
    return ACTION_NORMAL


def choose_size_conditioned_action(prev_score, prev_bbox_area, size_sf_thresh=60.0,
                                   recovery_thresh=0.35, base_sf=4.0, recovery_sf=5.0,
                                   large_sf=6.0):
    if prev_score >= recovery_thresh:
        return ACTION_NORMAL, base_sf
    sqrt_area = prev_bbox_area ** 0.5
    if sqrt_area >= size_sf_thresh:
        return ACTION_RECOVERY, large_sf
    return ACTION_RECOVERY, recovery_sf


def choose_score_size_hybrid_action(prev_score, prev_bbox_area, severe_thresh=0.20,
                                    size_sf_thresh=60.0, recovery_thresh=0.35,
                                    base_sf=4.0, recovery_sf=5.0, severe_sf=6.0):
    if prev_score < severe_thresh:
        sqrt_area = prev_bbox_area ** 0.5
        if sqrt_area >= size_sf_thresh:
            return ACTION_SEVERE, severe_sf, False
        return ACTION_RECOVERY, recovery_sf, False
    if prev_score < recovery_thresh:
        return ACTION_RECOVERY, recovery_sf, False
    return ACTION_NORMAL, base_sf, True


def choose_boundary_severe_action(prev_score, p_recovery, tau_action,
                                  severe_thresh=0.20, severe_sf=6.0,
                                  recovery_sf=5.0, base_sf=4.0):
    if prev_score < severe_thresh:
        return ACTION_SEVERE, severe_sf, False
    if prev_score < 0.35:
        return ACTION_RECOVERY, recovery_sf, False
    if prev_score >= 0.50:
        return ACTION_NORMAL, base_sf, True
    if p_recovery > tau_action:
        return ACTION_RECOVERY, recovery_sf, False
    return ACTION_NORMAL, base_sf, True


def choose_action_from_score_head(prev_score, p_recovery, tau_action):
    if prev_score < 0.35:
        return ACTION_RECOVERY
    if prev_score >= 0.50:
        return ACTION_NORMAL
    if p_recovery > tau_action:
        return ACTION_RECOVERY
    return ACTION_NORMAL


def choose_edge_proxy_action(prev_score, edge_risk, tau_edge,
                             score_low=0.35, score_high=0.50):
    if prev_score < score_low:
        return ACTION_RECOVERY
    if prev_score >= score_high:
        return ACTION_NORMAL
    if edge_risk >= tau_edge:
        return ACTION_RECOVERY
    return ACTION_NORMAL


def edge_risk_from_score_map(score_map):
    flat_idx = score_map.view(-1).argmax().item()
    feat_sz = score_map.shape[-1]
    peak_y = flat_idx // feat_sz
    peak_x = flat_idx % feat_sz
    center = (feat_sz - 1) / 2.0
    denom = max(center, 1.0)
    return max(abs(peak_x - center) / denom, abs(peak_y - center) / denom)


def bbox_iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_visible(box):
    return len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0


def region_oracle_action(gt_bbox, search_state, search_size, search_factor, edge_thresh=0.75):
    if not bbox_visible(gt_bbox):
        return ACTION_RECOVERY

    sx, sy, sw, sh = [float(v) for v in search_state]
    gx, gy, gw, gh = [float(v) for v in gt_bbox]
    cx = sx + 0.5 * sw
    cy = sy + 0.5 * sh
    gt_cx = gx + 0.5 * gw
    gt_cy = gy + 0.5 * gh
    crop_half = 0.5 * float(search_size) / float(search_factor)
    if crop_half <= 0:
        return ACTION_NORMAL

    dx = abs(gt_cx - cx) / crop_half
    dy = abs(gt_cy - cy) / crop_half
    if max(dx, dy) >= edge_thresh:
        return ACTION_RECOVERY
    return ACTION_NORMAL
