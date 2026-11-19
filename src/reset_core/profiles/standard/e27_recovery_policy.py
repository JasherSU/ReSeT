"""Observation gating and deterministic routing policy for ReSeT.

Implements the observation gate and bounded routing decisions described in the
main-paper Method section and Supplement Algorithm S1.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Sequence, Tuple

import torch


def apce_from_score_map(score_map: torch.Tensor) -> float:
    sm = score_map.detach().float()
    mn = sm.min()
    mx = sm.max()
    denom = torch.mean((sm - mn) ** 2).clamp_min(1e-12)
    return float(((mx - mn) ** 2 / denom).item())


def top2_margin(score_map: torch.Tensor) -> float:
    flat = score_map.detach().flatten()
    if flat.numel() < 2:
        return 0.0
    vals = torch.topk(flat, 2).values
    return float((vals[0] - vals[1]).item())


def score_map_stats(score_map: torch.Tensor, feat_sz: int) -> Dict[str, float]:
    flat = score_map.detach().flatten()
    top = torch.topk(flat, min(2, flat.numel())).values
    top1 = float(top[0].item()) if top.numel() else 0.0
    top2 = float(top[1].item()) if top.numel() > 1 else top1
    idx = int(torch.argmax(flat).item()) if flat.numel() else 0
    row = idx // max(int(feat_sz), 1)
    col = idx % max(int(feat_sz), 1)
    center = (float(feat_sz) - 1.0) / 2.0
    denom = max(center, 1.0)
    edge_risk = max(abs(float(row) - center), abs(float(col) - center)) / denom
    return {
        "score": top1,
        "top2": top2,
        "margin": top1 - top2,
        "apce": apce_from_score_map(score_map),
        "edge_risk": float(edge_risk),
        "peak_row": int(row),
        "peak_col": int(col),
    }


class E27RecoveryPolicy:
    def __init__(self, mode: str = "C1", apce_low: float = 80.0,
                 margin_low: float = 0.08, gap_thresh: float = 32.0,
                 displacement_thresh: float = 48.0, long_unreliable: int = 10,
                 safe_min_frames_since_reliable: int = 5, cooldown: int = 0,
                 trigger_policy: str = "legacy",
                 dynamic_history: int = 80,
                 dynamic_min_frames: int = 20,
                 score_low_q: float = 0.20,
                 score_gray_q: float = 0.35,
                 apce_low_q: float = 0.25,
                 relative_gray_min_frames: int = 0,
                 relative_high_score_protect: float = 1.0,
                 calibration_engine=None):
        self.mode = mode
        self.apce_low = float(apce_low)
        self.margin_low = float(margin_low)
        self.gap_thresh = float(gap_thresh)
        self.displacement_thresh = float(displacement_thresh)
        self.long_unreliable = int(long_unreliable)
        self.safe_min_frames_since_reliable = int(safe_min_frames_since_reliable)
        self.cooldown = int(cooldown)
        self.trigger_policy = str(trigger_policy or "legacy")
        self.dynamic_history = int(dynamic_history)
        self.dynamic_min_frames = int(dynamic_min_frames)
        self.score_low_q = float(score_low_q)
        self.score_gray_q = float(score_gray_q)
        self.apce_low_q = float(apce_low_q)
        self.relative_gray_min_frames = int(relative_gray_min_frames)
        self.relative_high_score_protect = float(relative_high_score_protect)
        self.score_hist: Deque[float] = deque(maxlen=max(self.dynamic_history, 1))
        self.apce_hist: Deque[float] = deque(maxlen=max(self.dynamic_history, 1))
        self.margin_hist: Deque[float] = deque(maxlen=max(self.dynamic_history, 1))
        # ReSeT-Cal calibration engine
        self.calibration_engine = calibration_engine

    def reset(self) -> None:
        self.score_hist.clear()
        self.apce_hist.clear()
        self.margin_hist.clear()
        if self.calibration_engine is not None:
            self.calibration_engine.reset()

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        vals = sorted(float(v) for v in values)
        if not vals:
            return 0.0
        q = min(max(float(q), 0.0), 1.0)
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    @staticmethod
    def _norm_quantile(value: float, values: Sequence[float], low_q: float = 0.10,
                       high_q: float = 0.90, default: float = 0.5) -> float:
        if len(values) < 2:
            return float(default)
        lo = E27RecoveryPolicy._quantile(values, low_q)
        hi = E27RecoveryPolicy._quantile(values, high_q)
        if hi <= lo + 1e-6:
            return float(default)
        return min(max((float(value) - lo) / (hi - lo), 0.0), 1.0)

    def dynamic_thresholds(self) -> Tuple[float, float, float]:
        if len(self.score_hist) < self.dynamic_min_frames or len(self.apce_hist) < self.dynamic_min_frames:
            return 0.35, 0.40, self.apce_low
        score_low = self._quantile(self.score_hist, self.score_low_q)
        score_gray = self._quantile(self.score_hist, self.score_gray_q)
        if score_gray <= score_low:
            score_gray = score_low + 1e-6
        apce_low = self._quantile(self.apce_hist, self.apce_low_q)
        return float(score_low), float(score_gray), float(apce_low)

    def reliability_score(self, score: float, apce: float, margin: float,
                          edge_risk: float, raw_ha_gap: float,
                          age: int = 0,
                          exp13_norm_cues: dict = None) -> float:
        if exp13_norm_cues is not None:
            score_n = exp13_norm_cues.get('score', 0.5) or 0.5
            apce_n = exp13_norm_cues.get('apce', 0.5) or 0.5
            margin_n = exp13_norm_cues.get('margin', 0.5) or 0.5
        else:
            score_n = self._norm_quantile(score, self.score_hist, default=min(max(float(score), 0.0), 1.0))
            apce_n = self._norm_quantile(apce, self.apce_hist, default=0.5)
            margin_n = self._norm_quantile(margin, self.margin_hist, default=0.5)
        consistency = 1.0 / (1.0 + max(float(raw_ha_gap), 0.0) / 48.0)
        age_penalty = min(max(float(age), 0.0) / 80.0, 1.0)
        rel = (
            0.35 * score_n
            + 0.20 * apce_n
            + 0.15 * margin_n
            + 0.15 * consistency
            - 0.10 * min(max(float(edge_risk), 0.0), 1.0)
            - 0.05 * age_penalty
        )
        return float(rel)

    def update_reliable_stats(self, score: float, apce: float, margin: float) -> None:
        self.score_hist.append(float(score))
        self.apce_hist.append(float(apce))
        self.margin_hist.append(float(margin))

    def classify_trigger(self, raw_score: float, apce: float, margin: float,
                         displacement: float, raw_ha_gap: float,
                         frames_since_last_reliable: int,
                         low_score_streak: int,
                         cooldown_left: int,
                         exp13_norm_cues: dict = None,
                         exp13_norm_thresholds: tuple = None) -> str:
        if self.mode == "C0":
            allow_high = False
        else:
            allow_high = self.mode in {"C2", "C2safe", "C2-safe"}

        if exp13_norm_cues is not None and exp13_norm_thresholds is not None:
            norm_score = exp13_norm_cues.get('score')
            norm_apce = exp13_norm_cues.get('apce')
            if norm_score is not None and norm_apce is not None:
                ns_low, ns_gray, na_low = exp13_norm_thresholds
                if norm_score < ns_low or (norm_score < ns_gray and norm_apce <= na_low) or low_score_streak >= 2:
                    return "hard_lost"
                if frames_since_last_reliable >= self.long_unreliable:
                    return "long_unreliable"
                return "normal"

        # Get calibration thresholds if engine is present
        cal_thr = None
        if self.calibration_engine is not None:
            cal_thr = self.calibration_engine.get_thresholds()

        # BARS: If burst detected and in cooldown, suppress all triggers
        if cal_thr is not None and cal_thr.get("bars_suppress", False):
            return "normal"

        if self.trigger_policy == "score_apce_gate":
            if cal_thr is not None:
                # Optional APCE co-anomaly gate requires score corroboration.
                if cal_thr.get("apce_co_anomaly", False):
                    if raw_score < cal_thr["score_thr_hard"] or (raw_score < cal_thr["score_thr_uncertain"] and apce <= cal_thr["apce_thr"]) or low_score_streak >= 2:
                        return "hard_lost"
                else:
                    if raw_score < cal_thr["score_thr_hard"] or apce <= cal_thr["apce_thr"] or low_score_streak >= 2:
                        return "hard_lost"
            else:
                if raw_score < 0.35 or (raw_score < 0.40 and apce <= self.apce_low) or low_score_streak >= 2:
                    return "hard_lost"
        elif self.trigger_policy == "relative_score_apce_gate":
            if cal_thr is not None:
                # Optional APCE co-anomaly gate requires score corroboration.
                if cal_thr.get("apce_co_anomaly", False):
                    if raw_score < cal_thr["score_thr_hard"] or (raw_score < cal_thr["score_thr_uncertain"] and apce <= cal_thr["apce_thr"]) or low_score_streak >= 2:
                        return "hard_lost"
                else:
                    if raw_score < cal_thr["score_thr_hard"] or apce <= cal_thr["apce_thr"] or low_score_streak >= 2:
                        return "hard_lost"
            else:
                score_low, score_gray, apce_low = self.dynamic_thresholds()
            gray_ok = (
                raw_score < score_gray
                and apce <= apce_low
                and frames_since_last_reliable >= self.relative_gray_min_frames
            )
            high_score_protected = raw_score >= self.relative_high_score_protect
            if (not high_score_protected) and (raw_score < score_low or gray_ok or low_score_streak >= 2):
                return "hard_lost"
        else:
            # legacy trigger policy
            if cal_thr is not None:
                # Optional APCE co-anomaly gate requires score corroboration.
                if cal_thr.get("apce_co_anomaly", False):
                    if (raw_score < cal_thr["score_thr_hard"]
                            or (raw_score < cal_thr["score_thr_uncertain"] and apce <= cal_thr["apce_thr"])
                            or low_score_streak >= 2):
                        return "hard_lost"
                else:
                    if (raw_score < cal_thr["score_thr_hard"]
                            or apce <= cal_thr["apce_thr"]
                            or low_score_streak >= 2):
                        return "hard_lost"
            else:
                if raw_score < 0.35 or apce <= self.apce_low or low_score_streak >= 2:
                    return "hard_lost"
        long_unreliable_thr = self.long_unreliable
        if cal_thr is not None and "long_unreliable_threshold" in cal_thr:
            long_unreliable_thr = cal_thr["long_unreliable_threshold"]
        if frames_since_last_reliable >= long_unreliable_thr:
            return "long_unreliable"
        if allow_high and 0.50 <= raw_score <= 0.65:
            weak = apce <= self.apce_low or margin <= self.margin_low
            abnormal = displacement >= self.displacement_thresh or raw_ha_gap >= self.gap_thresh
            if weak and abnormal:
                if self.mode in {"C2safe", "C2-safe"} and frames_since_last_reliable < self.safe_min_frames_since_reliable:
                    return "normal"
                if cooldown_left > 0 and raw_score >= 0.20:
                    return "normal"
                return "high_conf_risk_safe" if self.mode in {"C2safe", "C2-safe"} else "high_conf_risk"
        return "normal"
