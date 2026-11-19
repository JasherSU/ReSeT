from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def box_center(box: Sequence[float]):
    return float(box[0]) + 0.5 * float(box[2]), float(box[1]) + 0.5 * float(box[3])


def size_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    area_a = max(float(a[2]) * float(a[3]), 1e-6)
    area_b = max(float(b[2]) * float(b[3]), 1e-6)
    return area_a / area_b


def aspect_ratio_delta(a: Sequence[float], b: Sequence[float]) -> float:
    ar_a = float(a[2]) / max(float(a[3]), 1e-6)
    ar_b = float(b[2]) / max(float(b[3]), 1e-6)
    return ar_a / max(ar_b, 1e-6)


class E27LocalVerifier:
    def __init__(self, delta_score: float = 0.05, margin_thresh: float = 0.08,
                 edge_thresh: float = 0.85, strict_delta_score: float = 0.08,
                 strict_margin_thresh: float = 0.12,
                 bank23_delta_score: Optional[float] = None,
                 bank23_margin_thresh: Optional[float] = None,
                 c1_fallback_only: bool = False,
                 reset_confidence: bool = False,
                 reset_confidence_thresh: float = 0.12,
                 verifier_mode: str = "full",
                 include_sources: Optional[Sequence[str]] = None,
                 exclude_sources: Optional[Sequence[str]] = None):
        self.delta_score = float(delta_score)
        self.margin_thresh = float(margin_thresh)
        self.edge_thresh = float(edge_thresh)
        self.strict_delta_score = float(strict_delta_score)
        self.strict_margin_thresh = float(strict_margin_thresh)
        self.bank23_delta_score = None if bank23_delta_score is None else float(bank23_delta_score)
        self.bank23_margin_thresh = None if bank23_margin_thresh is None else float(bank23_margin_thresh)
        self.c1_fallback_only = bool(c1_fallback_only)
        self.reset_confidence = bool(reset_confidence)
        self.reset_confidence_thresh = float(reset_confidence_thresh)
        self.verifier_mode = str(verifier_mode or "full")
        self.include_sources = tuple(s for s in (include_sources or []) if s)
        self.exclude_sources = tuple(s for s in (exclude_sources or []) if s)

    def choose(self, candidates: List[Dict[str, object]], ha_score: float, raw_apce: float,
               last_reliable_box: Optional[Sequence[float]], trigger_type: str,
               crop_side: float,
               exp13_delta_override: float = None,
               exp13_margin_override: float = None) -> Optional[Dict[str, object]]:
        accepted = []
        strict = trigger_type in {"high_conf_risk", "high_conf_risk_safe"}
        if exp13_delta_override is not None:
            delta = exp13_delta_override
        else:
            delta = self.strict_delta_score if strict else self.delta_score
        if exp13_margin_override is not None:
            margin_thr = exp13_margin_override
        else:
            margin_thr = self.strict_margin_thresh if strict else self.margin_thresh
        for cand in candidates:
            source = str(cand.get("source", ""))
            if source == "C0_current_HA":
                continue
            if self.c1_fallback_only and source.startswith("C1_last_reliable"):
                continue
            if self.include_sources and not any(source.startswith(prefix) for prefix in self.include_sources):
                continue
            if any(source.startswith(prefix) for prefix in self.exclude_sources):
                continue
            cand_delta = delta
            cand_margin_thr = margin_thr
            if source.startswith("C4_bank_top2") or source.startswith("C4_bank_top3"):
                if self.bank23_delta_score is not None:
                    cand_delta = max(cand_delta, self.bank23_delta_score)
                if self.bank23_margin_thresh is not None:
                    cand_margin_thr = max(cand_margin_thr, self.bank23_margin_thresh)
            score = float(cand.get("score", 0.0))
            margin = float(cand.get("margin", 0.0))
            apce = float(cand.get("apce", 0.0))
            edge = float(cand.get("edge_risk", 1.0))
            if score <= float(ha_score) + cand_delta:
                continue
            use_sharpness = self.verifier_mode in {"score_sharpness", "score_sharpness_edge", "full"}
            use_edge = self.verifier_mode in {"score_sharpness_edge", "full"}
            use_geometry = self.verifier_mode == "full"
            if use_sharpness and apce <= float(raw_apce):
                continue
            if use_sharpness and margin <= cand_margin_thr:
                continue
            if use_edge and edge > self.edge_thresh:
                continue
            reset_confidence = (
                2.0 * max(score - float(ha_score), 0.0)
                + 0.20 * max(min(apce / max(float(raw_apce), 1e-6) - 1.0, 1.0), -1.0)
                + 0.20 * min(max(margin / 0.20, 0.0), 1.0)
                - 0.10 * min(max(edge, 0.0), 1.0)
            )
            cand["reset_confidence"] = float(reset_confidence)
            if self.reset_confidence and reset_confidence < self.reset_confidence_thresh:
                continue
            if use_geometry and last_reliable_box is not None:
                sr = size_ratio(cand["bbox"], last_reliable_box)
                ar = aspect_ratio_delta(cand["bbox"], last_reliable_box)
                if not (0.5 <= sr <= 2.0):
                    continue
                if not (0.33 <= ar <= 3.0):
                    continue
                if strict:
                    cx, cy = box_center(cand["bbox"])
                    hx, hy = box_center(candidates[0]["bbox"])
                    dist = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5
                    if dist > float(crop_side) and score <= float(ha_score) + 0.10:
                        continue
            accepted.append(cand)
        if not accepted:
            return None
        accepted.sort(key=lambda x: (
            float(x.get("reset_confidence", 0.0)),
            float(x.get("score", 0.0)),
            float(x.get("margin", 0.0)),
            float(x.get("apce", 0.0)),
        ), reverse=True)
        return accepted[0]
