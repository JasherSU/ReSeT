from __future__ import annotations

from typing import Dict, List, Sequence

from .state_cache import E27StateCache, box_area_sqrt, box_center, center_to_box


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


class E27CandidateGenerator:
    def __init__(self, bank_topk: int = 3, use_offsets: bool = True,
                 offset_topk: int = 1, max_candidates: int = 10,
                 bank_ranker: str = "legacy",
                 local_neighborhood_mode: str = "legacy"):
        self.bank_topk = int(bank_topk)
        self.use_offsets = bool(use_offsets)
        self.offset_topk = int(offset_topk)
        self.max_candidates = int(max_candidates)
        self.bank_ranker = str(bank_ranker or "legacy")
        self.local_neighborhood_mode = str(local_neighborhood_mode or "legacy")

    def _add(self, out, source, box, score, center=None, anchor_frame=-1):
        if center is None:
            center = [box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]]
        for item in out:
            item_center = [item["box"][0] + 0.5 * item["box"][2], item["box"][1] + 0.5 * item["box"][3]]
            item_size = box_area_sqrt(item["box"])
            ref_size = box_area_sqrt(box)
            if _dist(center, item_center) < 0.25 * max(ref_size, item_size, 1.0):
                if float(score) > float(item.get("bank_score", 0.0)):
                    item.update({
                        "source": source,
                        "center": center,
                        "box": [float(v) for v in box],
                        "bank_score": float(score),
                        "anchor_frame": int(anchor_frame),
                    })
                return
        out.append({
            "source": source,
            "center": center,
            "box": [float(v) for v in box],
            "bank_score": float(score),
            "anchor_frame": int(anchor_frame),
        })

    def build(self, ha_box: Sequence[float], cache: E27StateCache,
              current_frame: int = 0) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        self._add(out, "C0_current_HA", ha_box, 0.0, anchor_frame=current_frame)
        if cache.last_reliable is not None:
            self._add(out, "C1_last_reliable", cache.last_reliable.box, cache.last_reliable.raw_score,
                      anchor_frame=cache.last_reliable.frame_id)
        if cache.last_high_conf is not None:
            self._add(out, "C3_last_high_conf", cache.last_high_conf.box, cache.last_high_conf.raw_score,
                      anchor_frame=cache.last_high_conf.frame_id)
        bank = cache.top_bank(self.bank_topk, ranker=self.bank_ranker, current_frame=current_frame)
        for i, st in enumerate(bank):
            score = st.reliability_score if self.bank_ranker == "reliability" else st.raw_score
            self._add(out, f"C4_bank_top{i + 1}", st.box, score, anchor_frame=st.frame_id)
        if self.use_offsets:
            for i, st in enumerate(bank[:self.offset_topk]):
                side = box_area_sqrt(st.box) * 5.0
                d = 0.5 * side
                cx, cy = st.center
                if self.local_neighborhood_mode == "local_lr":
                    directions = [("right", (cx + d, cy)), ("left", (cx - d, cy))]
                else:
                    directions = [
                        ("right", (cx + d, cy)),
                        ("left", (cx - d, cy)),
                        ("down", (cx, cy + d)),
                        ("up", (cx, cy - d)),
                    ]
                for name, center in directions:
                    score = st.reliability_score if self.bank_ranker == "reliability" else st.raw_score
                    self._add(out, f"C4_offset_top{i + 1}_{name}", center_to_box(center, st.box), score,
                              anchor_frame=st.frame_id)
        return out[:self.max_candidates]
