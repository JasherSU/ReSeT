"""Reliable-state bank and anchor ranking for Bounded Support Routing.

Implements the bank state used by the main-paper Method section and the
initialization/ranking operations in Supplement Algorithm S1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return float(box[0]) + 0.5 * float(box[2]), float(box[1]) + 0.5 * float(box[3])


def box_area_sqrt(box: Sequence[float]) -> float:
    return max(float(box[2]) * float(box[3]), 1.0) ** 0.5


def center_to_box(center: Sequence[float], ref_box: Sequence[float]) -> List[float]:
    return [
        float(center[0]) - 0.5 * float(ref_box[2]),
        float(center[1]) - 0.5 * float(ref_box[3]),
        float(ref_box[2]),
        float(ref_box[3]),
    ]


@dataclass
class CachedState:
    frame_id: int
    box: List[float]
    raw_score: float
    ha_score: float
    apce: float
    margin: float
    source: str
    edge_risk: float = 0.0
    raw_ha_gap: float = 0.0
    reliability_score: float = 0.0
    r_write_score: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return box_center(self.box)

    @property
    def size(self) -> float:
        return box_area_sqrt(self.box)


class E27StateCache:
    def __init__(self, bank_size: int = 10):
        self.bank_size = bank_size
        self.bank: Deque[CachedState] = deque(maxlen=bank_size)
        self.last_reliable: Optional[CachedState] = None
        self.prev_reliable: Optional[CachedState] = None
        self.last_high_conf: Optional[CachedState] = None
        self.search_state: Optional[List[float]] = None
        self.prev_search_state: Optional[List[float]] = None
        self.pending_recovery: Optional[CachedState] = None
        self.pending_recovery_required = 1
        self.pending_recovery_good = 0
        self.pending_recovery_source = ""
        self.pending_recovery_rollback_box: Optional[List[float]] = None
        self.pending_recovery_start_frame = -1
        self.low_score_streak = 0
        self.cooldown_left = 0
        self.trigger_count = 0

    def reset(self, init_box: Sequence[float]) -> None:
        self.bank.clear()
        self.last_reliable = None
        self.prev_reliable = None
        self.last_high_conf = None
        self.search_state = list(init_box)
        self.prev_search_state = list(init_box)
        self.pending_recovery = None
        self.pending_recovery_required = 1
        self.pending_recovery_good = 0
        self.pending_recovery_source = ""
        self.pending_recovery_rollback_box = None
        self.pending_recovery_start_frame = -1
        self.low_score_streak = 0
        self.cooldown_left = 0
        self.trigger_count = 0

    def begin_frame(self) -> List[float]:
        return list(self.search_state) if self.search_state is not None else []

    def frames_since_last_reliable(self, frame_id: int) -> int:
        if self.last_reliable is None:
            return -1
        return int(frame_id) - int(self.last_reliable.frame_id)

    def make_state(self, frame_id: int, box: Sequence[float], raw_score: float,
                   ha_score: float, apce: float, margin: float, source: str,
                   edge_risk: float = 0.0, raw_ha_gap: float = 0.0,
                   reliability_score: float = 0.0) -> CachedState:
        return CachedState(
            frame_id=int(frame_id),
            box=[float(v) for v in box],
            raw_score=float(raw_score),
            ha_score=float(ha_score),
            apce=float(apce),
            margin=float(margin),
            source=source,
            edge_risk=float(edge_risk),
            raw_ha_gap=float(raw_ha_gap),
            reliability_score=float(reliability_score),
        )

    def update_reliable(self, state: CachedState, skip_bank_append: bool = False) -> None:
        self.prev_reliable = self.last_reliable
        self.last_reliable = state
        if not skip_bank_append:
            self.bank.append(state)
        if state.raw_score > 0.70 or state.ha_score > 0.70:
            self.last_high_conf = state

    def start_pending_recovery(self, state: CachedState, required_frames: int = 1,
                               rollback_box: Optional[Sequence[float]] = None) -> None:
        self.pending_recovery = state
        self.pending_recovery_required = max(1, int(required_frames))
        self.pending_recovery_good = 0
        self.pending_recovery_source = str(state.source)
        self.pending_recovery_rollback_box = (
            [float(v) for v in rollback_box] if rollback_box is not None else None
        )
        self.pending_recovery_start_frame = int(state.frame_id)

    def clear_pending_recovery(self) -> None:
        self.pending_recovery = None
        self.pending_recovery_required = 1
        self.pending_recovery_good = 0
        self.pending_recovery_source = ""
        self.pending_recovery_rollback_box = None
        self.pending_recovery_start_frame = -1

    def pending_recovery_active(self) -> bool:
        return self.pending_recovery is not None

    def update_high_conf(self, state: CachedState) -> None:
        self.last_high_conf = state

    def set_search_state(self, box: Sequence[float]) -> None:
        if self.search_state is not None:
            self.prev_search_state = list(self.search_state)
        self.search_state = [float(v) for v in box]

    def fallback_box(self, mode: str = "last_high_conf") -> List[float]:
        if mode == "bank_top1":
            bank = self.top_bank(1, ranker="reliability")
            if bank:
                return list(bank[0].box)
        if mode == "last_reliable" and self.last_reliable is not None:
            return list(self.last_reliable.box)
        if mode == "previous_search_state" and self.prev_search_state is not None:
            return list(self.prev_search_state)
        if self.last_high_conf is not None:
            return list(self.last_high_conf.box)
        if self.last_reliable is not None:
            return list(self.last_reliable.box)
        if self.prev_search_state is not None:
            return list(self.prev_search_state)
        return list(self.search_state) if self.search_state is not None else []

    def top_bank(self, k: int, ranker: str = "legacy", current_frame: Optional[int] = None) -> List[CachedState]:
        states = list(self.bank)
        if ranker == "reliability":
            if current_frame is None:
                states.sort(key=lambda s: (s.reliability_score, s.raw_score, s.margin, s.apce), reverse=True)
            else:
                states.sort(
                    key=lambda s: (
                        s.reliability_score - min(max(int(current_frame) - int(s.frame_id), 0), 80) * 0.0005,
                        s.raw_score,
                        s.margin,
                        s.apce,
                    ),
                    reverse=True)
        else:
            states.sort(key=lambda s: (s.raw_score, s.margin, s.apce), reverse=True)
        return states[:max(0, int(k))]

    def reliable_velocity_box(self) -> Optional[List[float]]:
        if self.last_reliable is None:
            return None
        if self.prev_reliable is None:
            return list(self.last_reliable.box)
        cx1, cy1 = self.prev_reliable.center
        cx2, cy2 = self.last_reliable.center
        pred_center = (cx2 + (cx2 - cx1), cy2 + (cy2 - cy1))
        return center_to_box(pred_center, self.last_reliable.box)
