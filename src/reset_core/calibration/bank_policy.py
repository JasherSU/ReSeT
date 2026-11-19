"""Optional bank admission, diversity, and support-safety policies.

B2: Separate Admission and Retrieval Utility
B3: Diversity-Preserving Bank
B4: Support-Aware Candidate Selection

All components are GT-free and operate on normalized cues where available.
No learned modules, no clustering, no large weight grids.
"""

from __future__ import annotations

import math
from typing import Optional


# Admission and retrieval utility.

class BankAdmissionRetrieval:
    """B2: Separate R_write (admission) and R_ret (retrieval) scores.

    R_write(x) = normalized response quality + geometry safety
    R_ret(a|t) = R_write + diversity - stale risk + reachability

    Uses grouped ablations rather than a large weight grid.
    """

    def __init__(
        self,
        write_components: tuple[str, ...] = ("score", "apce", "margin"),
        retrieval_components: tuple[str, ...] = ("write", "freshness", "diversity"),
        geometry_weight: float = 0.15,
    ):
        self.write_components = write_components
        self.retrieval_components = retrieval_components
        self.geometry_weight = geometry_weight

    def r_write(
        self,
        normalized_cues: dict[str, Optional[float]],
        geometry_valid: bool,
        raw_reliability: float = 0.0,
    ) -> float:
        """Admission score: should this state be stored in the bank?"""
        parts = []
        for comp in self.write_components:
            val = normalized_cues.get(comp)
            if val is not None:
                parts.append(val)
        if not parts:
            return raw_reliability
        base = sum(parts) / len(parts)
        geo = 1.0 if geometry_valid else 0.0
        return base + self.geometry_weight * geo

    def r_retrieval(
        self,
        r_write_val: float,
        anchor_age: int,
        diversity_score: float,
        reachability: float = 1.0,
        current_frame: int = 0,
    ) -> float:
        """Retrieval utility: should this anchor be retrieved for recovery?"""
        result = r_write_val
        if "freshness" in self.retrieval_components:
            stale_risk = min(anchor_age / 100.0, 1.0) if anchor_age > 0 else 0.0
            result -= 0.10 * stale_risk
        if "diversity" in self.retrieval_components:
            result += 0.15 * diversity_score
        if "reachability" in self.retrieval_components:
            result += 0.10 * reachability
        return result


# Diversity-preserving bank.

class DiversityBank:
    """B3: Greedy redundancy replacement for bank diversity.

    Uses normalized position, scale, and time distance to detect
    near-duplicates.  If a new state is near-duplicate of an existing
    bank entry, keep the one with better R_write.  If the bank is full,
    evict the state with low retrieval value and high redundancy.
    """

    def __init__(
        self,
        pos_threshold: float = 30.0,
        scale_threshold: float = 0.3,
        time_threshold: int = 50,
    ):
        self.pos_threshold = pos_threshold
        self.scale_threshold = scale_threshold
        self.time_threshold = time_threshold

    @staticmethod
    def _box_center(box: list[float]) -> tuple[float, float]:
        return (box[0] + 0.5 * box[2], box[1] + 0.5 * box[3])

    @staticmethod
    def _box_scale(box: list[float]) -> float:
        return math.sqrt(max(box[2] * box[3], 1e-6))

    def is_near_duplicate(
        self,
        box_a: list[float],
        frame_a: int,
        box_b: list[float],
        frame_b: int,
    ) -> bool:
        cx_a, cy_a = self._box_center(box_a)
        cx_b, cy_b = self._box_center(box_b)
        pos_dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
        if pos_dist > self.pos_threshold:
            return False
        scale_a = self._box_scale(box_a)
        scale_b = self._box_scale(box_b)
        scale_ratio = min(scale_a, scale_b) / max(scale_a, scale_b)
        if (1.0 - scale_ratio) > self.scale_threshold:
            return False
        time_dist = abs(frame_a - frame_b)
        if time_dist > self.time_threshold:
            return False
        return True

    def find_replacement_target(
        self,
        new_box: list[float],
        new_frame: int,
        new_r_write: float,
        bank_entries: list[dict],
    ) -> Optional[int]:
        """Return index of bank entry to replace, or None if no replacement.

        bank_entries: list of dicts with keys 'box', 'frame', 'r_write', 'r_ret'.
        """
        # Check for near-duplicate first
        for i, entry in enumerate(bank_entries):
            if self.is_near_duplicate(new_box, new_frame, entry["box"], entry["frame"]):
                if new_r_write > entry.get("r_write", 0.0):
                    return i
                return None  # New state is worse than existing duplicate

        # If bank is full, evict lowest r_ret with highest redundancy
        if len(bank_entries) >= 10:  # bank_size default
            worst_idx = -1
            worst_score = float("inf")
            for i, entry in enumerate(bank_entries):
                r_ret = entry.get("r_ret", entry.get("r_write", 0.0))
                if r_ret < worst_score:
                    worst_score = r_ret
                    worst_idx = i
            return worst_idx

        return None

    def compute_diversity_score(
        self,
        candidate_box: list[float],
        bank_boxes: list[list[float]],
    ) -> float:
        """Compute how diverse a candidate is from the current bank."""
        if not bank_boxes:
            return 1.0
        cx, cy = self._box_center(candidate_box)
        min_dist = float("inf")
        for bb in bank_boxes:
            bx, by = self._box_center(bb)
            d = math.hypot(cx - bx, cy - by)
            min_dist = min(min_dist, d)
        return min(min_dist / self.pos_threshold, 1.0)


# Support-aware candidate selection.

class SupportSafetyProxy:
    """B4: GT-free proxy for whether an accepted state restores support.

    Components logged separately:
    - predicted center distance to crop center
    - predicted-box boundary margin
    - predicted box fully inside crop
    - scale/aspect consistency
    - committed-state or short-trajectory consistency
    """

    def __init__(
        self,
        center_dist_weight: float = 0.30,
        boundary_margin_weight: float = 0.25,
        inside_weight: float = 0.15,
        scale_consistency_weight: float = 0.15,
        trajectory_consistency_weight: float = 0.15,
    ):
        self.center_dist_weight = center_dist_weight
        self.boundary_margin_weight = boundary_margin_weight
        self.inside_weight = inside_weight
        self.scale_consistency_weight = scale_consistency_weight
        self.trajectory_consistency_weight = trajectory_consistency_weight

    @staticmethod
    def _box_center(box: list[float]) -> tuple[float, float]:
        return (box[0] + 0.5 * box[2], box[1] + 0.5 * box[3])

    @staticmethod
    def _box_area(box: list[float]) -> float:
        return max(box[2] * box[3], 1e-6)

    def compute(
        self,
        candidate_box: list[float],
        crop_center: tuple[float, float],
        crop_half_size: float,
        committed_box: Optional[list[float]] = None,
        anchor_box: Optional[list[float]] = None,
    ) -> dict[str, float]:
        """Return support safety score and individual components."""
        cx, cy = self._box_center(candidate_box)

        # 1. Center distance (normalized to [0, 1], closer = higher)
        center_dist = math.hypot(cx - crop_center[0], cy - crop_center[1])
        center_score = max(0.0, 1.0 - center_dist / max(crop_half_size, 1e-6))

        # 2. Boundary margin (how far inside the crop the box is)
        margin_left = candidate_box[0] - (crop_center[0] - crop_half_size)
        margin_right = (crop_center[0] + crop_half_size) - (candidate_box[0] + candidate_box[2])
        margin_top = candidate_box[1] - (crop_center[1] - crop_half_size)
        margin_bottom = (crop_center[1] + crop_half_size) - (candidate_box[1] + candidate_box[3])
        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
        boundary_score = max(0.0, min_margin / max(crop_half_size, 1e-6))

        # 3. Fully inside crop
        inside_score = 1.0 if min_margin >= 0 else 0.0

        # 4. Scale consistency with anchor or committed state
        cand_area = self._box_area(candidate_box)
        if anchor_box is not None:
            ref_area = self._box_area(anchor_box)
            ratio = min(cand_area, ref_area) / max(cand_area, ref_area)
            scale_score = ratio
        else:
            scale_score = 1.0

        # 5. Trajectory consistency with committed state
        if committed_box is not None:
            ccx, ccy = self._box_center(committed_box)
            traj_dist = math.hypot(cx - ccx, cy - ccy)
            traj_score = max(0.0, 1.0 - traj_dist / max(crop_half_size * 2, 1e-6))
        else:
            traj_score = 1.0

        support_safety = (
            self.center_dist_weight * center_score
            + self.boundary_margin_weight * boundary_score
            + self.inside_weight * inside_score
            + self.scale_consistency_weight * scale_score
            + self.trajectory_consistency_weight * traj_score
        )

        return {
            "support_safety": support_safety,
            "center_dist_score": center_score,
            "boundary_margin_score": boundary_score,
            "inside_score": inside_score,
            "scale_consistency_score": scale_score,
            "trajectory_consistency_score": traj_score,
        }
