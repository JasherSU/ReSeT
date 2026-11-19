"""Reusable adapter API for center-response local trackers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


NATIVE_ADAPTER_HOOKS = (
    "_e27_forward_center_candidate",
    "_e27_finalize",
)


class ReSeTAdapterMixin:
    """Default tracker-independent parts of the ReSeT adapter contract.

    A new center-response tracker implements only candidate forwarding and
    final output/state routing. Response statistics and geometry checks are
    shared here.
    """

    def _e27_branch_stats(self, score_map: Any) -> Mapping[str, float]:
        from .e27_recovery_policy import score_map_stats

        return score_map_stats(score_map, self.feat_sz)

    def _e27_is_bbox_normal(
        self,
        bbox: Optional[Sequence[float]],
        ref_bbox: Optional[Sequence[float]] = None,
    ) -> bool:
        if bbox is None or len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return False
        if ref_bbox is None or len(ref_bbox) < 4 or ref_bbox[2] <= 0 or ref_bbox[3] <= 0:
            return True
        area = max(float(bbox[2]) * float(bbox[3]), 1e-6)
        ref_area = max(float(ref_bbox[2]) * float(ref_bbox[3]), 1e-6)
        size_ratio = area / ref_area
        aspect = (float(bbox[2]) / max(float(bbox[3]), 1e-6)) / (
            float(ref_bbox[2]) / max(float(ref_bbox[3]), 1e-6)
        )
        return 0.5 <= size_ratio <= 2.0 and 0.33 <= aspect <= 3.0

    def _e27_candidate_record(
        self,
        candidate: Mapping[str, Any],
        branch: Mapping[str, Any],
        include_branch: bool = True,
    ) -> dict:
        stats = self._e27_branch_stats(branch["out_dict"]["score_map"])
        record = dict(candidate)
        record.update(
            {
                "bbox": branch["target_bbox"],
                "score": stats["score"],
                "apce": stats["apce"],
                "margin": stats["margin"],
                "edge_risk": stats["edge_risk"],
            }
        )
        if include_branch:
            record["branch"] = branch
        return record

    def _e27_forward_center_candidate(self, image: Any, candidate: Mapping[str, Any], height: int, width: int) -> Mapping[str, Any]:
        raise NotImplementedError("adapter must implement candidate forwarding")

    def _e27_finalize(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("adapter must implement output/state routing")
