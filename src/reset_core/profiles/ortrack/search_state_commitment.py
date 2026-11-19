from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .state_cache import CachedState


PROTOCOLS = {
    "legacy",
    "immediate",
    "fixed_delay",
    "reliability_confirmed",
    "three_state",
}

GATES = {
    "score",
    "apce",
    "score_apce",
    "score_apce_disp",
    "score_apce_disp_scale",
}

RELATIVE_ACCEPTANCE = {"none", "current", "committed", "both"}

ANCHOR_POLICIES = {
    "last_committed",
    "best_score",
    "best_reliability",
    "reliability_freshness",
    "topm",
    "oracle_diag",
}

ORACLE_MODES = {"none", "commit", "isolation", "anchor"}


def _center(box: Sequence[float]) -> tuple[float, float]:
    return float(box[0]) + 0.5 * float(box[2]), float(box[1]) + 0.5 * float(box[3])


def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _area_sqrt(box: Sequence[float]) -> float:
    return max(float(box[2]) * float(box[3]), 1.0) ** 0.5


def _scale_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    aw = max(float(a[2]), 1e-6)
    ah = max(float(a[3]), 1e-6)
    bw = max(float(b[2]), 1e-6)
    bh = max(float(b[3]), 1e-6)
    return max(aw / bw, bw / aw, ah / bh, bh / ah)


def candidate_reliability(candidate: Dict[str, Any]) -> float:
    score = float(candidate.get("score", candidate.get("raw_score", 0.0)))
    apce = min(float(candidate.get("apce", 0.0)) / 100.0, 2.0)
    margin = float(candidate.get("margin", 0.0))
    edge_penalty = float(candidate.get("edge_risk", 0.0)) * 0.25
    return score + 0.20 * apce + 0.50 * margin - edge_penalty


def state_reliability(state: Optional[CachedState]) -> float:
    if state is None:
        return -1e9
    if state.reliability_score:
        return float(state.reliability_score)
    return candidate_reliability({
        "score": state.raw_score,
        "apce": state.apce,
        "margin": state.margin,
        "edge_risk": state.edge_risk,
    })


@dataclass
class CommitmentDecision:
    state_role: str = "tentative"
    commit_decision: str = "none"
    commit_gate_pass: bool = False
    bank_write_allowed: bool = False
    isolated_flag: bool = False
    committed_source: str = ""
    tentative_source: str = ""
    anchor_source: str = ""

    def as_diag(self) -> Dict[str, Any]:
        return {
            "state_role": self.state_role,
            "commit_decision": self.commit_decision,
            "commit_gate_pass": bool(self.commit_gate_pass),
            "committed_source": self.committed_source,
            "tentative_source": self.tentative_source,
            "isolated_flag": bool(self.isolated_flag),
            "anchor_source": self.anchor_source,
        }


class SearchStateCommitment:
    """Reliability-guided search-state commitment protocol for SSRM."""

    def __init__(
        self,
        protocol: str = "legacy",
        commit_gate: str = "score_apce",
        delay_frames: int = 1,
        relative_acceptance: str = "none",
        anchor_policy: str = "last_committed",
        oracle_mode: str = "none",
        score_thresh: float = 0.65,
        apce_thresh: float = 80.0,
        margin_thresh: float = 0.12,
        edge_thresh: float = 0.75,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown commit protocol: {protocol}")
        if commit_gate not in GATES:
            raise ValueError(f"Unknown commit gate: {commit_gate}")
        if relative_acceptance not in RELATIVE_ACCEPTANCE:
            raise ValueError(f"Unknown relative acceptance: {relative_acceptance}")
        if anchor_policy not in ANCHOR_POLICIES:
            raise ValueError(f"Unknown anchor policy: {anchor_policy}")
        if oracle_mode not in ORACLE_MODES:
            raise ValueError(f"Unknown commit oracle mode: {oracle_mode}")
        self.protocol = protocol
        self.commit_gate = commit_gate
        self.delay_frames = max(1, int(delay_frames))
        self.relative_acceptance = relative_acceptance
        self.anchor_policy = anchor_policy
        self.oracle_mode = oracle_mode
        self.score_thresh = float(score_thresh)
        self.apce_thresh = float(apce_thresh)
        self.margin_thresh = float(margin_thresh)
        self.edge_thresh = float(edge_thresh)
        self.failure_segment_id = 0
        self.failure_active = False
        self.failure_start_frame = -1

    @property
    def enabled(self) -> bool:
        return self.protocol != "legacy" or self.oracle_mode != "none"

    def commit_gate_passes(
        self,
        state: CachedState,
        committed_state: Optional[CachedState] = None,
    ) -> bool:
        score_pass = float(state.raw_score) >= self.score_thresh
        apce_pass = float(state.apce) >= self.apce_thresh and float(state.margin) >= self.margin_thresh
        if float(state.edge_risk) > self.edge_thresh:
            apce_pass = False
        if self.commit_gate == "score":
            return score_pass
        if self.commit_gate == "apce":
            return apce_pass
        if self.commit_gate == "score_apce":
            return score_pass and apce_pass

        disp_pass = True
        scale_pass = True
        if committed_state is not None:
            max_disp = max(24.0, 2.5 * _area_sqrt(committed_state.box))
            disp_pass = _center_distance(state.box, committed_state.box) <= max_disp
            scale_pass = _scale_ratio(state.box, committed_state.box) <= 2.0
        if self.commit_gate == "score_apce_disp":
            return score_pass and apce_pass and disp_pass
        return score_pass and apce_pass and disp_pass and scale_pass

    def current_observation_decision(
        self,
        state: CachedState,
        reliable: bool,
        high_conf: bool,
        gate_pass: bool,
        gt_iou: Optional[float] = None,
    ) -> CommitmentDecision:
        if self.oracle_mode == "commit" and gt_iou is not None:
            gate_pass = gt_iou >= 0.5
            reliable = gate_pass
        if self.oracle_mode == "isolation" and gt_iou is not None and gt_iou < 0.5:
            return CommitmentDecision(
                state_role="isolated",
                commit_decision="oracle_isolate",
                commit_gate_pass=gate_pass,
                isolated_flag=True,
                tentative_source=state.source,
            )
        if self.protocol == "immediate" and (reliable or high_conf) and gate_pass:
            return CommitmentDecision(
                state_role="committed",
                commit_decision="immediate_commit",
                commit_gate_pass=True,
                bank_write_allowed=True,
                committed_source=state.source,
            )
        if reliable and gate_pass:
            return CommitmentDecision(
                state_role="committed",
                commit_decision="current_commit",
                commit_gate_pass=True,
                bank_write_allowed=True,
                committed_source=state.source,
            )
        if self.protocol == "three_state" and not gate_pass:
            return CommitmentDecision(
                state_role="isolated",
                commit_decision="isolate_current",
                commit_gate_pass=False,
                isolated_flag=True,
                tentative_source=state.source,
            )
        return CommitmentDecision(
            state_role="tentative",
            commit_decision="tentative_output",
            commit_gate_pass=gate_pass,
            tentative_source=state.source,
        )

    def pending_commit_decision(
        self,
        cache: Any,
        commit_gate_pass: Optional[bool] = None,
        reliable: bool = False,
        gate_pass: Optional[bool] = None,
    ) -> CommitmentDecision:
        if commit_gate_pass is None:
            commit_gate_pass = bool(gate_pass)
        pending = getattr(cache, "pending_recovery", None)
        if pending is None:
            return CommitmentDecision(commit_gate_pass=commit_gate_pass)
        if self.protocol == "immediate":
            return CommitmentDecision(
                state_role="committed",
                commit_decision="pending_immediate_commit",
                commit_gate_pass=True,
                bank_write_allowed=True,
                committed_source=str(pending.source),
            )
        if self.protocol in {"fixed_delay", "reliability_confirmed", "three_state"}:
            if reliable and commit_gate_pass:
                cache.pending_recovery_good += 1
            required = max(self.delay_frames, int(getattr(cache, "pending_recovery_required", 1)))
            if cache.pending_recovery_good >= required:
                return CommitmentDecision(
                    state_role="committed",
                    commit_decision=f"{self.protocol}_commit",
                    commit_gate_pass=commit_gate_pass,
                    bank_write_allowed=True,
                    committed_source=str(pending.source),
                )
            return CommitmentDecision(
                state_role="tentative",
                commit_decision=f"{self.protocol}_wait",
                commit_gate_pass=commit_gate_pass,
                tentative_source=str(pending.source),
            )
        return CommitmentDecision(commit_gate_pass=commit_gate_pass)

    def recovery_decision(
        self,
        recovered_state: CachedState,
        required_frames: int,
    ) -> CommitmentDecision:
        if self.protocol == "immediate":
            return CommitmentDecision(
                state_role="committed",
                commit_decision="accepted_immediate_commit",
                commit_gate_pass=True,
                bank_write_allowed=True,
                committed_source=recovered_state.source,
            )
        if self.protocol in {"fixed_delay", "reliability_confirmed", "three_state"}:
            return CommitmentDecision(
                state_role="tentative",
                commit_decision=f"accepted_pending_{max(self.delay_frames, int(required_frames))}",
                commit_gate_pass=True,
                tentative_source=recovered_state.source,
            )
        return CommitmentDecision(
            state_role="tentative",
            commit_decision="legacy_recovery",
            commit_gate_pass=True,
            tentative_source=recovered_state.source,
        )

    def accepts_recovery(
        self,
        accepted: Optional[Dict[str, Any]],
        current_state: Optional[CachedState],
        committed_state: Optional[CachedState],
    ) -> bool:
        if accepted is None or self.relative_acceptance == "none":
            return True
        cand_rel = candidate_reliability(accepted)
        checks = []
        if self.relative_acceptance in {"current", "both"}:
            checks.append(cand_rel > state_reliability(current_state))
        if self.relative_acceptance in {"committed", "both"}:
            checks.append(cand_rel >= state_reliability(committed_state) - 0.02)
        return all(checks) if checks else True

    def anchor_box(
        self,
        cache: Any,
        current_frame: int = 0,
        gt_bbox: Optional[Sequence[float]] = None,
    ) -> tuple[Optional[list[float]], str]:
        if self.oracle_mode == "anchor" and gt_bbox is not None:
            best_state = None
            best_iou = -1.0
            for state in list(getattr(cache, "bank", [])):
                iou = self._iou_xywh(state.box, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_state = state
            if best_state is not None:
                return list(best_state.box), "oracle_anchor"
        if self.anchor_policy == "last_committed" and getattr(cache, "last_reliable", None) is not None:
            return list(cache.last_reliable.box), "last_committed"
        bank = list(getattr(cache, "bank", []))
        if not bank:
            fallback = cache.fallback_box("last_reliable") if hasattr(cache, "fallback_box") else []
            return (list(fallback), "fallback") if fallback else (None, "")
        if self.anchor_policy == "best_score":
            state = max(bank, key=lambda s: (s.raw_score, s.margin, s.apce))
            return list(state.box), "best_score"
        if self.anchor_policy == "best_reliability":
            state = max(bank, key=state_reliability)
            return list(state.box), "best_reliability"
        if self.anchor_policy in {"reliability_freshness", "topm"}:
            state = max(
                bank,
                key=lambda s: state_reliability(s)
                - min(max(int(current_frame) - int(s.frame_id), 0), 80) * 0.0005,
            )
            return list(state.box), self.anchor_policy
        return list(bank[-1].box), "bank_tail"

    def outcome_diag(
        self,
        frame_id: int,
        output_iou: Optional[float],
        committed_iou: Optional[float],
        decision: CommitmentDecision,
        accepted_recovery: bool,
    ) -> Dict[str, Any]:
        corrupted_commit = bool(
            decision.bank_write_allowed
            and committed_iou is not None
            and committed_iou < 0.5
        )
        false_isolation = bool(
            decision.isolated_flag
            and output_iou is not None
            and output_iou >= 0.5
        )
        recovery10 = False
        recovery20 = False
        if output_iou is not None:
            if output_iou < 0.5 and not self.failure_active:
                self.failure_active = True
                self.failure_segment_id += 1
                self.failure_start_frame = int(frame_id)
            elif output_iou >= 0.5 and self.failure_active:
                age = int(frame_id) - int(self.failure_start_frame)
                recovery10 = age <= 10
                recovery20 = age <= 20
                self.failure_active = False
                self.failure_start_frame = -1
        return {
            "corrupted_commit": corrupted_commit,
            "false_isolation": false_isolation,
            "recovery_within_10": bool(recovery10 and accepted_recovery),
            "recovery_within_20": bool(recovery20 and accepted_recovery),
            "failure_segment_id": int(self.failure_segment_id if self.failure_active else 0),
        }

    @staticmethod
    def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, aw, ah = [float(v) for v in a[:4]]
        bx1, by1, bw, bh = [float(v) for v in b[:4]]
        ax2, ay2 = ax1 + max(aw, 0.0), ay1 + max(ah, 0.0)
        bx2, by2 = bx1 + max(bw, 0.0), by1 + max(bh, 0.0)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        inter = iw * ih
        union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
        return inter / union if union > 0.0 else 0.0
