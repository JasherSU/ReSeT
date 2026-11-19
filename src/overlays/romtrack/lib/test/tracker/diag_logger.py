"""Unified per-frame diagnostic logger for ReSeT cross-tracker generalization.

Standardizes the diagnostic fields across all trackers so that downstream
analysis scripts can consume a single schema.

Usage in tracker track():
    writer = DiagWriter(results_dir, dataset, seq_name, mode='reset')
    ...
    frame = DiagFrame(video_id=seq_name, frame_id=self.frame_id, ...)
    writer.write(frame)
    ...
    writer.close()   # called once after the sequence finishes
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiagFrame:
    """Per-frame diagnostic record.

    Fields are split into:
    - base fields: always present (Base & ReSeT)
    - reset fields: only meaningful when ReSeT is active
    """
    # --- base fields (always present) ---
    video_id: str = ""
    frame_id: int = 0
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    score: float = 0.0
    apce: float = 0.0
    margin: float = 0.0
    runtime: float = 0.0  # seconds for this frame

    # --- ReSeT fields (absent or None for Base) ---
    triggered: Optional[bool] = None
    trigger_type: Optional[str] = None
    num_candidates: Optional[int] = None
    accepted: Optional[bool] = None
    accepted_candidate_source: Optional[str] = None
    anchor_rank: Optional[int] = None  # rank of accepted candidate in bank
    anchor_age: Optional[int] = None   # frames since accepted candidate was cached
    extra_forward_count: Optional[int] = None  # candidate forwards this frame
    bank_size: Optional[int] = None
    bank_write: Optional[bool] = None  # whether delayed bank write committed this frame
    # RelVerify extra
    relverify_rejected: Optional[bool] = None  # whether RelVerify rejected an accepted candidate
    # ReSeT-Cal calibration fields
    calibration_mode: Optional[str] = None         # 'fixed' / 'online_quantile' / 'rate_control' / 'hybrid'
    score_thr_hard: Optional[float] = None         # current hard_lost score threshold
    score_thr_uncertain: Optional[float] = None    # current uncertain score threshold
    apce_thr: Optional[float] = None               # current APCE threshold
    num_reliable_samples: Optional[int] = None     # reliable frames in window
    warmup_done: Optional[bool] = None             # whether warmup is complete
    q_hard_current: Optional[float] = None         # current q_hard (may be adjusted by rate controller)
    q_uncertain_current: Optional[float] = None    # current q_uncertain
    q_apce_current: Optional[float] = None         # current q_apce
    trigger_rate_window: Optional[float] = None    # recent trigger rate in controller window
    calibration_update_flag: Optional[bool] = None # whether calibration was updated this frame


class DiagWriter:
    """Writes DiagFrame records to a JSONL file."""

    def __init__(self, results_dir: str, dataset: str, seq_name: str,
                 mode: str = "reset"):
        """
        Args:
            results_dir: tracker results root dir
            dataset: dataset name (e.g. 'antiuav410_fair')
            seq_name: sequence name
            mode: 'base' or 'reset' -- used in output path
        """
        out_dir = os.path.join(results_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_reset" if mode == "reset" else ""
        self.path = os.path.join(out_dir, f"{seq_name}{suffix}_diag.jsonl")
        self._fh = open(self.path, "w")

    def write(self, frame: DiagFrame) -> None:
        d = asdict(frame)
        # Remove None values to keep jsonl compact
        d = {k: v for k, v in d.items() if v is not None}
        self._fh.write(json.dumps(d) + "\n")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
