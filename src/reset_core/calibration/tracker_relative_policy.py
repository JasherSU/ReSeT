"""Tracker-relative gates and safe online statistics updates.

This module provides the normalization and statistics-update policies
that convert absolute cues (score, APCE, margin, edge_risk, disagreement)
into tracker-relative values.

Design constraints (from handoff Section 5/7):
- No learned module; all normalization is online and statistics-based.
- The policy is shared across trackers; only online statistics adapt.
- Normalization must not change raw cue values, only produce normalized
  companions.
- Statistics may update only on reliable ordinary local frames (B1-S).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional


class StreamingQuantile:
    """Approximate streaming quantile via adaptive buffer.

    Maintains a sorted buffer of recent values and provides quantile
    queries.  When the buffer exceeds ``max_size``, the oldest entry
    is removed before inserting the new one.
    """

    __slots__ = ("buffer", "max_size")

    def __init__(self, max_size: int = 500):
        self.buffer: list[float] = []
        self.max_size = max_size

    def push(self, val: float) -> None:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return
        self.buffer.append(val)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def quantile(self, q: float) -> float:
        if not self.buffer:
            return 0.0
        s = sorted(self.buffer)
        pos = (len(s) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    def median(self) -> float:
        return self.quantile(0.5)

    def iqr(self) -> float:
        return self.quantile(0.75) - self.quantile(0.25)

    def count(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


class TrackerRelativeNormalizer:
    """B1: Converts raw cues to tracker-relative normalized values.

    Supported normalization families:
    - ``median_iqr``: (x - median) / IQR
    - ``percentile_rank``: fraction of historical values below x
    - ``rolling_quantile``: same as percentile_rank but with sliding window
    - ``zscore``: (x - mean) / std (diagnostic only, not preferred)

    The normalizer maintains streaming statistics for each cue and
    produces normalized values on demand.  Statistics only update when
    ``update_allowed=True`` (controlled by B1-S policy).
    """

    CUE_NAMES = ("score", "apce", "margin", "edge_risk", "disagreement")

    def __init__(
        self,
        normalization: str = "median_iqr",
        buffer_size: int = 500,
        min_samples: int = 20,
        stats_mode: str = "streaming",
        exp_alpha: float = 0.05,
    ):
        self.normalization = normalization
        self.min_samples = min_samples
        self.stats_mode = stats_mode
        self.exp_alpha = exp_alpha
        if stats_mode == "exponential":
            self._stats = {
                cue: ExponentialStats(alpha=exp_alpha)
                for cue in self.CUE_NAMES
            }
        else:
            self._stats = {
                cue: StreamingQuantile(max_size=buffer_size)
                for cue in self.CUE_NAMES
            }

    def update(self, cues: dict[str, float], update_allowed: bool) -> None:
        """Push new cue values into the streaming statistics.

        Only called when ``update_allowed`` is True (B1-S gate).
        """
        if not update_allowed:
            return
        for cue in self.CUE_NAMES:
            val = cues.get(cue)
            if val is not None:
                self._stats[cue].push(val)

    def reset(self) -> None:
        """Clear all streaming statistics (called on sequence re-initialization)."""
        for sq in self._stats.values():
            sq.clear()

    def normalize(self, cue: str, raw_value: float) -> Optional[float]:
        """Return the normalized value, or None if insufficient samples."""
        sq = self._stats[cue]
        if sq.count() < self.min_samples:
            return None
        if self.normalization == "median_iqr":
            med = sq.median()
            iqr = sq.iqr()
            if iqr < 1e-8:
                return 0.0
            return (raw_value - med) / iqr
        elif self.normalization in ("percentile_rank", "rolling_quantile"):
            if isinstance(sq, ExponentialStats):
                std = math.sqrt(sq.var) if sq.var and sq.var > 0 else 1.0
                z = (raw_value - sq.mean) / std if sq.mean is not None else 0.0
                return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            s = sorted(sq.buffer)
            count_below = sum(1 for v in s if v < raw_value)
            return count_below / max(len(s), 1)
        elif self.normalization == "zscore":
            if isinstance(sq, ExponentialStats):
                if sq.mean is None:
                    return None
                std = math.sqrt(sq.var) if sq.var and sq.var > 0 else 1.0
                return (raw_value - sq.mean) / std
            n = sq.count()
            if n == 0:
                return None
            mean = sum(sq.buffer) / n
            var = sum((v - mean) ** 2 for v in sq.buffer) / n
            std = math.sqrt(var) if var > 0 else 1.0
            return (raw_value - mean) / std
        else:
            return None

    def normalize_all(self, cues: dict[str, float]) -> dict[str, Optional[float]]:
        return {cue: self.normalize(cue, cues.get(cue, 0.0)) for cue in self.CUE_NAMES}

    def stats_count(self) -> int:
        return self._stats["score"].count()

    def ready(self) -> bool:
        """Whether the normalizer has enough samples to produce normalized output."""
        return self._stats["score"].count() >= self.min_samples

    def raw_quantile(self, cue: str, q: float) -> float:
        """Return the raw-space quantile for a cue (useful for threshold conversion)."""
        return self._stats[cue].quantile(q)

    def trigger_thresholds(self) -> tuple[float, float, float]:
        """Return (score_low, score_gray, apce_low) in NORMALIZED space.

        These thresholds replace the absolute 0.35/0.40/80.0 values.
        A cue below score_low triggers hard_lost; below score_gray with
        low apce also triggers.
        """
        if self.normalization == "median_iqr":
            return (-1.5, -1.0, -1.5)
        elif self.normalization == "percentile_rank":
            return (0.20, 0.35, 0.25)
        elif self.normalization == "rolling_quantile":
            return (0.20, 0.35, 0.25)
        elif self.normalization == "zscore":
            return (-1.5, -1.0, -1.5)
        else:
            return (0.20, 0.35, 0.25)

    def write_threshold(self) -> float:
        """Return the normalized score threshold for bank write (reliable decision).

        A state is considered reliable if its normalized score is at or above
        this threshold.
        """
        if self.normalization == "median_iqr":
            return 0.0  # at median
        elif self.normalization == "percentile_rank":
            return 0.50  # 50th percentile
        elif self.normalization == "rolling_quantile":
            return 0.50
        elif self.normalization == "zscore":
            return 0.0
        else:
            return 0.50

    def acceptance_delta(self) -> float:
        """Return the normalized score delta for verifier acceptance.

        A candidate must have normalized_score > normalized_ha_score + delta.
        """
        if self.normalization == "median_iqr":
            return 0.5  # 0.5 IQR above HA
        elif self.normalization == "percentile_rank":
            return 0.10  # 10 percentile points above HA
        elif self.normalization == "rolling_quantile":
            return 0.10
        elif self.normalization == "zscore":
            return 0.5
        else:
            return 0.10


class ExponentialStats:
    """Exponentially-weighted statistics for a single cue.

    Maintains EWMA mean and variance.  Quantile estimates use the normal
    approximation (median = mean, IQR = 1.349 * std), which is sufficient
    for median_iqr normalization screening.
    """

    __slots__ = ("alpha", "mean", "var", "_count")

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mean: Optional[float] = None
        self.var: Optional[float] = None
        self._count: int = 0

    def push(self, val: float) -> None:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return
        if self.mean is None:
            self.mean = val
            self.var = 0.0
        else:
            diff = val - self.mean
            self.mean = self.alpha * val + (1.0 - self.alpha) * self.mean
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * diff * diff)
        self._count += 1

    def quantile(self, q: float) -> float:
        if self.mean is None:
            return 0.0
        std = math.sqrt(self.var) if self.var and self.var > 0 else 1.0
        # Inverse normal CDF via rational approximation (Acklam's algorithm)
        if q <= 0.0:
            return self.mean - 3.0 * std
        if q >= 1.0:
            return self.mean + 3.0 * std
        # Acklam's inverse CDF approximation
        a_coefs = [-3.969683028665376e+01, 2.209460984245205e+02,
                    -2.759285104469687e+02, 1.383577518672690e+02,
                    -3.066479806614716e+01, 2.506628277459239e+00]
        b_coefs = [-5.447609879822406e+01, 1.615858368580409e+02,
                    -1.556989798598866e+02, 6.680131188771972e+01,
                    -1.328068155288572e+01]
        p_low = 0.02425
        p_high = 1.0 - p_low
        if q < p_low:
            r = math.sqrt(-2.0 * math.log(q))
            z = (((((a_coefs[0] * r + a_coefs[1]) * r + a_coefs[2]) * r + a_coefs[3]) * r + a_coefs[4]) * r + a_coefs[5]) / \
                ((((b_coefs[0] * r + b_coefs[1]) * r + b_coefs[2]) * r + b_coefs[3]) * r + b_coefs[4])
        elif q <= p_high:
            r = q - 0.5
            s = r * r
            z = (((((a_coefs[0] * s + a_coefs[1]) * s + a_coefs[2]) * s + a_coefs[3]) * s + a_coefs[4]) * s + a_coefs[5]) * r / \
                (((((b_coefs[0] * s + b_coefs[1]) * s + b_coefs[2]) * s + b_coefs[3]) * s + b_coefs[4]) * s + 1.0)
        else:
            r = math.sqrt(-2.0 * math.log(1.0 - q))
            z = -(((((a_coefs[0] * r + a_coefs[1]) * r + a_coefs[2]) * r + a_coefs[3]) * r + a_coefs[4]) * r + a_coefs[5]) / \
                ((((b_coefs[0] * r + b_coefs[1]) * r + b_coefs[2]) * r + b_coefs[3]) * r + b_coefs[4])
        return self.mean + std * z

    def median(self) -> float:
        return self.mean if self.mean is not None else 0.0

    def iqr(self) -> float:
        if self.var is None:
            return 0.0
        std = math.sqrt(self.var)
        return 1.349 * std  # IQR for normal distribution

    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self.mean = None
        self.var = None
        self._count = 0


class SafeStatsUpdatePolicy:
    """B1-S: Decides whether statistics may update on a given frame.

    Statistics update is allowed only on reliable ordinary local frames
    that pass the strict write conditions and are NOT:
    - recovery frames
    - trigger frames
    - pending-confirmation frames
    - hard-loss frames
    - high-confidence-risk frames
    - geometry-invalid frames

    Extended policies for D2-B:
    - ``sliding_window``: update on all reliable frames, normalizer uses
      a small buffer (100) for adaptive local statistics.
    - ``exponential``: update on all frames, normalizer uses exponential
      weighting for smooth forgetting.
    """

    def __init__(self, policy: str = "reliable_local"):
        self.policy = policy

    def allow_update(
        self,
        is_recovery: bool = False,
        is_trigger: bool = False,
        is_pending_confirmation: bool = False,
        is_hard_loss: bool = False,
        is_high_conf_risk: bool = False,
        is_geometry_valid: bool = True,
        is_reliable: bool = False,
    ) -> bool:
        if self.policy == "all_frames":
            return True
        elif self.policy == "reliable_local":
            if is_recovery or is_trigger or is_pending_confirmation:
                return False
            if is_hard_loss or is_high_conf_risk:
                return False
            if not is_geometry_valid:
                return False
            return is_reliable
        elif self.policy == "bank_write_only":
            return is_reliable and not is_recovery
        elif self.policy == "exclude_trigger":
            return not is_trigger and is_reliable
        elif self.policy == "exclude_recovery_pending":
            if is_recovery or is_pending_confirmation:
                return False
            return is_reliable
        elif self.policy == "frozen":
            return False
        elif self.policy == "sliding_window":
            if is_recovery or is_pending_confirmation:
                return False
            return is_reliable and is_geometry_valid
        elif self.policy == "exponential":
            return True
        else:
            return False
