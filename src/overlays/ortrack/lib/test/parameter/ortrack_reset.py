"""Frozen ORTrack + ReSeT inference configuration used for the headline run."""

from lib.test.parameter.ortrack import parameters as base_parameters


def parameters(_: str):
    params = base_parameters("vit_tiny_antiuav")

    # Search-state intervention and ordinary local continuation.
    params.dynamic_search = True
    params.hann_control = True
    params.conf_thresh = 0.35
    params.enlarged_search_factor = 5.0

    # Bounded proposal ranking and current-frame verification.
    params.heuristic_rerank = True
    params.rerank_lambda = 0.10
    params.rerank_mu = 0.05
    params.rerank_nu = 0.05
    params.rerank_topk = 5
    params.e27_mode = "C1"
    params.e27_bank_size = 10
    params.e27_bank_topk = 3
    params.e27_use_offsets = True
    params.e27_offset_topk = 1
    params.e27_max_candidates = 10
    params.e27_delta_score = 0.05
    params.e27_margin_thresh = 0.08
    params.e27_strict_delta_score = 0.08
    params.e27_strict_margin_thresh = 0.12
    params.e27_apce_low = 80.0
    params.e27_long_unreliable = 999999
    params.e27_trigger_policy = "score_apce_gate"
    params.e27_bank_ranker = "reliability"
    params.e27_local_neighborhood = "local_lr"
    params.e27_exclude_sources = "C3_last_high_conf"

    # An accepted prediction may guide the next crop immediately, but its
    # recovery state is admitted to the reusable bank only after confirmation.
    params.e29_delayed_bank_write = True
    params.e29_reject_fallback = "ha"
    params.dump_frames = True
    return params
