"""Tracker adapter for ReSeT search-state management.

Runs the ordinary local forward first, invokes bounded alternative supports
only when the observation gate fails, routes an accepted current-frame
prediction immediately, and delays reusable-bank trust as described in the
main-paper Method section and Supplement Algorithm S1.
"""

import math
import time

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

from lib.test.tracker.reset_core.state_cache import E27StateCache
from lib.test.tracker.reset_core.e27_candidate_generator import E27CandidateGenerator
from lib.test.tracker.reset_core.e27_local_verifier import E27LocalVerifier
from lib.test.tracker.reset_core.e27_recovery_policy import E27RecoveryPolicy
from lib.test.tracker.reset_core.adapter_api import ReSeTAdapterMixin
from lib.test.tracker.diag_logger import DiagFrame
from lib.test.tracker.reset_core.search_state_commitment import SearchStateCommitment, candidate_reliability, state_reliability


class OSTrackReSeT(ReSeTAdapterMixin, BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrackReSeT, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # E29-C3 ReSeT/SSRM parameters
        self.conf_thresh = getattr(params, 'conf_thresh', 0.35)
        self.e27_mode = getattr(params, 'e27_mode', 'C1')
        self.e27_trigger_policy = getattr(params, 'e27_trigger_policy', 'legacy')
        self.e27_bank_ranker = getattr(params, 'e27_bank_ranker', 'reliability')
        self.e27_local_neighborhood = getattr(params, 'e27_local_neighborhood', 'local_lr')
        self.e29_delayed_bank_write = getattr(params, 'e29_delayed_bank_write', True)
        self.e27_exclude_sources = getattr(params, 'e27_exclude_sources', 'C3_last_high_conf')
        self.e27_bank_size = getattr(params, 'e27_bank_size', 10)
        self.e27_bank_topk = getattr(params, 'e27_bank_topk', 3)
        self.e27_use_offsets = getattr(params, 'e27_use_offsets', True)
        self.e27_offset_topk = getattr(params, 'e27_offset_topk', 1)
        self.e27_max_candidates = getattr(params, 'e27_max_candidates', 10)
        self.e27_delta_score = getattr(params, 'e27_delta_score', 0.05)
        self.e27_margin_thresh = getattr(params, 'e27_margin_thresh', 0.08)
        self.e27_strict_delta_score = getattr(params, 'e27_strict_delta_score', 0.08)
        self.e27_strict_margin_thresh = getattr(params, 'e27_strict_margin_thresh', 0.12)
        self.e27_apce_low = getattr(params, 'e27_apce_low', 80.0)
        self.e27_long_unreliable = getattr(params, 'e27_long_unreliable', 999999)
        self.e27_cooldown = getattr(params, 'e27_cooldown', 0)
        self.e27_reset_policy = getattr(params, 'e27_reset_policy', 'C')
        self.e27_state_quarantine = getattr(params, 'e27_state_quarantine', False)
        self.e27_reset_confidence = getattr(params, 'e27_reset_confidence', True)
        self.e27_reset_confidence_thresh = getattr(params, 'e27_reset_confidence_thresh', 0.12)
        self.e29_reject_fallback = getattr(params, 'e29_reject_fallback', 'ha')
        self.e29_c1_fallback_only = getattr(params, 'e29_c1_fallback_only', False)

        # RelVerify / SearchStateCommitment parameters
        self.commit_protocol = getattr(params, 'commit_protocol', 'legacy')
        self.relative_acceptance = getattr(params, 'relative_acceptance', 'none')
        self.commit_delay_frames = getattr(params, 'commit_delay_frames', 1)
        self.commit_gate = getattr(params, 'commit_gate', 'score_apce')
        self.anchor_policy = getattr(params, 'anchor_policy', 'last_committed')
        self.reset_commitment = None
        if self.commit_protocol != 'legacy':
            self.reset_commitment = SearchStateCommitment(
                protocol=self.commit_protocol,
                commit_gate=self.commit_gate,
                delay_frames=self.commit_delay_frames,
                relative_acceptance=self.relative_acceptance,
                anchor_policy=self.anchor_policy,
                apce_thresh=self.e27_apce_low,
                margin_thresh=self.e27_margin_thresh)

        self.dump_frames = getattr(params, 'dump_frames', False)

        # Initialize SSRM components
        self.e27_cache = E27StateCache(bank_size=self.e27_bank_size)
        self.e27_generator = E27CandidateGenerator(
            bank_topk=self.e27_bank_topk,
            use_offsets=self.e27_use_offsets,
            offset_topk=self.e27_offset_topk,
            max_candidates=self.e27_max_candidates,
            bank_ranker=self.e27_bank_ranker,
            local_neighborhood_mode=self.e27_local_neighborhood)
        self.e27_verifier = E27LocalVerifier(
            delta_score=self.e27_delta_score,
            margin_thresh=self.e27_margin_thresh,
            strict_delta_score=self.e27_strict_delta_score,
            strict_margin_thresh=self.e27_strict_margin_thresh,
            c1_fallback_only=self.e29_c1_fallback_only,
            reset_confidence=self.e27_reset_confidence,
            reset_confidence_thresh=self.e27_reset_confidence_thresh,
            exclude_sources=[s.strip() for s in str(self.e27_exclude_sources).split(',')])
        # --- ReSeT-Cal calibration ---
        self.calibration_mode = getattr(params, 'calibration_mode', 'fixed')
        self.calibration_params = getattr(params, 'calibration_params', {})
        self.calibration_engine = None

        if self.calibration_mode != 'fixed':
            import sys
            cal_engine_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
            cal_engine_dir = os.path.normpath(cal_engine_dir)
            if cal_engine_dir not in sys.path:
                sys.path.insert(0, cal_engine_dir)
            from calibration_engine import CalibrationEngine
            self.calibration_engine = CalibrationEngine(
                mode=self.calibration_mode, params=self.calibration_params)

        self.e27_policy = E27RecoveryPolicy(
            mode=self.e27_mode,
            apce_low=self.e27_apce_low,
            margin_low=self.e27_margin_thresh,
            long_unreliable=self.e27_long_unreliable,
            cooldown=self.e27_cooldown,
            trigger_policy=self.e27_trigger_policy,
            calibration_engine=self.calibration_engine)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        # Reset SSRM state
        self.e27_cache.reset(info['init_bbox'])
        self.e27_policy.reset()
        if self.calibration_engine is not None:
            self.calibration_engine.reset()

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _forward_candidate(self, image, search_state, H, W):
        """Forward pass for a candidate search region (used in ReSeT recovery)."""
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, search_state, self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=search.tensors, ce_template_mask=self.box_mask_z)

        pred_score_map = out_dict['score_map']
        raw_score_max = pred_score_map.max().item()
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()

        search_center = (search_state[0] + 0.5 * search_state[2],
                         search_state[1] + 0.5 * search_state[3])
        mapped_box = self._map_box_back_from_center(pred_box, resize_factor, search_center)
        clipped_state = clip_box(mapped_box, H, W, margin=10)

        return {
            'out_dict': out_dict,
            'pred_boxes': pred_boxes,
            'target_bbox': clipped_state,
            'raw_score_max': raw_score_max,
            'response_score_max': response.max().item(),
            'resize_factor': resize_factor,
        }

    def _map_box_back_from_center(self, pred_box, resize_factor, search_center):
        cx_prev, cy_prev = search_center
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _e27_forward_center_candidate(self, image, cand, H, W):
        """Forward a candidate center and collect verification stats."""
        branch = self._forward_candidate(image, cand['box'], H, W)
        stats = self._e27_branch_stats(branch['out_dict']['score_map'])
        item = dict(cand)
        item.update({
            'bbox': branch['target_bbox'],
            'score': stats['score'],
            'apce': stats['apce'],
            'margin': stats['margin'],
            'edge_risk': stats['edge_risk'],
        })
        return item

    def _e27_finalize(self, image, H, W, raw_box, ha_box, raw_stats, ha_score,
                      raw_is_lost, raw_displacement, raw_vs_ha_gap):
        cache = self.e27_cache

        score_low_dyn, score_gray_dyn, apce_low_dyn = self.e27_policy.dynamic_thresholds()
        if self.calibration_engine is not None:
            cal_thr = self.calibration_engine.get_thresholds()
            low_score_boundary = cal_thr.get("score_thr_hard", 0.35)
        elif self.e27_trigger_policy == 'relative_score_apce_gate':
            low_score_boundary = score_low_dyn
        else:
            low_score_boundary = 0.35
        # M10: Use calibrated streak boundary
        if self.calibration_engine is not None:
            cal_thr = self.calibration_engine.get_thresholds()
            streak_boundary = cal_thr.get("score_thr_streak", cal_thr.get("score_thr_hard", 0.35))
        else:
            streak_boundary = low_score_boundary  # existing logic
        if raw_stats['score'] < streak_boundary:
            cache.low_score_streak += 1
        else:
            cache.low_score_streak = 0
        if cache.cooldown_left > 0:
            cache.cooldown_left -= 1

        frames_since = cache.frames_since_last_reliable(self.frame_id)
        trigger_type = self.e27_policy.classify_trigger(
            raw_stats['score'], raw_stats['apce'], raw_stats['margin'],
            raw_displacement, raw_vs_ha_gap, frames_since,
            cache.low_score_streak, cache.cooldown_left)

        ref_for_reliable = cache.last_reliable.box if cache.last_reliable is not None else raw_box
        # M10: Use calibrated reliable thresholds if available
        if self.calibration_engine is not None:
            cal_thr = self.calibration_engine.get_thresholds()
            reliable_score_thr = cal_thr.get("score_thr_reliable", 0.65)
            reliable_apce_thr = cal_thr.get("apce_thr_reliable", 80.0)
            # M10 full_chain mode: reliable based only on calibrated score + APCE
            # (removes margin/edge_risk/ha_gap/bbox_normal conditions that prevent
            #  frames_since_last_reliable from resetting)
            m10_full_chain = getattr(self.calibration_engine, 'mode', '') == 'full_chain'
        else:
            reliable_score_thr = 0.65
            reliable_apce_thr = self.e27_apce_low
            m10_full_chain = False

        if m10_full_chain:
            # Optional simplified reliable check: score, APCE, and non-lost state.
            reliable = (
                raw_stats['score'] > reliable_score_thr
                and raw_stats['apce'] > reliable_apce_thr
                and not raw_is_lost
            )
        else:
            # Default reliable check with the complete condition set.
            reliable = (
                raw_stats['score'] > reliable_score_thr
                and raw_stats['apce'] > reliable_apce_thr
                and raw_stats['margin'] > 0.12
                and raw_stats['edge_risk'] < 0.75
                and raw_vs_ha_gap <= max(24.0, 2.5 * max(raw_box[2] * raw_box[3], 1.0) ** 0.5)
                and self._e27_is_bbox_normal(ha_box, ref_for_reliable)
                and not raw_is_lost
            )
        high_conf = (
            (raw_stats['score'] > 0.70 or ha_score > 0.70)
            and not raw_is_lost
            and self._e27_is_bbox_normal(ha_box, ref_for_reliable)
        )
        current_reliability = self.e27_policy.reliability_score(
            raw_stats['score'], raw_stats['apce'], raw_stats['margin'],
            raw_stats.get('edge_risk', 0.0), raw_vs_ha_gap, age=0)
        current_state = cache.make_state(
            self.frame_id, ha_box, raw_stats['score'], ha_score,
            raw_stats['apce'], raw_stats['margin'], 'HA',
            edge_risk=raw_stats.get('edge_risk', 0.0),
            raw_ha_gap=raw_vs_ha_gap,
            reliability_score=current_reliability)

        # Handle delayed bank write from pending recovery
        if cache.pending_recovery is not None and self.e29_delayed_bank_write and reliable:
            cache.pending_recovery_good += 1
            if cache.pending_recovery_good >= max(1, int(cache.pending_recovery_required)):
                cache.update_reliable(cache.pending_recovery)
                cache.clear_pending_recovery()

        accepted = None
        candidate_sources = []
        candidate_count = 0
        selected_source = 'HA'
        recovery_candidates = []
        defer_reliable_update = cache.pending_recovery is not None

        if trigger_type == 'normal':
            cache.set_search_state(ha_box)
            if reliable and not defer_reliable_update:
                cache.update_reliable(current_state)
            elif high_conf:
                cache.update_high_conf(current_state)
            output_box = ha_box
        else:
            output_box = ha_box  # default fallback
            candidates = self.e27_generator.build(ha_box, cache, current_frame=self.frame_id)
            candidate_sources = [c['source'] for c in candidates]
            candidate_count = len(candidates)
            for cand in candidates:
                recovery_candidates.append(
                    self._e27_forward_center_candidate(image, cand, H, W))
            last_rel_box = cache.last_reliable.box if cache.last_reliable is not None else None
            crop_side = max(raw_box[2] * raw_box[3], 1.0) ** 0.5 * 5.0
            accepted = self.e27_verifier.choose(
                recovery_candidates, ha_score, raw_stats['apce'],
                last_rel_box, trigger_type, crop_side)

            # --- RelVerify: check before using accepted candidate ---
            if accepted is not None and self.reset_commitment is not None:
                committed_state = cache.last_reliable
                if not self.reset_commitment.accepts_recovery(accepted, current_state, committed_state):
                    accepted = None
                    selected_source = 'HA'
                    output_box = ha_box
                    if not self.e27_state_quarantine:
                        fallback_box = None
                        if self.e29_reject_fallback != 'ha':
                            fallback_box = cache.fallback_box(self.e29_reject_fallback)
                        cache.set_search_state(fallback_box if fallback_box else ha_box)

            if accepted is not None:
                selected_source = accepted['source']
                out_box = list(accepted['bbox'])
                if self.e27_reset_policy == 'C' and cache.last_reliable is not None:
                    cx = out_box[0] + 0.5 * out_box[2]
                    cy = out_box[1] + 0.5 * out_box[3]
                    out_box[2] = 0.7 * cache.last_reliable.box[2] + 0.3 * out_box[2]
                    out_box[3] = 0.7 * cache.last_reliable.box[3] + 0.3 * out_box[3]
                    out_box[0] = cx - 0.5 * out_box[2]
                    out_box[1] = cy - 0.5 * out_box[3]
                output_box = clip_box(out_box, H, W, margin=10)
                cache.set_search_state(output_box)
                recovered_state = cache.make_state(
                    self.frame_id, output_box, accepted['score'], accepted['score'],
                    accepted['apce'], accepted['margin'], selected_source,
                    edge_risk=accepted.get('edge_risk', 0.0),
                    raw_ha_gap=0.0,
                    reliability_score=self.e27_policy.reliability_score(
                        accepted['score'], accepted['apce'], accepted['margin'],
                        accepted.get('edge_risk', 0.0), 0.0, age=0))
                if self.e27_reset_policy == 'A':
                    cache.update_reliable(recovered_state)
                elif self.e27_reset_policy == 'B':
                    rollback_box = cache.prev_search_state if cache.prev_search_state is not None else ha_box
                    cache.start_pending_recovery(
                        recovered_state, required_frames=1,
                        rollback_box=rollback_box)
                cache.cooldown_left = self.e27_cooldown
            else:
                # accepted was None - use default ha_box fallback
                if not self.e27_state_quarantine:
                    fallback_box = None
                    if self.e29_reject_fallback != 'ha':
                        fallback_box = cache.fallback_box(self.e29_reject_fallback)
                    cache.set_search_state(fallback_box if fallback_box else ha_box)

        if reliable:
            self.e27_policy.update_reliable_stats(
                raw_stats['score'], raw_stats['apce'], raw_stats['margin'])

        # --- Build diagnostic dict ---
        anchor_rank = None
        anchor_age = None
        if accepted is not None:
            # Find accepted candidate's bank rank
            sorted_bank = sorted(cache.bank, key=state_reliability, reverse=True)
            accepted_source = accepted.get('source', '')
            for rank_idx, st in enumerate(sorted_bank):
                if st.source == accepted_source:
                    anchor_rank = rank_idx + 1
                    break
            # Find accepted state's age
            for st in cache.bank:
                if st.source == accepted_source:
                    anchor_age = self.frame_id - st.frame_id
                    break

        # Update calibration engine with this frame's info
        cal_diag = {}
        if self.calibration_engine is not None:
            self.calibration_engine.update(
                score=raw_stats['score'],
                apce=raw_stats['apce'],
                margin=raw_stats['margin'],
                is_reliable=(trigger_type == 'normal'),
                is_triggered=(trigger_type != 'normal'),
                trigger_type=trigger_type if trigger_type != 'normal' else "")
            cal_diag = self.calibration_engine.get_diag_info()

        diag = {
            'trigger_type': trigger_type,
            'num_candidates': candidate_count,
            'accepted': accepted is not None,
            'accepted_candidate_source': selected_source if accepted is not None else None,
            'anchor_rank': anchor_rank,
            'anchor_age': anchor_age,
            'extra_forward_count': candidate_count if trigger_type != 'normal' else 0,
            'bank_size': len(cache.bank),
            'bank_write': bool(cache.pending_recovery is not None and self.e29_delayed_bank_write and reliable and cache.pending_recovery_good >= max(1, int(cache.pending_recovery_required))),
            'score': raw_stats['score'],
            'apce': raw_stats['apce'],
            'margin': raw_stats['margin'],
            'relverify_rejected': (accepted is None and trigger_type != 'normal'
                                   and self.reset_commitment is not None
                                   and selected_source == 'HA'),
            # ReSeT-Cal calibration diagnostics
            'calibration_mode': cal_diag.get('calibration_mode', 'fixed'),
            'score_thr_hard': cal_diag.get('score_thr_hard'),
            'score_thr_reliable': cal_diag.get('score_thr_reliable', 0.65),
            'apce_thr_reliable': cal_diag.get('apce_thr_reliable', 80.0),
            'score_thr_streak': cal_diag.get('score_thr_streak', 0.35),
            'long_unreliable_threshold': cal_diag.get('long_unreliable_threshold', 10),
            'score_thr_uncertain': cal_diag.get('score_thr_uncertain'),
            'apce_thr': cal_diag.get('apce_thr'),
            'num_reliable_samples': cal_diag.get('num_reliable_samples'),
            'warmup_done': cal_diag.get('warmup_done'),
            'q_hard_current': cal_diag.get('q_hard_current'),
            'q_uncertain_current': cal_diag.get('q_uncertain_current'),
            'q_apce_current': cal_diag.get('q_apce_current'),
            'trigger_rate_window': cal_diag.get('trigger_rate_window'),
            'calibration_update_flag': cal_diag.get('calibration_update_flag'),
            'sotd_aggressive': cal_diag.get('sotd_aggressive', False),
            'sotd_consecutive_high': cal_diag.get('sotd_consecutive_high', 0),
            'sotd_consecutive_low': cal_diag.get('sotd_consecutive_low', 0),
            'sotd_window_density': cal_diag.get('sotd_window_density', 0.0),
        }

        return output_box, diag

    def track(self, image, info: dict = None):
        _t0 = time.time()
        H, W, _ = image.shape
        self.frame_id += 1

        # Use SSRM search state if available, otherwise use self.state
        if self.e27_cache.search_state is not None:
            search_state = list(self.e27_cache.search_state)
        else:
            search_state = list(self.state)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, search_state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # --- ReSeT/SSRM post-processing ---
        raw_state_before_post = list(self.state)
        raw_stats = self._e27_branch_stats(pred_score_map)

        ha_state_before_e27 = list(self.state)
        prev_box = raw_state_before_post
        raw_cx = raw_state_before_post[0] + 0.5 * raw_state_before_post[2]
        raw_cy = raw_state_before_post[1] + 0.5 * raw_state_before_post[3]
        prev_cx = prev_box[0] + 0.5 * prev_box[2]
        prev_cy = prev_box[1] + 0.5 * prev_box[3]
        raw_displacement = ((raw_cx - prev_cx) ** 2 + (raw_cy - prev_cy) ** 2) ** 0.5
        ha_cx = ha_state_before_e27[0] + 0.5 * ha_state_before_e27[2]
        ha_cy = ha_state_before_e27[1] + 0.5 * ha_state_before_e27[3]
        raw_vs_ha_gap = ((raw_cx - ha_cx) ** 2 + (raw_cy - ha_cy) ** 2) ** 0.5
        raw_score_max = pred_score_map.max().item()
        raw_is_lost = raw_score_max < self.conf_thresh

        e27_output, e27_diag = self._e27_finalize(
            image, H, W, raw_state_before_post, ha_state_before_e27,
            raw_stats, raw_stats['score'], raw_is_lost,
            raw_displacement, raw_vs_ha_gap)
        self.state = e27_output

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            result = {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            result = {"target_bbox": self.state}

        if self.dump_frames and e27_diag is not None:
            _t1 = time.time()
            frame_diag = {
                'video_id': info.get('seq_name', '') if info else '',
                'frame_id': self.frame_id,
                'bbox': [round(v, 2) for v in self.state],
                'score': round(e27_diag.get('score', 0.0), 6),
                'apce': round(e27_diag.get('apce', 0.0), 4),
                'margin': round(e27_diag.get('margin', 0.0), 6),
                'runtime': round(_t1 - _t0, 6),
                'triggered': e27_diag.get('trigger_type', 'normal') != 'normal',
                'trigger_type': e27_diag.get('trigger_type', 'normal'),
                'num_candidates': e27_diag.get('num_candidates', 0),
                'accepted': e27_diag.get('accepted', False),
                'accepted_candidate_source': e27_diag.get('accepted_candidate_source'),
                'anchor_rank': e27_diag.get('anchor_rank'),
                'anchor_age': e27_diag.get('anchor_age'),
                'extra_forward_count': e27_diag.get('extra_forward_count', 0),
                'bank_size': e27_diag.get('bank_size', 0),
                'bank_write': e27_diag.get('bank_write', False),
                'relverify_rejected': e27_diag.get('relverify_rejected', False),
                # ReSeT-Cal calibration fields
                'calibration_mode': e27_diag.get('calibration_mode', 'fixed'),
                'score_thr_hard': e27_diag.get('score_thr_hard'),
                'score_thr_reliable': e27_diag.get('score_thr_reliable', 0.65),
                'apce_thr_reliable': e27_diag.get('apce_thr_reliable', 80.0),
                'score_thr_streak': e27_diag.get('score_thr_streak', 0.35),
                'long_unreliable_threshold': e27_diag.get('long_unreliable_threshold', 10),
                'score_thr_uncertain': e27_diag.get('score_thr_uncertain'),
                'apce_thr': e27_diag.get('apce_thr'),
                'num_reliable_samples': e27_diag.get('num_reliable_samples'),
                'warmup_done': e27_diag.get('warmup_done'),
                'q_hard_current': e27_diag.get('q_hard_current'),
                'q_uncertain_current': e27_diag.get('q_uncertain_current'),
                'q_apce_current': e27_diag.get('q_apce_current'),
                'trigger_rate_window': e27_diag.get('trigger_rate_window'),
                'calibration_update_flag': e27_diag.get('calibration_update_flag'),
                'sotd_aggressive': e27_diag.get('sotd_aggressive', False),
                'sotd_consecutive_high': e27_diag.get('sotd_consecutive_high', 0),
                'sotd_consecutive_low': e27_diag.get('sotd_consecutive_low', 0),
                'sotd_window_density': e27_diag.get('sotd_window_density', 0.0),
            }
            result['frame_diag'] = frame_diag

        return result

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrackReSeT
"""Tracker adapter for ReSeT search-state management.

The adapter runs the ordinary local forward first, invokes bounded alternative
supports only when the observation gate fails, routes an accepted current-frame
prediction immediately, and delays reusable-bank trust as described in the
main-paper Method section and Supplement Algorithm S1.
"""
