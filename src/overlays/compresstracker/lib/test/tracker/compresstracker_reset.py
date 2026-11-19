"""Tracker adapter for ReSeT search-state management.

Runs the ordinary local forward first, invokes bounded alternative supports
only when the observation gate fails, routes an accepted current-frame
prediction immediately, and delays reusable-bank trust as described in the
main-paper Method section and Supplement Algorithm S1.
"""

import math
import time

from lib.models.compresstracker import build_compresstracker_test
from lib.test.tracker.basetracker import BaseTracker
import torch
import numpy as np

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.diag_logger import DiagFrame
from lib.test.tracker.reset_core.adapter_api import ReSeTAdapterMixin
from lib.test.tracker.reset_core.search_state_commitment import SearchStateCommitment, candidate_reliability, state_reliability


class CompressTrackerReSeT(ReSeTAdapterMixin, BaseTracker):
    def __init__(self, params, dataset_name):
        super(CompressTrackerReSeT, self).__init__(params)
        network = build_compresstracker_test(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.network.set_bernoulli(1.0)

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
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        # --- E29-C3 SSRM / ReSeT parameters ---
        self.e27_mode = getattr(params, 'e27_mode', 'C1')
        self.use_e27 = self.e27_mode is not None and self.e27_mode != 'none'
        self.e27_bank_size = getattr(params, 'e27_bank_size', 10)
        self.e27_bank_topk = getattr(params, 'e27_bank_topk', 3)
        self.e27_use_offsets = getattr(params, 'e27_use_offsets', True)
        self.e27_offset_topk = getattr(params, 'e27_offset_topk', 1)
        self.e27_max_candidates = getattr(params, 'e27_max_candidates', 10)
        self.e27_delta_score = getattr(params, 'e27_delta_score', 0.05)
        self.e27_margin_thresh = getattr(params, 'e27_margin_thresh', 0.08)
        self.e27_strict_delta_score = getattr(params, 'e27_strict_delta_score', 0.08)
        self.e27_strict_margin_thresh = getattr(params, 'e27_strict_margin_thresh', 0.12)
        self.e27_bank23_delta_score = getattr(params, 'e27_bank23_delta_score', None)
        self.e27_bank23_margin_thresh = getattr(params, 'e27_bank23_margin_thresh', None)
        self.e29_c1_fallback_only = getattr(params, 'e29_c1_fallback_only', False)
        self.e29_trigger_budget_rate = getattr(params, 'e29_trigger_budget_rate', 0.0)
        self.e29_relative_gray_min_frames = getattr(params, 'e29_relative_gray_min_frames', 0)
        self.e29_relative_high_score_protect = getattr(params, 'e29_relative_high_score_protect', 1.0)
        self.e29_reject_fallback = getattr(params, 'e29_reject_fallback', 'ha')
        self.e29_delayed_bank_write = getattr(params, 'e29_delayed_bank_write', False)
        self.e30_confirmation_frames = getattr(params, 'e30_confirmation_frames', 0)
        self.e30_source_aware_confirmation = getattr(params, 'e30_source_aware_confirmation', False)
        self.e30_confirm_frames_c1 = getattr(params, 'e30_confirm_frames_c1', 2)
        self.e30_confirm_frames_bank23 = getattr(params, 'e30_confirm_frames_bank23', 2)
        self.e30_confirm_frames_offset = getattr(params, 'e30_confirm_frames_offset', 1)
        self.e30_rollback = getattr(params, 'e30_rollback', False)
        self.e30_rollback_apce_ratio = getattr(params, 'e30_rollback_apce_ratio', 0.70)
        self.e30_rollback_margin_thresh = getattr(params, 'e30_rollback_margin_thresh', 0.04)
        self.e30_rollback_score_thresh = getattr(params, 'e30_rollback_score_thresh', 0.35)
        self.e27_apce_low = getattr(params, 'e27_apce_low', 80.0)
        self.e27_trigger_policy = getattr(params, 'e27_trigger_policy', 'legacy')
        self.e27_dynamic_history = getattr(params, 'e27_dynamic_history', 80)
        self.e27_dynamic_min_frames = getattr(params, 'e27_dynamic_min_frames', 20)
        self.e27_score_low_q = getattr(params, 'e27_score_low_q', 0.20)
        self.e27_score_gray_q = getattr(params, 'e27_score_gray_q', 0.35)
        self.e27_apce_low_q = getattr(params, 'e27_apce_low_q', 0.25)
        self.e27_bank_ranker = getattr(params, 'e27_bank_ranker', 'legacy')
        self.e27_local_neighborhood = getattr(params, 'e27_local_neighborhood', 'legacy')
        self.e27_state_quarantine = getattr(params, 'e27_state_quarantine', False)
        self.e27_reset_confidence = getattr(params, 'e27_reset_confidence', False)
        self.e27_reset_confidence_thresh = getattr(params, 'e27_reset_confidence_thresh', 0.12)
        self.e27_fallback = getattr(params, 'e27_fallback', 'last_high_conf')
        self.e27_reset_policy = getattr(params, 'e27_reset_policy', 'B')
        self.e27_cooldown = getattr(params, 'e27_cooldown', 0)
        self.e27_long_unreliable = getattr(params, 'e27_long_unreliable', 10)
        self.e27_include_sources = getattr(params, 'e27_include_sources', '')
        self.e27_exclude_sources = getattr(params, 'e27_exclude_sources', '')
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
        self.conf_thresh = getattr(params, 'conf_thresh', 0.35)
        self.dump_frames = getattr(params, 'dump_frames', False)
        self._prev_result = {}
        self._prev_score_max = 1.0

        # --- ReSeT-Cal calibration ---
        self.calibration_mode = getattr(params, 'calibration_mode', 'fixed')
        self.calibration_params = getattr(params, 'calibration_params', {})
        self.calibration_engine = None

        if self.use_e27:
            from lib.test.tracker.reset_core.state_cache import E27StateCache
            from lib.test.tracker.reset_core.e27_candidate_generator import E27CandidateGenerator
            from lib.test.tracker.reset_core.e27_local_verifier import E27LocalVerifier
            from lib.test.tracker.reset_core.e27_recovery_policy import E27RecoveryPolicy
            # Create calibration engine if mode is not 'fixed'
            if self.calibration_mode != 'fixed':
                import sys
                cal_engine_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
                cal_engine_dir = os.path.normpath(cal_engine_dir)
                if cal_engine_dir not in sys.path:
                    sys.path.insert(0, cal_engine_dir)
                from calibration_engine import CalibrationEngine
                self.calibration_engine = CalibrationEngine(
                    mode=self.calibration_mode, params=self.calibration_params)
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
                bank23_delta_score=self.e27_bank23_delta_score,
                bank23_margin_thresh=self.e27_bank23_margin_thresh,
                c1_fallback_only=self.e29_c1_fallback_only,
                reset_confidence=self.e27_reset_confidence,
                reset_confidence_thresh=self.e27_reset_confidence_thresh,
                include_sources=[s.strip() for s in str(self.e27_include_sources).split(',')],
                exclude_sources=[s.strip() for s in str(self.e27_exclude_sources).split(',')])
            self.e27_policy = E27RecoveryPolicy(
                mode=self.e27_mode,
                apce_low=self.e27_apce_low,
                margin_low=self.e27_margin_thresh,
                long_unreliable=self.e27_long_unreliable,
                cooldown=self.e27_cooldown,
                trigger_policy=self.e27_trigger_policy,
                dynamic_history=self.e27_dynamic_history,
                dynamic_min_frames=self.e27_dynamic_min_frames,
                score_low_q=self.e27_score_low_q,
                score_gray_q=self.e27_score_gray_q,
                apce_low_q=self.e27_apce_low_q,
                relative_gray_min_frames=self.e29_relative_gray_min_frames,
                relative_high_score_protect=self.e29_relative_high_score_protect,
                calibration_engine=self.calibration_engine)
        else:
            self.e27_cache = None
            self.e27_generator = None
            self.e27_verifier = None
            self.e27_policy = None

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
        if self.e27_cache is not None:
            self.e27_cache.reset(info['init_bbox'])
        if self.e27_policy is not None:
            self.e27_policy.reset()
        if self.calibration_engine is not None:
            self.calibration_engine.reset()
        self.frame_id = 0
        self._prev_score_max = 1.0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _map_box_back_from_center(self, pred_box, resize_factor, search_center):
        cx_prev, cy_prev = search_center
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _forward_action_branch(self, image, search_state, search_factor, use_hann, H, W):
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, search_state, search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=search.tensors,
                ce_template_mask=self.box_mask_z, is_training='test')

        pred_score_map = out_dict['score_map']
        raw_score_max = pred_score_map.max().item()
        response = self.output_window * pred_score_map if use_hann else pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map']).view(-1, 4)
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
            'score_mean': response.mean().item(),
            'resize_factor': resize_factor,
        }

    def _e27_forward_center_candidate(self, image, cand, H, W):
        branch = self._forward_action_branch(
            image, cand['box'], 5.0, False, H, W)
        stats = self._e27_branch_stats(branch['out_dict']['score_map'])
        item = dict(cand)
        item.update({
            'bbox': branch['target_bbox'],
            'score': stats['score'],
            'apce': stats['apce'],
            'margin': stats['margin'],
            'edge_risk': stats['edge_risk'],
            'branch': branch,
        })
        return item

    def _e30_confirmation_required(self, source):
        base = int(self.e30_confirmation_frames)
        if not self.e30_source_aware_confirmation:
            return max(1, base) if base > 0 else 1
        source = str(source or "")
        if source.startswith("C1_last_reliable"):
            return max(1, int(self.e30_confirm_frames_c1))
        if source.startswith("C4_bank_top2") or source.startswith("C4_bank_top3"):
            return max(1, int(self.e30_confirm_frames_bank23))
        if source.startswith("C4_offset_top1"):
            return max(1, int(self.e30_confirm_frames_offset))
        return max(1, base) if base > 0 else 1

    def _e30_rollback_reason(self, raw_stats, current_box, pending_state):
        if pending_state is None:
            return ""
        if float(raw_stats.get('score', 0.0)) < float(self.e30_rollback_score_thresh):
            return "score"
        pending_apce = max(float(pending_state.apce), 1e-6)
        if float(raw_stats.get('apce', 0.0)) < pending_apce * float(self.e30_rollback_apce_ratio):
            return "apce"
        if float(raw_stats.get('margin', 0.0)) < float(self.e30_rollback_margin_thresh):
            return "margin"
        if not self._e27_is_bbox_normal(current_box, pending_state.box):
            return "bbox"
        return ""

    def _e27_finalize(self, image, H, W, raw_box, ha_box, raw_stats, ha_score,
                      raw_is_lost, raw_displacement, raw_vs_ha_gap):
        cache = self.e27_cache
        if cache is None:
            return ha_box, None

        score_low_dyn, score_gray_dyn, apce_low_dyn = self.e27_policy.dynamic_thresholds()
        # When calibration engine is active, use its score_thr_hard as low_score_boundary
        # instead of the fixed 0.35, which is too high for low-scoring trackers like CompressTracker
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
        trigger_budget_applied = False
        if trigger_type != 'normal' and self.e29_trigger_budget_rate > 0.0:
            max_triggers = max(1, int(round(max(self.frame_id, 1) * float(self.e29_trigger_budget_rate))))
            if cache.trigger_count >= max_triggers:
                trigger_type = 'normal'
                trigger_budget_applied = True
        if trigger_type != 'normal':
            cache.trigger_count += 1

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

        e30_commit_applied = False
        e30_rollback_applied = False
        e30_rollback_reason = ""
        if cache.pending_recovery is not None:
            if self.e30_rollback:
                e30_rollback_reason = self._e30_rollback_reason(
                    raw_stats, ha_box, cache.pending_recovery)
                if e30_rollback_reason:
                    rollback_box = cache.pending_recovery_rollback_box
                    if rollback_box is None:
                        rollback_box = cache.fallback_box("last_reliable")
                    if rollback_box is None or len(rollback_box) < 4:
                        rollback_box = ha_box
                    output_box = clip_box(rollback_box, H, W, margin=10)
                    cache.set_search_state(output_box)
                    cache.clear_pending_recovery()
                    e30_rollback_applied = True
            if (not e30_rollback_applied) and self.e29_delayed_bank_write and reliable:
                cache.pending_recovery_good += 1
                if cache.pending_recovery_good >= max(1, int(cache.pending_recovery_required)):
                    cache.update_reliable(cache.pending_recovery)
                    cache.clear_pending_recovery()
                    e30_commit_applied = True

        accepted = None
        candidate_sources = []
        candidate_count = 0
        selected_source = 'HA'
        recovery_candidates = []
        defer_reliable_update = cache.pending_recovery is not None
        _relverify_rejected = False

        if e30_rollback_applied:
            selected_source = 'E30_rollback'
        elif trigger_type == 'normal':
            cache.set_search_state(ha_box)
            if reliable and not defer_reliable_update:
                cache.update_reliable(current_state)
            elif high_conf:
                cache.update_high_conf(current_state)
            output_box = ha_box
        else:
            if self.e27_mode != 'C0':
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
            if accepted is not None:
                # --- RelVerify: check before accepting ---
                if self.reset_commitment is not None:
                    current_state_for_rv = cache.make_state(
                        self.frame_id, ha_box, raw_stats['score'], ha_score,
                        raw_stats['apce'], raw_stats['margin'], 'HA',
                        edge_risk=raw_stats.get('edge_risk', 0.0),
                        raw_ha_gap=raw_vs_ha_gap,
                        reliability_score=current_reliability)
                    committed_state = cache.last_reliable
                    if not self.reset_commitment.accepts_recovery(accepted, current_state_for_rv, committed_state):
                        _relverify_rejected = True
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
                        required_frames = self._e30_confirmation_required(selected_source)
                        rollback_box = cache.prev_search_state if cache.prev_search_state is not None else ha_box
                        cache.start_pending_recovery(
                            recovered_state, required_frames=required_frames,
                            rollback_box=rollback_box)
                    cache.cooldown_left = self.e27_cooldown
            else:
                output_box = ha_box
                if not self.e27_state_quarantine:
                    fallback_box = None
                    if self.e29_reject_fallback != 'ha':
                        fallback_box = cache.fallback_box(self.e29_reject_fallback)
                    cache.set_search_state(fallback_box if fallback_box else ha_box)

        if reliable:
            self.e27_policy.update_reliable_stats(
                raw_stats['score'], raw_stats['apce'], raw_stats['margin'])

        # Update calibration engine with this frame's info
        # Use "not triggered" as reliability proxy instead of tracker's reliable flag,
        # because CompressTracker's reliable threshold (score>0.65, apce>80) is too strict
        # for its own score distribution (median ~0.38)
        cal_diag = {}
        if self.calibration_engine is not None:
            self.calibration_engine.update(
                score=raw_stats['score'],
                apce=raw_stats['apce'],
                margin=raw_stats['margin'],
                is_reliable=(trigger_type == 'normal'),  # non-triggered = reliable for calibration
                is_triggered=(trigger_type != 'normal'),
                trigger_type=trigger_type if trigger_type != 'normal' else "")
            cal_diag = self.calibration_engine.get_diag_info()

        diag = {
            'e27_mode': self.e27_mode,
            'e27_include_sources': list(self.e27_verifier.include_sources) if self.e27_verifier is not None else [],
            'e27_exclude_sources': list(self.e27_verifier.exclude_sources) if self.e27_verifier is not None else [],
            'e27_bank23_delta_score': self.e27_bank23_delta_score,
            'e27_bank23_margin_thresh': self.e27_bank23_margin_thresh,
            'source_specific_rule': (
                'c1_fallback_only' if self.e29_c1_fallback_only else
                'bank23_delta' if self.e27_bank23_delta_score is not None else
                'bank23_margin' if self.e27_bank23_margin_thresh is not None else
                'none'
            ),
            'e27_trigger_policy': self.e27_trigger_policy,
            'trigger_budget_rate': round(float(self.e29_trigger_budget_rate), 6),
            'trigger_budget_applied': bool(trigger_budget_applied),
            'trigger_count': int(cache.trigger_count),
            'fallback_search_state': self.e29_reject_fallback,
            'delayed_bank_write': bool(self.e29_delayed_bank_write),
            'e30_confirmation_frames': int(self.e30_confirmation_frames),
            'e30_source_aware_confirmation': bool(self.e30_source_aware_confirmation),
            'e30_pending_recovery': bool(cache.pending_recovery is not None),
            'e30_pending_good': int(cache.pending_recovery_good),
            'e30_pending_required': int(cache.pending_recovery_required),
            'e30_pending_source': str(cache.pending_recovery_source),
            'e30_commit_applied': bool(e30_commit_applied),
            'e30_rollback_enabled': bool(self.e30_rollback),
            'e30_rollback_applied': bool(e30_rollback_applied),
            'e30_rollback_reason': e30_rollback_reason,
            'relative_gray_min_frames': int(self.e29_relative_gray_min_frames),
            'relative_high_score_protect': round(float(self.e29_relative_high_score_protect), 6),
            'reliability_gate_mode': 'relative' if self.e27_trigger_policy == 'relative_score_apce_gate' else 'abs',
            'dynamic_score_low': round(score_low_dyn, 6),
            'dynamic_score_gray': round(score_gray_dyn, 6),
            'dynamic_apce_low': round(apce_low_dyn, 6),
            'bank_ranker': self.e27_bank_ranker,
            'local_neighborhood_mode': self.e27_local_neighborhood,
            'quarantine_applied': bool(self.e27_state_quarantine and trigger_type != 'normal' and accepted is None),
            'reset_confidence_enabled': bool(self.e27_reset_confidence),
            'trigger_type': trigger_type,
            'candidate_count': candidate_count,
            'candidate_sources': candidate_sources,
            'bank_reliability_score': [
                round(float(c.get('bank_score', 0.0)), 6)
                for c in recovery_candidates
            ],
            'selected_source': selected_source,
            'accepted_recovery': accepted is not None,
            'bank_size': len(cache.bank),
            'frames_since_last_reliable': frames_since,
            'current_reliable_flag': reliable,
            'high_conf_flag': high_conf,
            'current_reliability_score': round(current_reliability, 6),
            'search_state_center': [
                round(cache.search_state[0] + 0.5 * cache.search_state[2], 2),
                round(cache.search_state[1] + 0.5 * cache.search_state[3], 2),
            ] if cache.search_state else None,
            'raw_bbox': [round(v, 2) for v in raw_box],
            'HA_bbox': [round(v, 2) for v in ha_box],
            'output_bbox': [round(v, 2) for v in output_box],
            'raw_APCE': round(raw_stats['apce'], 6),
            'top1_top2_margin': round(raw_stats['margin'], 6),
            'raw_vs_HA_center_gap': round(raw_vs_ha_gap, 4),
            'score': raw_stats['score'],
            'apce': raw_stats['apce'],
            'margin': raw_stats['margin'],
            'anchor_rank': None,
            'anchor_age': None,
            'extra_forward_count': candidate_count if trigger_type != 'normal' else 0,
            'bank_write': bool(cache.pending_recovery is not None and self.e29_delayed_bank_write and reliable and cache.pending_recovery_good >= max(1, int(cache.pending_recovery_required))),
            'relverify_rejected': _relverify_rejected,
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
        # Compute anchor_rank and anchor_age if accepted
        if accepted is not None:
            # anchor_rank: position of accepted candidate's source state in bank sorted by reliability
            try:
                sorted_bank = sorted(cache.bank, key=lambda s: state_reliability(s), reverse=True)
                source_state = accepted.get('source_state')
                if source_state is not None:
                    for rank_idx, s in enumerate(sorted_bank):
                        if s is source_state or (hasattr(s, 'frame_id') and hasattr(source_state, 'frame_id') and s.frame_id == source_state.frame_id):
                            diag['anchor_rank'] = rank_idx
                            break
            except Exception:
                pass
            # anchor_age: frames since the CachedState that generated this candidate
            try:
                source_state = accepted.get('source_state')
                if source_state is not None and hasattr(source_state, 'frame_id'):
                    diag['anchor_age'] = self.frame_id - source_state.frame_id
            except Exception:
                pass
        if recovery_candidates:
            diag['candidate_debug'] = [
                {
                    'source': c['source'],
                    'score': round(float(c.get('score', 0.0)), 6),
                    'apce': round(float(c.get('apce', 0.0)), 4),
                    'margin': round(float(c.get('margin', 0.0)), 6),
                    'edge_risk': round(float(c.get('edge_risk', 0.0)), 4),
                    'bank_reliability_score': round(float(c.get('bank_score', 0.0)), 6),
                    'reset_confidence': round(float(c.get('reset_confidence', 0.0)), 6),
                }
                for c in recovery_candidates
            ]
        return output_box, diag

    def track(self, image, info: dict = None):
        _t0 = time.time()
        H, W, _ = image.shape
        self.frame_id += 1

        if self.e27_cache is not None and self.e27_cache.search_state is not None:
            search_state = list(self.e27_cache.search_state)
        else:
            search_state = list(self.state)

        self._search_center_cx = search_state[0] + 0.5 * search_state[2]
        self._search_center_cy = search_state[1] + 0.5 * search_state[3]

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, search_state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z, is_training='test')

        # add hann windows
        pred_score_map = out_dict['score_map']
        raw_score_max = pred_score_map.max().item()
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        raw_state_before_post = list(self.state)
        raw_stats_e27 = self._e27_branch_stats(pred_score_map) if self.e27_cache is not None else None

        # --- E27/SSRM ReSeT finalize ---
        e27_diag = None
        if self.e27_cache is not None:
            ha_state_before_e27 = list(self.state)
            prev_box = self._prev_result.get('target_bbox', raw_state_before_post) if isinstance(self._prev_result, dict) else raw_state_before_post
            raw_cx = raw_state_before_post[0] + 0.5 * raw_state_before_post[2]
            raw_cy = raw_state_before_post[1] + 0.5 * raw_state_before_post[3]
            prev_cx = prev_box[0] + 0.5 * prev_box[2]
            prev_cy = prev_box[1] + 0.5 * prev_box[3]
            raw_displacement = ((raw_cx - prev_cx) ** 2 + (raw_cy - prev_cy) ** 2) ** 0.5
            ha_cx = ha_state_before_e27[0] + 0.5 * ha_state_before_e27[2]
            ha_cy = ha_state_before_e27[1] + 0.5 * ha_state_before_e27[3]
            raw_vs_ha_gap = ((raw_cx - ha_cx) ** 2 + (raw_cy - ha_cy) ** 2) ** 0.5
            raw_is_lost_for_e27 = raw_score_max < self.conf_thresh
            e27_output, e27_diag = self._e27_finalize(
                image, H, W, raw_state_before_post, ha_state_before_e27,
                raw_stats_e27, raw_stats_e27['score'], raw_is_lost_for_e27,
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
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            result = {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            result = {"target_bbox": self.state}

        _t1 = time.time()
        if self.dump_frames and e27_diag is not None:
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
                'num_candidates': e27_diag.get('candidate_count', 0),
                'accepted': e27_diag.get('accepted_recovery', False),
                'accepted_candidate_source': e27_diag.get('selected_source'),
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

        self._prev_result = result
        self._prev_score_max = raw_score_max

        return result

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev = getattr(self, '_search_center_cx', self.state[0] + 0.5 * self.state[2])
        cy_prev = getattr(self, '_search_center_cy', self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev = getattr(self, '_search_center_cx', self.state[0] + 0.5 * self.state[2])
        cy_prev = getattr(self, '_search_center_cy', self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return CompressTrackerReSeT
"""Tracker adapter for ReSeT search-state management.

The adapter runs the ordinary local forward first, invokes bounded alternative
supports only when the observation gate fails, routes an accepted current-frame
prediction immediately, and delays reusable-bank trust as described in the
main-paper Method section and Supplement Algorithm S1.
"""
