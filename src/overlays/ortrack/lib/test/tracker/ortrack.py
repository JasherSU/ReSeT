import math

from lib.models.ortrack import build_ortrack
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
from lib.test.tracker.action_policy import (
    ACTION_NORMAL,
    ACTION_RECOVERY,
    ACTION_SEVERE,
    action_params,
    bbox_iou_xywh,
    bbox_visible,
    choose_edge_proxy_action,
    choose_action_from_score_head,
    choose_score_severity_action,
    choose_size_conditioned_action,
    choose_score_size_hybrid_action,
    choose_boundary_severe_action,
    edge_risk_from_score_map,
    region_oracle_action,
)
from lib.test.tracker.reset_core.adapter_api import ReSeTAdapterMixin
from lib.test.tracker.reset_core.ortrack_compat import cal_bbox_topk


class ORTrack(ReSeTAdapterMixin, BaseTracker):
    def __init__(self, params, dataset_name):
        super(ORTrack, self).__init__(params)
        network = build_ortrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        # self.network = network.cpu()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.feat_template_sz = self.cfg.TEST.TEMPLATE_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cpu()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.visdom = None
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if self.use_visdom:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.is_distill = params.cfg.MODEL['IS_DISTILL']

        self.dynamic_search = getattr(params, 'dynamic_search', False)
        self.conf_thresh = getattr(params, 'conf_thresh', 0.4)
        self.base_search_factor = params.search_factor
        self.enlarged_search_factor = getattr(params, 'enlarged_search_factor', 6.0)
        self.is_lost = False
        self.hann_control = getattr(params, 'hann_control', False)
        self.dump_frames = getattr(params, 'dump_frames', False)
        self.dump_candidate_maps = getattr(params, 'dump_candidate_maps', False)
        self.dump_candidate_maps_dir = getattr(params, 'dump_candidate_maps_dir', '')
        self.dump_response_maps = getattr(params, 'dump_response_maps', False)
        self.dump_response_maps_dir = getattr(params, 'dump_response_maps_dir', '')
        self.dump_response_frame_ids = set(
            int(v) for v in getattr(params, 'dump_response_frame_ids', [])
        )
        self.fsr_controller = getattr(params, 'fsr_controller', None)
        self.fsr_search_factor_normal = getattr(params, 'fsr_search_factor_normal', 4.0)
        self.fsr_search_factor_search_miss = getattr(params, 'fsr_search_factor_search_miss', 6.0)
        self.fsr_search_factor_edge_visible = getattr(params, 'fsr_search_factor_edge_visible', 5.0)
        self.fsr_search_factor_unreliable = getattr(params, 'fsr_search_factor_unreliable', 5.0)
        self.fsr_hann_normal = getattr(params, 'fsr_hann_normal', True)
        self.fsr_hann_search_miss = getattr(params, 'fsr_hann_search_miss', False)
        self.fsr_hann_edge_visible = getattr(params, 'fsr_hann_edge_visible', False)
        self.fsr_hann_unreliable = getattr(params, 'fsr_hann_unreliable', False)
        self.fsr_score_thresh = getattr(params, 'fsr_score_thresh', 0.4)
        self._prev_result = {}
        self._prev_score_max = 1.0
        self._prev_edge_risk = 0.0

        self.use_sra = getattr(params, 'use_sra', False)
        self.sra_t_logits = getattr(params, 'sra_t_logits', 0.8)
        self.sra_t_score = getattr(params, 'sra_t_score', 0.5)
        self.sra_max_search_factor = getattr(params, 'sra_max_search_factor', 8.0)
        self.sra_enlarge_step = getattr(params, 'sra_enlarge_step', 1.0)
        self.sra_base_factor = params.search_factor

        self.soft_hann = getattr(params, 'soft_hann', None)
        if self.soft_hann:
            from lib.test.tracker.soft_hann_controller import SoftHannController
            self.sh_controller = SoftHannController(
                mode=self.soft_hann, feat_sz=self.feat_sz)
        else:
            self.sh_controller = None

        self.sra_hysteresis = getattr(params, 'sra_hysteresis', False)
        if self.sra_hysteresis:
            from lib.test.tracker.sra_hysteresis_controller import SRAHysteresisController
            self.hyst_controller = SRAHysteresisController(
                tau_enter=getattr(params, 'hyst_enter', 0.40),
                tau_exit=getattr(params, 'hyst_exit', 0.50),
                hysteresis_frames=getattr(params, 'hyst_frames', 2),
                search_normal=self.base_search_factor,
                search_lost=self.enlarged_search_factor)
        else:
            self.hyst_controller = None

        self.kalman_sra = getattr(params, 'kalman_sra', False)
        if self.kalman_sra:
            from lib.test.tracker.kalman_sra_controller import KalmanSRAController
            self.kf_controller = KalmanSRAController(
                conf_thresh=self.conf_thresh,
                base_search_factor=self.base_search_factor,
                enlarged_search_factor=self.enlarged_search_factor,
                process_noise=getattr(params, 'kf_process_noise', 1e-2),
                measurement_noise=getattr(params, 'kf_measurement_noise', 1e-1),
                uncertainty_thresh=getattr(params, 'kf_uncertainty_thresh', 15.0),
                velocity_thresh=getattr(params, 'kf_velocity_thresh', 5.0),
                use_center_predict=getattr(params, 'kf_center_predict', True),
                use_uncertainty_sra=getattr(params, 'kf_uncertainty_sra', True))
        else:
            self.kf_controller = None

        self.use_sf_head = getattr(params, 'use_sf_head', False)
        self.sf_head_sf_min = getattr(params, 'sf_head_sf_min', 4.0)
        self.sf_head_sf_max = getattr(params, 'sf_head_sf_max', 7.0)
        self.use_action_head = getattr(params, 'use_action_head', False)
        self.tau_action = getattr(params, 'tau_action', 0.7)
        self.oracle_action = getattr(params, 'oracle_action', None)
        self.oracle_state = getattr(params, 'oracle_state', False)
        self.oracle_state_diag = getattr(params, 'oracle_state_diag', False)
        self.edge_proxy_sra = getattr(params, 'edge_proxy_sra', False)
        self.tau_edge = getattr(params, 'tau_edge', 0.75)

        self.sra_policy = getattr(params, 'sra_policy', None)
        self.severe_score_thresh = getattr(params, 'severe_score_thresh', 0.20)
        self.severe_search_factor = getattr(params, 'severe_search_factor', 6.0)
        self.size_sf_thresh = getattr(params, 'size_sf_thresh', 60.0)
        self.gray_score_high = getattr(params, 'gray_score_high', 0.50)

        self.use_gmc = getattr(params, 'use_gmc', False)
        if self.use_gmc:
            from lib.test.tracker.gmc_ir_controller import GMC_IR
            self.gmc_controller = GMC_IR(
                downscale=getattr(params, 'gmc_downscale', 2),
                max_shift_ratio=getattr(params, 'gmc_max_shift_ratio', 0.3),
                use_clahe=getattr(params, 'gmc_use_clahe', True),
                use_target_mask=getattr(params, 'gmc_use_target_mask', True),
                mask_padding=getattr(params, 'gmc_mask_padding', 50))
            self.gmc_safety_valve = getattr(params, 'gmc_safety_valve', True)
        else:
            self.gmc_controller = None
            self.gmc_safety_valve = True

        self.topk_oracle = getattr(params, 'topk_oracle', 0)
        self.topk_oracle_diag_only = getattr(params, 'topk_oracle_diag_only', False)
        self.response_gamma = getattr(params, 'response_gamma', 1.0)
        self.heuristic_rerank = getattr(params, 'heuristic_rerank', False)
        self.feature_extraction_mode = getattr(params, 'feature_extraction_mode', False)
        self.ltr_topk = getattr(params, 'ltr_topk', 5)
        self.use_lgbm_reranker = getattr(params, 'use_lgbm_reranker', False)
        self.lgbm_model_path = getattr(params, 'lgbm_model_path', '')
        self.ltr_ambiguous_only = getattr(params, 'ltr_ambiguous_only', True)
        self.ltr_fallback_top1 = getattr(params, 'ltr_fallback_top1', True)
        self.use_tcr_reranker = getattr(params, 'use_tcr_reranker', False)
        self.tcr_model_path = getattr(params, 'tcr_model_path', '')
        self.tcr_gate_path = getattr(params, 'tcr_gate_path', '')
        self.tcr_topk = getattr(params, 'tcr_topk', 5)
        self.tcr_gate_thr = getattr(params, 'tcr_gate_thr', 0.70)
        self.tcr_margin = getattr(params, 'tcr_margin', 0.05)

        self.e27_mode = getattr(params, 'e27_mode', None)
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
        # Optional tracker-relative calibration.
        self.exp13_variant = getattr(params, 'exp13_variant', 'B0')
        self.exp13_normalization = getattr(params, 'exp13_normalization', 'none')
        self.exp13_norm_buffer_size = getattr(params, 'exp13_norm_buffer_size', 500)
        self.exp13_norm_min_samples = getattr(params, 'exp13_norm_min_samples', 20)
        self.exp13_stats_update_policy = getattr(params, 'exp13_stats_update_policy', 'reliable_local')
        self.exp13_commitment_variant = getattr(params, 'exp13_commitment_variant', 'pending')
        self.exp13_enable_diversity = getattr(params, 'exp13_enable_diversity', False)
        self.exp13_enable_support_safety = getattr(params, 'exp13_enable_support_safety', False)
        self.exp13_norm_scope = getattr(params, 'exp13_norm_scope', 'full')
        self.exp13_stats_mode = getattr(params, 'exp13_stats_mode', 'streaming')
        self.exp13_anchor_id_counter = 0
        self.exp13_normalizer = None
        self.exp13_stats_policy = None
        self.exp13_bank_admission = None
        self.exp13_diversity_bank = None
        self.exp13_support_safety = None
        if self.exp13_variant != 'B0':
            from lib.test.tracker.reset_core.calibration.tracker_relative_policy import (
                TrackerRelativeNormalizer, SafeStatsUpdatePolicy)
            from lib.test.tracker.reset_core.calibration.bank_policy import (
                BankAdmissionRetrieval, DiversityBank, SupportSafetyProxy)
            self.exp13_normalizer = TrackerRelativeNormalizer(
                normalization=self.exp13_normalization if self.exp13_normalization != 'none' else 'median_iqr',
                buffer_size=self.exp13_norm_buffer_size,
                min_samples=self.exp13_norm_min_samples,
                stats_mode=self.exp13_stats_mode)
            self.exp13_stats_policy = SafeStatsUpdatePolicy(
                policy=self.exp13_stats_update_policy)
            self.exp13_bank_admission = BankAdmissionRetrieval()
            if self.exp13_enable_diversity:
                self.exp13_diversity_bank = DiversityBank()
            if self.exp13_enable_support_safety:
                self.exp13_support_safety = SupportSafetyProxy()
        self.e27_apce_low = getattr(params, 'e27_apce_low', 80.0)
        self.e27_trigger_policy = getattr(params, 'e27_trigger_policy', 'legacy')
        self.e27_dynamic_history = getattr(params, 'e27_dynamic_history', 80)
        self.e27_dynamic_min_frames = getattr(params, 'e27_dynamic_min_frames', 20)
        self.e27_score_low_q = getattr(params, 'e27_score_low_q', 0.20)
        self.e27_score_gray_q = getattr(params, 'e27_score_gray_q', 0.35)
        self.e27_apce_low_q = getattr(params, 'e27_apce_low_q', 0.25)
        self.e27_bank_ranker = getattr(params, 'e27_bank_ranker', 'legacy')
        self.e27_rank_ablation = getattr(params, 'e27_rank_ablation', 'full')
        self.e27_local_neighborhood = getattr(params, 'e27_local_neighborhood', 'legacy')
        self.e27_verifier_mode = getattr(params, 'e27_verifier_mode', 'full')
        self.e27_state_quarantine = getattr(params, 'e27_state_quarantine', False)
        self.e27_reset_confidence = getattr(params, 'e27_reset_confidence', False)
        self.e27_reset_confidence_thresh = getattr(params, 'e27_reset_confidence_thresh', 0.12)
        self.e27_fallback = getattr(params, 'e27_fallback', 'last_high_conf')
        self.e27_reset_policy = getattr(params, 'e27_reset_policy', 'B')
        self.e27_cooldown = getattr(params, 'e27_cooldown', 0)
        self.e27_long_unreliable = getattr(params, 'e27_long_unreliable', 10)
        self.e27_include_sources = getattr(params, 'e27_include_sources', '')
        self.e27_exclude_sources = getattr(params, 'e27_exclude_sources', '')
        self.use_e31_verifier = getattr(params, 'use_e31_verifier', False)
        self.e31_model_path = getattr(params, 'e31_model_path', '')
        self.e31_prob_thresh = getattr(params, 'e31_prob_thresh', 0.55)
        self.e31_verifier = None
        if self.use_e31_verifier:
            from lib.test.tracker.e31_candidate_verifier import E31CandidateVerifier
            self.e31_verifier = E31CandidateVerifier(self.e31_model_path)
        self.commit_protocol = getattr(params, 'commit_protocol', 'legacy')
        self.commit_gate = getattr(params, 'commit_gate', 'score_apce')
        self.commit_delay_frames = getattr(params, 'commit_delay_frames', 1)
        self.relative_acceptance = getattr(params, 'relative_acceptance', 'none')
        self.anchor_policy = getattr(params, 'anchor_policy', 'last_committed')
        self.commit_oracle = getattr(params, 'commit_oracle', 'none')
        self.reset_commitment = None
        if self.commit_protocol != 'legacy' or self.commit_oracle != 'none':
            from lib.test.tracker.reset_core.search_state_commitment import SearchStateCommitment
            self.reset_commitment = SearchStateCommitment(
                protocol=self.commit_protocol,
                commit_gate=self.commit_gate,
                delay_frames=self.commit_delay_frames,
                relative_acceptance=self.relative_acceptance,
                anchor_policy=self.anchor_policy,
                oracle_mode=self.commit_oracle,
                apce_thresh=self.e27_apce_low,
                margin_thresh=self.e27_margin_thresh)
        if self.use_e27:
            from lib.test.tracker.reset_core.state_cache import E27StateCache
            from lib.test.tracker.reset_core.e27_candidate_generator import E27CandidateGenerator
            from lib.test.tracker.reset_core.e27_local_verifier import E27LocalVerifier
            from lib.test.tracker.reset_core.e27_recovery_policy import E27RecoveryPolicy
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
                verifier_mode=self.e27_verifier_mode,
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
                rank_ablation=self.e27_rank_ablation,
                relative_gray_min_frames=self.e29_relative_gray_min_frames,
                relative_high_score_protect=self.e29_relative_high_score_protect)
        else:
            self.e27_cache = None
            self.e27_generator = None
            self.e27_verifier = None
            self.e27_policy = None
        if self.heuristic_rerank:
            from lib.test.tracker.reranker import HeuristicReranker
            lens_z = self.cfg.MODEL.BACKBONE.TEMPLATE_NUM if hasattr(self.cfg.MODEL.BACKBONE, 'TEMPLATE_NUM') else 64
            self._reranker = HeuristicReranker(
                lambda_cos=getattr(params, 'rerank_lambda', 0.2),
                mu_disp=getattr(params, 'rerank_mu', 0.1),
                nu_edge=getattr(params, 'rerank_nu', 0.0),
                ambig_low=getattr(params, 'rerank_ambig_low', 0.25),
                ambig_high=getattr(params, 'rerank_ambig_high', 0.45),
                margin_thresh=getattr(params, 'rerank_margin_thresh', 0.05),
                topk=getattr(params, 'rerank_topk', 5),
                lens_z=lens_z)
            self._rerank_topk = getattr(params, 'rerank_topk', 5)
        else:
            self._reranker = None
            self._rerank_topk = 5

        self._lgbm_reranker = None
        if self.use_lgbm_reranker and self.lgbm_model_path:
            from lib.test.tracker.lgbm_reranker import LGBMReranker
            lens_z = self.cfg.MODEL.BACKBONE.TEMPLATE_NUM if hasattr(self.cfg.MODEL.BACKBONE, 'TEMPLATE_NUM') else 64
            self._lgbm_reranker = LGBMReranker(
                model_path=self.lgbm_model_path,
                topk=self.ltr_topk,
                lens_z=lens_z,
                ambiguous_only=self.ltr_ambiguous_only,
                fallback_top1=self.ltr_fallback_top1)

        self._tcr_reranker = None
        if self.use_tcr_reranker and self.tcr_model_path:
            from lib.test.tracker.tcr_reranker import TCRReranker
            lens_z = self.cfg.MODEL.BACKBONE.TEMPLATE_NUM if hasattr(self.cfg.MODEL.BACKBONE, 'TEMPLATE_NUM') else 64
            self._tcr_reranker = TCRReranker(
                model_path=self.tcr_model_path,
                gate_path=self.tcr_gate_path,
                topk=self.tcr_topk,
                lens_z=lens_z,
                gate_thr=self.tcr_gate_thr,
                score_margin=self.tcr_margin)

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
        if self.exp13_normalizer is not None:
            self.exp13_normalizer.reset()
        self.frame_id = 0
        self._prev_score_max = 1.0
        if self.sh_controller is not None:
            self.sh_controller.reset()
        if self.hyst_controller is not None:
            self.hyst_controller.reset()
        if self.kf_controller is not None:
            init_cx = info['init_bbox'][0] + 0.5 * info['init_bbox'][2]
            init_cy = info['init_bbox'][1] + 0.5 * info['init_bbox'][3]
            self.kf_controller.initialize(init_cx, init_cy)
        if self.gmc_controller is not None:
            self.gmc_controller.reset()
        self._prev_edge_risk = 0.0
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
                template=self.z_dict1.tensors, search=search.tensors, is_distill=self.is_distill)

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

    def _oracle_action_from_gt(self, mode, gt_bbox, search_state):
        if mode == 'visibility':
            return ACTION_NORMAL if bbox_visible(gt_bbox) else ACTION_RECOVERY
        if mode == 'region':
            return region_oracle_action(
                gt_bbox, search_state, search_size=self.params.search_size,
                search_factor=self.base_search_factor)
        return ACTION_NORMAL

    def _e27_forward_center_candidate(self, image, cand, H, W):
        branch = self._forward_action_branch(
            image, cand['box'], 5.0, False, H, W)
        stats = self._e27_branch_stats(branch['out_dict']['score_map'])
        item = dict(cand)
        anchor_box = [float(v) for v in cand.get('box', [])]
        anchor_center = [float(v) for v in cand.get('center', [])]
        resize_factor = max(float(branch.get('resize_factor', 1.0)), 1e-6)
        item.update({
            'anchor_box': anchor_box,
            'anchor_center': anchor_center,
            'anchor_crop_half_size': 0.5 * float(self.params.search_size) / resize_factor,
            'bbox': branch['target_bbox'],
            'score': stats['score'],
            'apce': stats['apce'],
            'margin': stats['margin'],
            'edge_risk': stats['edge_risk'],
            'branch': branch,
        })
        if (
            self.dump_candidate_maps
            and self.dump_candidate_maps_dir
            and (not self.dump_response_frame_ids or self.frame_id in self.dump_response_frame_ids)
        ):
            score_map = branch['out_dict']['score_map']
            seq_id = getattr(self, '_dump_seq_name', 'unknown')
            frame_id = self.frame_id
            source = str(cand.get('source', 'unknown'))
            safe_source = source.replace('/', '_').replace('\\', '_')
            fname = f"{seq_id}_f{frame_id:06d}_{safe_source}.npz"
            fpath = os.path.join(self.dump_candidate_maps_dir, fname)
            np.savez_compressed(
                fpath,
                score_map=score_map.detach().cpu().float().numpy(),
                bbox=np.array(item['bbox'], dtype=np.float32),
            )
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

    def _exp13_update_reliable(self, cache, state, exp13_norm_cues=None):
        """cache.update_reliable wrapper with B2/B3 bank admission checks."""
        skip_bank_append = False
        if self.exp13_variant in ('B2', 'B3', 'B4', 'B5') and self.exp13_bank_admission is not None:
            normalized_cues = {}
            if exp13_norm_cues is not None:
                normalized_cues = {
                    'score': exp13_norm_cues.get('score'),
                    'apce': exp13_norm_cues.get('apce'),
                    'margin': exp13_norm_cues.get('margin'),
                }
            r_write_val = self.exp13_bank_admission.r_write(
                normalized_cues, True, state.reliability_score)
            state.r_write_score = r_write_val
            if self.exp13_diversity_bank is not None:
                for bank_state in cache.bank:
                    if self.exp13_diversity_bank.is_near_duplicate(
                        state.box, state.frame_id, bank_state.box, bank_state.frame_id):
                        if r_write_val <= bank_state.r_write_score:
                            skip_bank_append = True
                        break
        cache.update_reliable(state, skip_bank_append=skip_bank_append)

    def _e27_finalize(self, image, H, W, raw_box, ha_box, raw_stats, ha_score,
                      raw_is_lost, raw_displacement, raw_vs_ha_gap, gt_bbox=None):
        cache = self.e27_cache
        if cache is None:
            return ha_box, None

        exp13_support_safety_val = None

        if cache.cooldown_left > 0:
            cache.cooldown_left -= 1

        # Pre-compute normalized cues for optional calibration.
        exp13_norm_cues = None
        exp13_norm_thresholds = None
        exp13_use_trigger = False
        exp13_use_ranking = False
        exp13_use_write = False
        exp13_use_accept = False
        exp13_write_thresh = None
        exp13_accept_delta = None
        exp13_accept_margin = None
        if self.exp13_variant != 'B0' and self.exp13_normalizer is not None:
            scope = self.exp13_norm_scope
            exp13_use_ranking = True
            exp13_use_trigger = scope in ('ranking_trigger', 'ranking_trigger_write', 'full')
            exp13_use_write = scope in ('ranking_trigger_write', 'full')
            exp13_use_accept = scope == 'full'
            _cues = {
                'score': raw_stats['score'],
                'apce': raw_stats['apce'],
                'margin': raw_stats['margin'],
                'edge_risk': raw_stats.get('edge_risk', 0.0),
                'disagreement': raw_vs_ha_gap,
            }
            exp13_norm_cues = self.exp13_normalizer.normalize_all(_cues)
            exp13_norm_thresholds = self.exp13_normalizer.trigger_thresholds()
            if exp13_use_write:
                exp13_write_thresh = self.exp13_normalizer.write_threshold()
            if exp13_use_accept and self.exp13_normalizer.ready():
                _iqr = self.exp13_normalizer._stats['score'].iqr()
                exp13_accept_delta = max(
                    self.e27_verifier.delta_score,
                    self.exp13_normalizer.acceptance_delta() * max(_iqr, 1e-6))
                _margin_iqr = self.exp13_normalizer._stats['margin'].iqr()
                exp13_accept_margin = max(
                    self.e27_verifier.margin_thresh,
                    0.5 * max(_margin_iqr, 1e-6))

        frames_since = cache.frames_since_last_reliable(self.frame_id)

        # Update the low-score streak using the active calibration.
        score_low_dyn, score_gray_dyn, apce_low_dyn = self.e27_policy.dynamic_thresholds()
        if exp13_use_trigger and exp13_norm_cues.get('score') is not None:
            _ns_low = exp13_norm_thresholds[0]
            if exp13_norm_cues['score'] < _ns_low:
                cache.low_score_streak += 1
            else:
                cache.low_score_streak = 0
        else:
            low_score_boundary = score_low_dyn if self.e27_trigger_policy == 'relative_score_apce_gate' else 0.35
            if raw_stats['score'] < low_score_boundary:
                cache.low_score_streak += 1
            else:
                cache.low_score_streak = 0

        trigger_type = self.e27_policy.classify_trigger(
            raw_stats['score'], raw_stats['apce'], raw_stats['margin'],
            raw_displacement, raw_vs_ha_gap, frames_since,
            cache.low_score_streak, cache.cooldown_left,
            exp13_norm_cues=(exp13_norm_cues if exp13_use_trigger and exp13_norm_cues.get('score') is not None else None),
            exp13_norm_thresholds=(exp13_norm_thresholds if exp13_use_trigger and exp13_norm_cues.get('score') is not None else None))
        trigger_budget_applied = False
        if trigger_type != 'normal' and self.e29_trigger_budget_rate > 0.0:
            max_triggers = max(1, int(round(max(self.frame_id, 1) * float(self.e29_trigger_budget_rate))))
            if cache.trigger_count >= max_triggers:
                trigger_type = 'normal'
                trigger_budget_applied = True
        if trigger_type != 'normal':
            cache.trigger_count += 1

        ref_for_reliable = cache.last_reliable.box if cache.last_reliable is not None else raw_box
        if exp13_use_write and exp13_norm_cues is not None and exp13_norm_cues.get('score') is not None:
            reliable = (
                exp13_norm_cues['score'] >= exp13_write_thresh
                and raw_stats['apce'] > self.e27_apce_low
                and raw_stats['margin'] > 0.12
                and raw_stats['edge_risk'] < 0.75
                and raw_vs_ha_gap <= max(24.0, 2.5 * max(raw_box[2] * raw_box[3], 1.0) ** 0.5)
                and self._e27_is_bbox_normal(ha_box, ref_for_reliable)
                and not raw_is_lost
            )
        else:
            reliable = (
                raw_stats['score'] > 0.65
                and raw_stats['apce'] > self.e27_apce_low
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
            raw_stats.get('edge_risk', 0.0), raw_vs_ha_gap, age=0,
            exp13_norm_cues=(exp13_norm_cues if exp13_use_ranking else None))
        current_state = cache.make_state(
            self.frame_id, ha_box, raw_stats['score'], ha_score,
            raw_stats['apce'], raw_stats['margin'], 'HA',
            edge_risk=raw_stats.get('edge_risk', 0.0),
            raw_ha_gap=raw_vs_ha_gap,
            reliability_score=current_reliability)
        reset_commit = self.reset_commitment if (
            self.reset_commitment is not None and self.reset_commitment.enabled
        ) else None
        current_gt_iou = None
        if gt_bbox is not None and len(gt_bbox) >= 4 and bbox_visible(gt_bbox):
            current_gt_iou = bbox_iou_xywh(ha_box, gt_bbox)
        commit_gate_pass = (
            reset_commit.commit_gate_passes(current_state, cache.last_reliable)
            if reset_commit is not None else reliable
        )
        reset_decision = None
        reset_anchor_source = ''
        relative_recovery_rejected = False

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
            if (not e30_rollback_applied) and reset_commit is not None:
                reset_decision = reset_commit.pending_commit_decision(
                    cache, commit_gate_pass=commit_gate_pass, reliable=reliable)
                if reset_decision.bank_write_allowed:
                    self._exp13_update_reliable(cache, cache.pending_recovery)
                    cache.clear_pending_recovery()
                    e30_commit_applied = True
            elif (not e30_rollback_applied) and self.e29_delayed_bank_write and reliable:
                cache.pending_recovery_good += 1
                if cache.pending_recovery_good >= max(1, int(cache.pending_recovery_required)):
                    self._exp13_update_reliable(cache, cache.pending_recovery)
                    cache.clear_pending_recovery()
                    e30_commit_applied = True

        accepted = None
        e31_accept_prob = None
        e31_rejected = False
        e31_reject_source = ''
        candidate_sources = []
        candidate_count = 0
        selected_source = 'HA'
        recovery_candidates = []
        defer_reliable_update = cache.pending_recovery is not None

        if e30_rollback_applied:
            selected_source = 'E30_rollback'
        elif trigger_type == 'normal':
            if reset_commit is not None:
                reset_decision = reset_commit.current_observation_decision(
                    current_state, reliable and not defer_reliable_update,
                    high_conf, commit_gate_pass, gt_iou=current_gt_iou)
                if reset_decision.isolated_flag:
                    anchor_box, reset_anchor_source = reset_commit.anchor_box(
                        cache, current_frame=self.frame_id, gt_bbox=gt_bbox)
                    cache.set_search_state(anchor_box if anchor_box else ha_box)
                    cache.set_isolated(current_state)
                else:
                    cache.set_search_state(ha_box)
                    if reset_decision.bank_write_allowed:
                        self._exp13_update_reliable(cache, current_state, exp13_norm_cues)
                    elif high_conf:
                        cache.update_high_conf(current_state)
                    else:
                        cache.set_tentative_state(current_state)
            else:
                cache.set_search_state(ha_box)
                if reliable and not defer_reliable_update:
                    self._exp13_update_reliable(cache, current_state, exp13_norm_cues)
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
                    last_rel_box, trigger_type, crop_side,
                    exp13_delta_override=exp13_accept_delta,
                    exp13_margin_override=exp13_accept_margin)
                if accepted is not None and self.e31_verifier is not None:
                    verifier_row = {
                        'selected_source': accepted.get('source', ''),
                        'trigger_type': trigger_type,
                        'raw_score': raw_stats.get('score', 0.0),
                        'raw_apce': raw_stats.get('apce', 0.0),
                        'raw_margin': raw_stats.get('margin', 0.0),
                        'edge_risk': raw_stats.get('edge_risk', 0.0),
                        'frames_since_last_reliable': frames_since,
                        'bank_size': len(cache.bank),
                        'candidate_count': candidate_count,
                        'candidate_score': accepted.get('score', 0.0),
                        'candidate_apce': accepted.get('apce', 0.0),
                        'candidate_margin': accepted.get('margin', 0.0),
                        'candidate_edge_risk': accepted.get('edge_risk', 0.0),
                        'candidate_reset_confidence': accepted.get('reset_confidence', 0.0),
                        'candidate_bank_reliability_score': accepted.get('bank_score', 0.0),
                        'current_reliability_score': current_reliability,
                    }
                    e31_accept_prob = self.e31_verifier.predict_accept_prob(verifier_row)
                    if e31_accept_prob < float(self.e31_prob_thresh):
                        e31_rejected = True
                        e31_reject_source = str(accepted.get('source', ''))
                        accepted = None
                # B4: Support-aware candidate selection
                if accepted is not None and self.exp13_support_safety is not None:
                    crop_cx = accepted.get('anchor_center', [0, 0])[0]
                    crop_cy = accepted.get('anchor_center', [0, 0])[1]
                    crop_hs = accepted.get('anchor_crop_half_size', 0.0)
                    committed_box = cache.last_reliable.box if cache.last_reliable is not None else None
                    ss = self.exp13_support_safety.compute(
                        candidate_box=list(accepted.get('bbox', [])),
                        crop_center=(crop_cx, crop_cy),
                        crop_half_size=crop_hs,
                        committed_box=committed_box,
                        anchor_box=accepted.get('anchor_box'))
                    exp13_support_safety_val = ss['support_safety']
                    if self.exp13_variant in ('B4', 'B5') and ss['support_safety'] < 0.2:
                        accepted = None
                        e31_rejected = True
                        e31_reject_source = 'B4_support_safety'
                if accepted is not None and reset_commit is not None:
                    if not reset_commit.accepts_recovery(
                            accepted, current_state, cache.last_reliable):
                        relative_recovery_rejected = True
                        accepted = None
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
                recovered_state = cache.make_state(
                    self.frame_id, output_box, accepted['score'], accepted['score'],
                    accepted['apce'], accepted['margin'], selected_source,
                    edge_risk=accepted.get('edge_risk', 0.0),
                    raw_ha_gap=0.0,
                    reliability_score=self.e27_policy.reliability_score(
                        accepted['score'], accepted['apce'], accepted['margin'],
                        accepted.get('edge_risk', 0.0), 0.0, age=0,
                        exp13_norm_cues=(exp13_norm_cues if exp13_use_ranking else None)))
                if reset_commit is not None:
                    required_frames = self._e30_confirmation_required(selected_source)
                    reset_decision = reset_commit.recovery_decision(
                        recovered_state, required_frames=required_frames)
                    if reset_decision.bank_write_allowed:
                        cache.update_reliable(recovered_state)
                        cache.set_search_state(output_box)
                    else:
                        rollback_box = cache.prev_search_state if cache.prev_search_state is not None else ha_box
                        cache.start_pending_recovery(
                            recovered_state,
                            required_frames=max(int(self.commit_delay_frames), int(required_frames)),
                            rollback_box=rollback_box)
                        if self.commit_protocol == 'three_state':
                            anchor_box, reset_anchor_source = reset_commit.anchor_box(
                                cache, current_frame=self.frame_id, gt_bbox=gt_bbox)
                            cache.set_search_state(anchor_box if anchor_box else output_box)
                        else:
                            cache.set_search_state(output_box)
                elif self.e27_reset_policy == 'A':
                    cache.set_search_state(output_box)
                    cache.update_reliable(recovered_state)
                elif self.e27_reset_policy == 'B':
                    cache.set_search_state(output_box)
                    required_frames = self._e30_confirmation_required(selected_source)
                    rollback_box = cache.prev_search_state if cache.prev_search_state is not None else ha_box
                    cache.start_pending_recovery(
                        recovered_state, required_frames=required_frames,
                        rollback_box=rollback_box)
                cache.cooldown_left = self.e27_cooldown
            else:
                output_box = ha_box
                if reset_commit is not None and self.commit_protocol == 'three_state':
                    anchor_box, reset_anchor_source = reset_commit.anchor_box(
                        cache, current_frame=self.frame_id, gt_bbox=gt_bbox)
                    cache.set_search_state(anchor_box if anchor_box else ha_box)
                    cache.set_isolated(current_state)
                    reset_decision = reset_commit.current_observation_decision(
                        current_state, False, high_conf, False, gt_iou=current_gt_iou)
                elif not self.e27_state_quarantine:
                    fallback_box = None
                    if self.e29_reject_fallback != 'ha':
                        fallback_box = cache.fallback_box(self.e29_reject_fallback)
                    cache.set_search_state(fallback_box if fallback_box else ha_box)

        if reliable:
            self.e27_policy.update_reliable_stats(
                raw_stats['score'], raw_stats['apce'], raw_stats['margin'])

        reset_diag = {}
        output_iou_diag = None
        committed_iou_diag = None
        if gt_bbox is not None and len(gt_bbox) >= 4 and bbox_visible(gt_bbox):
            output_iou_diag = bbox_iou_xywh(output_box, gt_bbox)
            if cache.last_reliable is not None:
                committed_iou_diag = bbox_iou_xywh(cache.last_reliable.box, gt_bbox)
        if reset_commit is not None:
            if reset_decision is None:
                reset_decision = reset_commit.current_observation_decision(
                    current_state, reliable and not defer_reliable_update,
                    high_conf, commit_gate_pass, gt_iou=current_gt_iou)
            if reset_anchor_source:
                reset_decision.anchor_source = reset_anchor_source
            if not reset_decision.committed_source and cache.last_reliable is not None:
                reset_decision.committed_source = str(cache.last_reliable.source)
            if not reset_decision.tentative_source and cache.tentative_state is not None:
                reset_decision.tentative_source = str(cache.tentative_state.source)
            reset_diag.update({
                'commit_protocol': self.commit_protocol,
                'commit_gate': self.commit_gate,
                'commit_delay_frames': int(self.commit_delay_frames),
                'relative_acceptance': self.relative_acceptance,
                'anchor_policy': self.anchor_policy,
                'commit_oracle': self.commit_oracle,
                'relative_recovery_rejected': bool(relative_recovery_rejected),
            })
            reset_diag.update(reset_decision.as_diag())
            reset_diag.update(reset_commit.outcome_diag(
                self.frame_id, output_iou_diag, committed_iou_diag,
                reset_decision, accepted_recovery=accepted is not None))

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
            'rank_ablation': self.e27_rank_ablation,
            'local_neighborhood_mode': self.e27_local_neighborhood,
            'verifier_mode': self.e27_verifier_mode,
            'quarantine_applied': bool(self.e27_state_quarantine and trigger_type != 'normal' and accepted is None),
            'reset_confidence_enabled': bool(self.e27_reset_confidence),
            'e31_verifier_enabled': bool(self.e31_verifier is not None),
            'e31_accept_prob': None if e31_accept_prob is None else round(float(e31_accept_prob), 6),
            'e31_prob_thresh': round(float(self.e31_prob_thresh), 6),
            'e31_rejected': bool(e31_rejected),
            'e31_reject_source': e31_reject_source,
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
            'output_iou': None if output_iou_diag is None else round(float(output_iou_diag), 6),
            'committed_iou': None if committed_iou_diag is None else round(float(committed_iou_diag), 6),
            'raw_APCE': round(raw_stats['apce'], 6),
            'top1_top2_margin': round(raw_stats['margin'], 6),
            'raw_vs_HA_center_gap': round(raw_vs_ha_gap, 4),
        }
        # Optional calibration diagnostics.
        diag['exp13_variant'] = self.exp13_variant
        diag['pending_state'] = (
            [round(v, 2) for v in cache.pending_recovery.box]
            if cache.pending_recovery is not None else None
        )
        diag['pending_source_frame'] = (
            int(cache.pending_recovery.frame_id)
            if cache.pending_recovery is not None else None
        )
        diag['confirmation_pass'] = bool(e30_commit_applied) if cache.pending_recovery is not None else None
        diag['bank_write'] = bool(e30_commit_applied or (
            trigger_type == 'normal' and reliable and not defer_reliable_update and (
                (reset_commit is not None and reset_decision is not None and reset_decision.bank_write_allowed) or
                (reset_commit is None and reliable))))
        diag['bank_write_source'] = (
            str(cache.last_reliable.source) if diag['bank_write'] and cache.last_reliable is not None else ''
        )
        diag['stats_update_allowed'] = None
        diag['normalized_score'] = None
        diag['normalized_apce'] = None
        diag['normalized_margin'] = None
        diag['normalized_edge_risk'] = None
        diag['normalized_disagreement'] = None
        diag['support_safety'] = None
        if self.exp13_variant != 'B0' and self.exp13_normalizer is not None:
            is_trigger_frame = trigger_type != 'normal'
            is_recovery_frame = accepted is not None
            is_pending_frame = cache.pending_recovery is not None
            is_hard_loss = trigger_type == 'hard_lost'
            is_high_conf_risk = trigger_type == 'high_conf_risk'
            is_geom_valid = True
            if self.exp13_stats_policy is not None:
                diag['stats_update_allowed'] = self.exp13_stats_policy.allow_update(
                    is_recovery=is_recovery_frame,
                    is_trigger=is_trigger_frame,
                    is_pending_confirmation=is_pending_frame,
                    is_hard_loss=is_hard_loss,
                    is_high_conf_risk=is_high_conf_risk,
                    is_geometry_valid=is_geom_valid,
                    is_reliable=reliable)
            _cues = {
                'score': raw_stats['score'],
                'apce': raw_stats['apce'],
                'margin': raw_stats['margin'],
                'edge_risk': raw_stats.get('edge_risk', 0.0),
                'disagreement': raw_vs_ha_gap,
            }
            if diag['stats_update_allowed']:
                self.exp13_normalizer.update(_cues, update_allowed=True)
            diag['normalized_score'] = (round(float(exp13_norm_cues['score']), 6) if exp13_norm_cues and exp13_norm_cues['score'] is not None else None)
            diag['normalized_apce'] = (round(float(exp13_norm_cues['apce']), 6) if exp13_norm_cues and exp13_norm_cues['apce'] is not None else None)
            diag['normalized_margin'] = (round(float(exp13_norm_cues['margin']), 6) if exp13_norm_cues and exp13_norm_cues['margin'] is not None else None)
            diag['normalized_edge_risk'] = (round(float(exp13_norm_cues['edge_risk']), 6) if exp13_norm_cues and exp13_norm_cues['edge_risk'] is not None else None)
            diag['normalized_disagreement'] = (round(float(exp13_norm_cues['disagreement']), 6) if exp13_norm_cues and exp13_norm_cues['disagreement'] is not None else None)
            diag['exp13_stats_count'] = self.exp13_normalizer.stats_count()
        if exp13_support_safety_val is not None:
            diag['support_safety'] = round(float(exp13_support_safety_val), 6)
        if reset_diag:
            diag.update(reset_diag)
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
                    'anchor_box': [round(float(v), 2) for v in c.get('anchor_box', [])],
                    'anchor_center': [round(float(v), 2) for v in c.get('anchor_center', [])],
                    'anchor_crop_half_size': round(float(c.get('anchor_crop_half_size', 0.0)), 2),
                    'bbox': [round(float(v), 2) for v in c.get('bbox', [])],
                    'iou': (
                        None if gt_bbox is None or len(gt_bbox) < 4 or not bbox_visible(gt_bbox)
                        else round(float(bbox_iou_xywh(c.get('bbox', []), gt_bbox)), 6)
                    ),
                    'candidate_id': idx,
                    'anchor_frame': int(c.get('anchor_frame', -1)),
                    'anchor_age': int(self.frame_id - c.get('anchor_frame', self.frame_id)) if c.get('anchor_frame', -1) >= 0 else None,
                    'anchor_rank': idx,
                    'candidate_crop': [
                        round(float(c.get('anchor_center', [0, 0])[0]), 2),
                        round(float(c.get('anchor_center', [0, 0])[1]), 2),
                        round(float(c.get('anchor_crop_half_size', 0.0)), 2),
                    ],
                    'candidate_box': [round(float(v), 2) for v in c.get('bbox', [])],
                }
                for idx, c in enumerate(recovery_candidates)
            ]
        return output_box, diag

    def _pack_branch_result(self, branch, action, current_search_factor, use_hann, frame_diag=None):
        self.state = branch['target_bbox']
        self.is_lost = action == ACTION_RECOVERY
        out_dict = branch['out_dict']

        if self.kf_controller is not None:
            meas_cx = self.state[0] + 0.5 * self.state[2]
            meas_cy = self.state[1] + 0.5 * self.state[3]
            self.kf_controller.update(meas_cx, meas_cy)

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                branch['pred_boxes'] * self.params.search_size / branch['resize_factor'],
                branch['resize_factor'])
            result = {"target_bbox": self.state, "all_boxes": all_boxes.view(-1).tolist()}
        else:
            result = {"target_bbox": self.state}

        if 'failure_logit' in out_dict:
            logit = out_dict['failure_logit']
            if logit.shape[-1] > 1:
                probs = torch.softmax(logit, dim=-1)
                result['failure_prob'] = probs[0].detach().cpu().tolist()
            else:
                result['failure_prob'] = torch.sigmoid(logit).item()
        if 'logits' in out_dict:
            sra_probs = torch.softmax(out_dict['logits'], dim=-1)
            result['sra_flag'] = sra_probs[0][-1].item()
        if 'sf_pred' in out_dict:
            result['sf_pred'] = out_dict['sf_pred'].item()
            result['sf_cls_logits'] = out_dict['sf_cls_logits']
        if 'action_logits' in out_dict:
            result['action_logits'] = out_dict['action_logits'].detach()
            action_probs = torch.softmax(out_dict['action_logits'], dim=-1)
            result['action_prob'] = action_probs[0].detach().cpu().tolist()

        if frame_diag is not None:
            frame_diag.update({
                'pred_bbox': [round(v, 2) for v in self.state],
                'search_factor': current_search_factor,
                'raw_score_max': round(branch['raw_score_max'], 6),
                'response_score_max': round(branch['response_score_max'], 6),
                'score_mean': round(branch['score_mean'], 6),
                'is_lost': self.is_lost,
                'use_hann': use_hann,
                'action': action,
            })
            if 'action_prob' in result:
                frame_diag['action_prob'] = [round(v, 6) for v in result['action_prob']]
            result['frame_diag'] = frame_diag

        self._prev_result = result
        self._prev_score_max = branch['raw_score_max']
        return result

    def _track_oracle_best_action(self, image, info, search_state, H, W):
        gt_bbox = info.get('gt_bbox') if info is not None else None
        normal_sf, normal_hann = action_params(ACTION_NORMAL)
        recovery_sf, recovery_hann = action_params(ACTION_RECOVERY)
        normal_branch = self._forward_action_branch(image, search_state, normal_sf, normal_hann, H, W)
        recovery_branch = self._forward_action_branch(image, search_state, recovery_sf, recovery_hann, H, W)

        normal_iou = 0.0
        recovery_iou = 0.0
        if gt_bbox is not None and bbox_visible(gt_bbox):
            normal_iou = bbox_iou_xywh(normal_branch['target_bbox'], gt_bbox)
            recovery_iou = bbox_iou_xywh(recovery_branch['target_bbox'], gt_bbox)

        if gt_bbox is not None and not bbox_visible(gt_bbox):
            chosen_action = ACTION_RECOVERY
        elif recovery_iou > normal_iou:
            chosen_action = ACTION_RECOVERY
        else:
            chosen_action = ACTION_NORMAL
        chosen_branch = recovery_branch if chosen_action == ACTION_RECOVERY else normal_branch
        chosen_sf, chosen_hann = action_params(chosen_action)

        frame_diag = None
        if self.dump_frames:
            cx_prev = search_state[0] + 0.5 * search_state[2]
            cy_prev = search_state[1] + 0.5 * search_state[3]
            crop_half = 0.5 * self.params.search_size / chosen_branch['resize_factor']
            frame_diag = {
                'frame_id': self.frame_id,
                'search_center': [round(cx_prev, 2), round(cy_prev, 2)],
                'crop_half_size': round(crop_half, 2),
                'oracle_action': 'best_action',
                'normal_iou': round(normal_iou, 6),
                'recovery_iou': round(recovery_iou, 6),
                'chosen_action': chosen_action,
            }
        return self._pack_branch_result(
            chosen_branch, chosen_action, chosen_sf, chosen_hann, frame_diag=frame_diag)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        current_search_factor = self.base_search_factor
        fsr_state = None
        sra_flag = None
        forced_action = None
        forced_use_hann = None

        if self.kf_controller is not None:
            self.kf_controller.predict()
            kf_center = self.kf_controller.get_predicted_center()
        else:
            kf_center = None

        if self.sh_controller is not None:
            current_search_factor = self.sh_controller.get_search_factor()
            if self.soft_hann == 'h3':
                pass
            else:
                if self.use_sra and 'sra_flag' in self._prev_result:
                    sra_flag = self._prev_result['sra_flag']
                if sra_flag is not None:
                    if sra_flag >= self.sra_t_logits:
                        current_search_factor = min(current_search_factor + self.sra_enlarge_step,
                                                    self.sra_max_search_factor)
                    else:
                        if self._prev_score_max < self.sra_t_score:
                            current_search_factor = min(current_search_factor + self.sra_enlarge_step,
                                                        self.sra_max_search_factor)
                        else:
                            current_search_factor = self.sra_base_factor
                elif self.dynamic_search and self.is_lost:
                    current_search_factor = self.enlarged_search_factor
        elif self.hyst_controller is not None:
            current_search_factor = self.hyst_controller.get_search_factor()
        elif self.kf_controller is not None:
            pass
        elif self.use_action_head and 'action_logits' in self._prev_result:
            action_probs = torch.softmax(self._prev_result['action_logits'], dim=-1)
            p_recovery = action_probs[0][-1].item()
            forced_action = choose_action_from_score_head(
                self._prev_score_max, p_recovery, self.tau_action)
            current_search_factor, forced_use_hann = action_params(forced_action)
        elif self.edge_proxy_sra:
            forced_action = choose_edge_proxy_action(
                self._prev_score_max, self._prev_edge_risk, self.tau_edge)
            current_search_factor, forced_use_hann = action_params(forced_action)
        elif self.sra_policy is not None:
            prev_bbox = self.state
            prev_bbox_area = max(float(prev_bbox[2]) * float(prev_bbox[3]), 1.0)
            if self.sra_policy == 'score_severity':
                forced_action = choose_score_severity_action(
                    self._prev_score_max,
                    severe_thresh=self.severe_score_thresh,
                    recovery_thresh=self.conf_thresh)
                current_search_factor, forced_use_hann = action_params(
                    forced_action, base_sf=self.base_search_factor,
                    recovery_sf=self.enlarged_search_factor,
                    severe_sf=self.severe_search_factor)
            elif self.sra_policy == 'size_conditioned':
                forced_action, current_search_factor = choose_size_conditioned_action(
                    self._prev_score_max, prev_bbox_area,
                    size_sf_thresh=self.size_sf_thresh,
                    recovery_thresh=self.conf_thresh,
                    base_sf=self.base_search_factor,
                    recovery_sf=self.enlarged_search_factor,
                    large_sf=self.severe_search_factor)
                forced_use_hann = False if forced_action != ACTION_NORMAL else True
            elif self.sra_policy == 'score_size_hybrid':
                forced_action, current_search_factor, forced_use_hann = choose_score_size_hybrid_action(
                    self._prev_score_max, prev_bbox_area,
                    severe_thresh=self.severe_score_thresh,
                    size_sf_thresh=self.size_sf_thresh,
                    recovery_thresh=self.conf_thresh,
                    base_sf=self.base_search_factor,
                    recovery_sf=self.enlarged_search_factor,
                    severe_sf=self.severe_search_factor)
            elif self.sra_policy == 'boundary_severe':
                p_recovery = 0.0
                if 'action_prob' in self._prev_result:
                    ap = self._prev_result['action_prob']
                    if isinstance(ap, list) and len(ap) >= 2:
                        p_recovery = ap[-1]
                forced_action, current_search_factor, forced_use_hann = choose_boundary_severe_action(
                    self._prev_score_max, p_recovery, self.tau_action,
                    severe_thresh=self.severe_score_thresh,
                    severe_sf=self.severe_search_factor,
                    recovery_sf=self.enlarged_search_factor,
                    base_sf=self.base_search_factor)
            self.is_lost = forced_action != ACTION_NORMAL
        elif self.use_sf_head and 'sf_pred' in self._prev_result:
            current_search_factor = self._prev_result['sf_pred']
        else:
            if self.use_sra and 'sra_flag' in self._prev_result:
                sra_flag = self._prev_result['sra_flag']
            if self.fsr_controller is not None and 'failure_prob' in self._prev_result:
                probs = self._prev_result['failure_prob']
                if isinstance(probs, list) and len(probs) == 4:
                    fsr_state = probs.index(max(probs))
            if self.fsr_controller == 'fsr_only':
                if fsr_state == 1:
                    current_search_factor = self.fsr_search_factor_search_miss
                elif fsr_state == 2:
                    current_search_factor = self.fsr_search_factor_edge_visible
                elif fsr_state == 3:
                    current_search_factor = self.fsr_search_factor_unreliable
                else:
                    current_search_factor = self.fsr_search_factor_normal
            elif self.fsr_controller == 'score_fsr':
                if self._prev_score_max < self.fsr_score_thresh:
                    if fsr_state == 1:
                        current_search_factor = self.fsr_search_factor_search_miss
                    elif fsr_state == 2:
                        current_search_factor = self.fsr_search_factor_edge_visible
                    elif fsr_state == 3:
                        current_search_factor = self.fsr_search_factor_unreliable
                    else:
                        current_search_factor = self.fsr_search_factor_search_miss
                else:
                    current_search_factor = self.fsr_search_factor_normal
            elif self.dynamic_search and self.is_lost:
                current_search_factor = self.enlarged_search_factor

            if self.use_sra and sra_flag is not None:
                if sra_flag >= self.sra_t_logits:
                    current_search_factor = min(current_search_factor + self.sra_enlarge_step,
                                                self.sra_max_search_factor)
                else:
                    if self._prev_score_max < self.sra_t_score:
                        current_search_factor = min(current_search_factor + self.sra_enlarge_step,
                                                    self.sra_max_search_factor)
                    else:
                        current_search_factor = self.sra_base_factor

        if self.e27_cache is not None and self.e27_cache.search_state is not None:
            search_state = list(self.e27_cache.search_state)
        else:
            search_state = list(self.state)
        if self.kf_controller is not None and kf_center is not None and self.kf_controller.use_center_predict:
            search_state[0] = kf_center[0] - 0.5 * search_state[2]
            search_state[1] = kf_center[1] - 0.5 * search_state[3]

        gmc_warp = None
        if self.gmc_controller is not None:
            max_shift_px = None
            if self.gmc_safety_valve:
                max_shift_px = self.gmc_max_shift_ratio * self.params.search_size / current_search_factor
            gmc_warp = self.gmc_controller.apply(image, target_bbox=self.state, max_shift_px=max_shift_px)
            if gmc_warp is not None and not np.allclose(gmc_warp, np.eye(2, 3)):
                cx = search_state[0] + 0.5 * search_state[2]
                cy = search_state[1] + 0.5 * search_state[3]
                new_cx, new_cy = self.gmc_controller.compensate_center(cx, cy, gmc_warp)
                search_state[0] = new_cx - 0.5 * search_state[2]
                search_state[1] = new_cy - 0.5 * search_state[3]

        normal_search_state = list(search_state)
        oracle_state_applied = False
        oracle_state_reason = 'disabled'
        prev_gt_bbox = info.get('prev_gt_bbox') if info is not None else None
        if self.oracle_state:
            if prev_gt_bbox is not None and len(prev_gt_bbox) >= 4 and bbox_visible(prev_gt_bbox):
                search_state = [float(v) for v in prev_gt_bbox[:4]]
                oracle_state_applied = True
                oracle_state_reason = 'prev_gt'
            else:
                oracle_state_reason = 'prev_gt_missing_or_invisible'

        if self.oracle_action == 'best_action':
            self._search_center_cx = search_state[0] + 0.5 * search_state[2]
            self._search_center_cy = search_state[1] + 0.5 * search_state[3]
            return self._track_oracle_best_action(image, info or {}, search_state, H, W)

        if self.oracle_action in ['visibility', 'region']:
            gt_bbox = info.get('gt_bbox') if info is not None else None
            if gt_bbox is not None:
                forced_action = self._oracle_action_from_gt(self.oracle_action, gt_bbox, search_state)
                current_search_factor, forced_use_hann = action_params(forced_action)

        self._search_center_cx = search_state[0] + 0.5 * search_state[2]
        self._search_center_cy = search_state[1] + 0.5 * search_state[3]

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, search_state, current_search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, is_distill=self.is_distill)

        pred_score_map = out_dict['score_map']
        raw_score_max = pred_score_map.max().item()
        current_edge_risk = edge_risk_from_score_map(pred_score_map)

        if self.sh_controller is not None:
            edge_risk = current_edge_risk
            action = self.sh_controller.update(raw_score_max, edge_risk)
            soft_window = (1 - action.hann_alpha) + action.hann_alpha * self.output_window
            response = soft_window * pred_score_map
            use_hann = None
        else:
            use_hann = True
            if forced_action is not None:
                use_hann = forced_use_hann
            else:
                if self.use_sra and sra_flag is not None:
                    if sra_flag >= self.sra_t_logits:
                        use_hann = False
                    elif self._prev_score_max < self.sra_t_score:
                        use_hann = False
                if self.fsr_controller == 'fsr_only':
                    if fsr_state == 1:
                        use_hann = self.fsr_hann_search_miss
                    elif fsr_state == 2:
                        use_hann = self.fsr_hann_edge_visible
                    elif fsr_state == 3:
                        use_hann = self.fsr_hann_unreliable
                    else:
                        use_hann = self.fsr_hann_normal
                elif self.fsr_controller == 'score_fsr':
                    if self._prev_score_max < self.fsr_score_thresh:
                        if fsr_state == 1:
                            use_hann = self.fsr_hann_search_miss
                        elif fsr_state == 2:
                            use_hann = self.fsr_hann_edge_visible
                        elif fsr_state == 3:
                            use_hann = self.fsr_hann_unreliable
                        else:
                            use_hann = False
                    else:
                        use_hann = self.fsr_hann_normal
                elif self.hann_control and self.dynamic_search and self.is_lost:
                    use_hann = False
                if self.kf_controller is not None and self.kf_controller.should_disable_hann(raw_score_max):
                    use_hann = False
            if use_hann:
                if self.response_gamma != 1.0 and not self.is_lost:
                    response = self.output_window * pred_score_map.pow(self.response_gamma)
                else:
                    response = self.output_window * pred_score_map
            else:
                response = pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        raw_state_before_post = list(self.state)
        if (
            self.dump_response_maps
            and self.dump_response_maps_dir
            and (not self.dump_response_frame_ids or self.frame_id in self.dump_response_frame_ids)
        ):
            os.makedirs(self.dump_response_maps_dir, exist_ok=True)
            seq_id = getattr(self, '_dump_seq_name', 'unknown')
            fpath = os.path.join(
                self.dump_response_maps_dir,
                f"{seq_id}_f{self.frame_id:06d}_ordinary.npz",
            )
            crop_half = 0.5 * float(self.params.search_size) / max(float(resize_factor), 1e-6)
            np.savez_compressed(
                fpath,
                score_map=pred_score_map.detach().cpu().float().numpy(),
                response_map=response.detach().cpu().float().numpy(),
                search_crop=np.asarray(x_patch_arr),
                search_state=np.asarray(search_state, dtype=np.float32),
                search_factor=np.asarray([current_search_factor], dtype=np.float32),
                search_center=np.asarray([
                    search_state[0] + 0.5 * search_state[2],
                    search_state[1] + 0.5 * search_state[3],
                ], dtype=np.float32),
                crop_half_size=np.asarray([crop_half], dtype=np.float32),
                raw_prediction=np.asarray(raw_state_before_post, dtype=np.float32),
                resize_factor=np.asarray([resize_factor], dtype=np.float32),
                use_hann=np.asarray([-1 if use_hann is None else int(bool(use_hann))], dtype=np.int8),
            )
        raw_stats_e27 = self._e27_branch_stats(pred_score_map) if self.e27_cache is not None else None

        topk_diag = None
        topk_oracle_applied = False
        if self.topk_oracle > 0:
            k = self.topk_oracle
            bbox_k, topk_scores = cal_bbox_topk(
                self.network.box_head, pred_score_map,
                out_dict['size_map'], out_dict['offset_map'], k=k)
            topk_scores = topk_scores.view(k)
            bbox_k = bbox_k.view(k, 4)
            candidates_orig = []
            for ki in range(k):
                cand = (bbox_k[ki] * self.params.search_size / resize_factor).tolist()
                cand_mapped = self.map_box_back(cand, resize_factor)
                cand_clipped = clip_box(cand_mapped, H, W, margin=10)
                candidates_orig.append(cand_clipped)
            gt_bbox = info.get('gt_bbox') if info is not None else None
            oracle_idx = 0
            oracle_iou = 0.0
            top1_iou = 0.0
            topk_hit = False
            if gt_bbox is not None and len(gt_bbox) >= 4:
                top1_iou = bbox_iou_xywh(self.state, gt_bbox)
                best_iou = 0.0
                best_dist = float('inf')
                best_dist_idx = 0
                for ki, cand in enumerate(candidates_orig):
                    iou_ki = bbox_iou_xywh(cand, gt_bbox)
                    if iou_ki > best_iou:
                        best_iou = iou_ki
                        oracle_idx = ki
                        oracle_iou = iou_ki
                    cx_c = cand[0] + 0.5 * cand[2]
                    cy_c = cand[1] + 0.5 * cand[3]
                    cx_g = gt_bbox[0] + 0.5 * gt_bbox[2]
                    cy_g = gt_bbox[1] + 0.5 * gt_bbox[3]
                    dist = ((cx_c - cx_g) ** 2 + (cy_c - cy_g) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_dist_idx = ki
                if best_iou <= 0:
                    oracle_idx = best_dist_idx
                topk_hit = oracle_iou > 0.5
            if not self.topk_oracle_diag_only:
                self.state = candidates_orig[oracle_idx]
                topk_oracle_applied = True
            topk_diag = {
                'topk_candidates': [[round(v, 2) for v in c] for c in candidates_orig],
                'topk_scores': [round(s, 6) for s in topk_scores.tolist()],
                'oracle_choice': oracle_idx,
                'oracle_iou': round(oracle_iou, 6),
                'top1_iou': round(top1_iou, 6),
                'topk_hit': topk_hit,
                'topk_oracle_diag_only': bool(self.topk_oracle_diag_only),
                'topk_oracle_applied': bool(topk_oracle_applied),
            }

        hr_diag = None
        ltr_feat_diag = None
        if self._reranker is not None and (
            self.topk_oracle <= 0 or self.topk_oracle_diag_only
        ):
            k = self._rerank_topk
            topk_scores_hr, topk_idx_hr = torch.topk(pred_score_map.flatten(1), k, dim=1)
            top1_score_hr = topk_scores_hr[0, 0].item()
            top2_score_hr = topk_scores_hr[0, 1].item() if k > 1 else top1_score_hr
            if self._reranker.is_ambiguous(self._prev_score_max, top1_score_hr, top2_score_hr):
                bbox_k_hr, _ = cal_bbox_topk(
                    self.network.box_head, pred_score_map,
                    out_dict['size_map'], out_dict['offset_map'], k=k)
                bbox_k_hr = bbox_k_hr.view(k, 4)
                candidates_hr = []
                for ki in range(k):
                    cand = (bbox_k_hr[ki] * self.params.search_size / resize_factor).tolist()
                    cand_mapped = self.map_box_back(cand, resize_factor)
                    cand_clipped = clip_box(cand_mapped, H, W, margin=10)
                    candidates_hr.append(cand_clipped)
                before_norm = out_dict.get('before_norm', None)
                best_idx_hr, final_scores_hr = self._reranker.rerank(
                    topk_scores_hr, topk_idx_hr, candidates_hr, self.state,
                    before_norm, self.feat_sz)
                self.state = candidates_hr[best_idx_hr]
                hr_diag = {
                    'hr_best_idx': best_idx_hr,
                    'hr_final_scores': [round(s, 6) for s in final_scores_hr],
                }

        if self.feature_extraction_mode and (
            self.topk_oracle <= 0 or self.topk_oracle_diag_only
        ):
            k = self.ltr_topk
            topk_scores_ltr, topk_idx_ltr = torch.topk(pred_score_map.flatten(1), k, dim=1)
            top1_sc = topk_scores_ltr[0, 0].item()
            top2_sc = topk_scores_ltr[0, 1].item() if k > 1 else top1_sc
            is_amb = self._reranker.is_ambiguous(self._prev_score_max, top1_sc, top2_sc) if self._reranker is not None else (top1_sc < 0.45)
            bbox_k_ltr, _ = cal_bbox_topk(
                self.network.box_head, pred_score_map,
                out_dict['size_map'], out_dict['offset_map'], k=k)
            bbox_k_ltr = bbox_k_ltr.view(k, 4)
            candidates_ltr = []
            for ki in range(k):
                cand = (bbox_k_ltr[ki] * self.params.search_size / resize_factor).tolist()
                cand_mapped = self.map_box_back(cand, resize_factor)
                cand_clipped = clip_box(cand_mapped, H, W, margin=10)
                candidates_ltr.append(cand_clipped)
            ambig_reason = 'unknown'
            if top1_sc < 0.25:
                ambig_reason = 'low_score'
            elif (top1_sc - top2_sc) < 0.05:
                ambig_reason = 'low_margin'
            elif top1_sc < 0.45 and (top1_sc - top2_sc) < 0.05:
                ambig_reason = 'both'
            ltr_feat_diag = {
                'is_ambiguous': is_amb,
                'candidates': candidates_ltr,
                'topk_raw_scores': topk_scores_ltr[0].tolist(),
                'topk_idx': topk_idx_ltr,
                'pred_score_map': pred_score_map,
                'before_norm': out_dict.get('before_norm', None),
                'prev_state': list(self.state) if not is_amb else list(self._prev_result.get('target_bbox', self.state)),
                'search_center': [float(self._search_center_cx), float(self._search_center_cy)],
                'prev_raw_score': self._prev_score_max,
                'search_size': self.params.search_size,
                'resize_factor': resize_factor,
                'is_lost': self.is_lost,
                'sra_state': 1 if self.is_lost else 0,
                'ambiguous_reason': ambig_reason,
                'feat_sz': self.feat_sz,
            }

        lgbm_diag = None
        if self._lgbm_reranker is not None and (
            self.topk_oracle <= 0 or self.topk_oracle_diag_only
        ) and not self.feature_extraction_mode:
            k = self.ltr_topk
            topk_scores_lgbm, topk_idx_lgbm = torch.topk(pred_score_map.flatten(1), k, dim=1)
            top1_sc_lgbm = topk_scores_lgbm[0, 0].item()
            top2_sc_lgbm = topk_scores_lgbm[0, 1].item() if k > 1 else top1_sc_lgbm
            should_rerank = True
            if self.ltr_ambiguous_only:
                should_rerank = (top1_sc_lgbm < 0.45) or (top1_sc_lgbm - top2_sc_lgbm < 0.05)
            if should_rerank:
                bbox_k_lgbm, _ = cal_bbox_topk(
                    self.network.box_head, pred_score_map,
                    out_dict['size_map'], out_dict['offset_map'], k=k)
                bbox_k_lgbm = bbox_k_lgbm.view(k, 4)
                candidates_lgbm = []
                for ki in range(k):
                    cand = (bbox_k_lgbm[ki] * self.params.search_size / resize_factor).tolist()
                    cand_mapped = self.map_box_back(cand, resize_factor)
                    cand_clipped = clip_box(cand_mapped, H, W, margin=10)
                    candidates_lgbm.append(cand_clipped)
                before_norm_lgbm = out_dict.get('before_norm', None)
                prev_state_lgbm = list(self._prev_result.get('target_bbox', self.state))
                best_idx_lgbm, lgbm_scores = self._lgbm_reranker.rerank(
                    candidates_lgbm, topk_scores_lgbm[0].tolist(), topk_idx_lgbm,
                    pred_score_map, before_norm_lgbm, prev_state_lgbm,
                    self._prev_score_max, self.params.search_size, resize_factor,
                    self.is_lost, 1 if self.is_lost else 0, 'lgbm', self.feat_sz)
                self.state = candidates_lgbm[best_idx_lgbm]
                lgbm_diag = {
                    'lgbm_best_idx': best_idx_lgbm,
                    'lgbm_scores': [round(s, 6) for s in lgbm_scores],
                }

        tcr_diag = None
        if self._tcr_reranker is not None and (
            self.topk_oracle <= 0 or self.topk_oracle_diag_only
        ) and not self.feature_extraction_mode:
            from lib.test.tracker.tcr_reranker import build_runtime_scalar_rows
            k = self.tcr_topk
            topk_scores_tcr, topk_idx_tcr = torch.topk(pred_score_map.flatten(1), k, dim=1)
            top1_sc_tcr = topk_scores_tcr[0, 0].item()
            top2_sc_tcr = topk_scores_tcr[0, 1].item() if k > 1 else top1_sc_tcr
            is_amb_tcr = (top1_sc_tcr < 0.45) or (top1_sc_tcr - top2_sc_tcr < 0.05)
            if is_amb_tcr:
                bbox_k_tcr, _ = cal_bbox_topk(
                    self.network.box_head, pred_score_map,
                    out_dict['size_map'], out_dict['offset_map'], k=k)
                bbox_k_tcr = bbox_k_tcr.view(k, 4)
                candidates_tcr = []
                for ki in range(k):
                    cand = (bbox_k_tcr[ki] * self.params.search_size / resize_factor).tolist()
                    cand_mapped = self.map_box_back(cand, resize_factor)
                    cand_clipped = clip_box(cand_mapped, H, W, margin=10)
                    candidates_tcr.append(cand_clipped)
                prev_state_tcr = list(self._prev_result.get('target_bbox', self.state))
                before_norm_tcr = out_dict.get('before_norm', None)
                scalar_rows = build_runtime_scalar_rows(
                    candidates_tcr, topk_scores_tcr[0].tolist(), topk_idx_tcr,
                    pred_score_map, before_norm_tcr, prev_state_tcr,
                    self._prev_score_max, self.params.search_size, resize_factor,
                    self.is_lost, 1 if self.is_lost else 0, self.feat_sz)
                best_idx_tcr, tcr_scores, gate_prob = self._tcr_reranker.rerank(
                    candidates_tcr, topk_scores_tcr[0].tolist(), topk_idx_tcr,
                    scalar_rows, before_norm_tcr, self.feat_sz,
                    gate_row=scalar_rows[0], is_lost=self.is_lost)
                allow_change = best_idx_tcr != 0
                if top1_sc_tcr > 0.55 and (top1_sc_tcr - top2_sc_tcr) > 0.12:
                    margin = tcr_scores[best_idx_tcr] - tcr_scores[0] if tcr_scores else 0.0
                    allow_change = gate_prob > 0.90 and margin > 0.10
                if allow_change:
                    prev_size = max((prev_state_tcr[2] * prev_state_tcr[3]) ** 0.5, 1.0)
                    cand = candidates_tcr[best_idx_tcr]
                    prev_cx = prev_state_tcr[0] + 0.5 * prev_state_tcr[2]
                    prev_cy = prev_state_tcr[1] + 0.5 * prev_state_tcr[3]
                    cand_cx = cand[0] + 0.5 * cand[2]
                    cand_cy = cand[1] + 0.5 * cand[3]
                    max_disp = max(4.0 * prev_size, 40.0) if self.is_lost else max(2.5 * prev_size, 24.0)
                    disp = ((cand_cx - prev_cx) ** 2 + (cand_cy - prev_cy) ** 2) ** 0.5
                    size_ratio = (cand[2] * cand[3]) / (prev_state_tcr[2] * prev_state_tcr[3]) if prev_state_tcr[2] * prev_state_tcr[3] > 0 else 1.0
                    edge_risk = scalar_rows[best_idx_tcr].get('edge_risk_top1', 0.0)
                    chosen_cos = scalar_rows[best_idx_tcr].get('template_cos', 1.0)
                    if disp > max_disp:
                        allow_change = False
                    if not self.is_lost and edge_risk > 0.95:
                        allow_change = False
                    if not self.is_lost and not (0.5 <= size_ratio <= 2.0):
                        allow_change = False
                    if not self.is_lost and chosen_cos < 0.3:
                        allow_change = False
                if allow_change:
                    self.state = candidates_tcr[best_idx_tcr]
                tcr_diag = {
                    'tcr_best_idx': best_idx_tcr,
                    'tcr_applied': bool(allow_change),
                    'tcr_gate_prob': round(gate_prob, 6),
                    'tcr_scores': [round(s, 6) for s in tcr_scores],
                }

        e27_diag = None
        if self.e27_cache is not None and (
            self.topk_oracle <= 0 or self.topk_oracle_diag_only
        ) and not self.feature_extraction_mode:
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
                raw_displacement, raw_vs_ha_gap,
                gt_bbox=info.get('gt_bbox') if info is not None else None)
            self.state = e27_output

        if self.kf_controller is not None:
            meas_cx = self.state[0] + 0.5 * self.state[2]
            meas_cy = self.state[1] + 0.5 * self.state[3]
            self.kf_controller.update(meas_cx, meas_cy)
            current_search_factor = self.kf_controller.get_search_factor(raw_score_max)

        if self.sh_controller is not None:
            self.is_lost = (self.sh_controller.state == 'LOST')
        elif self.hyst_controller is not None:
            self.is_lost = self.hyst_controller.update(raw_score_max)
        elif self.kf_controller is not None:
            self.is_lost = self.kf_controller.is_lost
        elif forced_action is not None:
            self.is_lost = forced_action == ACTION_RECOVERY
        elif self.use_sf_head and 'sf_cls_logits' in self._prev_result:
            sf_cls_probs = torch.softmax(self._prev_result['sf_cls_logits'], dim=-1)
            self.is_lost = sf_cls_probs[0][-1].item() > 0.5
        elif self.dynamic_search:
            max_score = response.max().item()
            if max_score < self.conf_thresh:
                self.is_lost = True
            else:
                self.is_lost = False

        if self.dump_frames or self.oracle_state_diag:
            cx_prev = self.state[0] + 0.5 * self.state[2]
            cy_prev = self.state[1] + 0.5 * self.state[3]
            crop_half = 0.5 * self.params.search_size / resize_factor
            frame_diag = {
                'frame_id': self.frame_id,
                'pred_bbox': [round(v, 2) for v in self.state],
                'search_factor': current_search_factor,
                'search_center': [round(cx_prev, 2), round(cy_prev, 2)],
                'crop_half_size': round(crop_half, 2),
                'raw_score_max': round(raw_score_max, 6),
                'response_score_max': round(response.max().item(), 6),
                'score_mean': round(response.mean().item(), 6),
                'edge_risk': round(current_edge_risk, 6),
                'is_lost': self.is_lost,
                'use_hann': use_hann,
            }
            if self.oracle_state or self.oracle_state_diag:
                current_gt_bbox = info.get('gt_bbox') if info is not None else None
                prev_gt_center = None
                if prev_gt_bbox is not None and len(prev_gt_bbox) >= 4 and bbox_visible(prev_gt_bbox):
                    prev_gt_center = [
                        round(float(prev_gt_bbox[0]) + 0.5 * float(prev_gt_bbox[2]), 2),
                        round(float(prev_gt_bbox[1]) + 0.5 * float(prev_gt_bbox[3]), 2),
                    ]
                current_gt_center = None
                output_iou = None
                if current_gt_bbox is not None and len(current_gt_bbox) >= 4 and bbox_visible(current_gt_bbox):
                    current_gt_center = [
                        round(float(current_gt_bbox[0]) + 0.5 * float(current_gt_bbox[2]), 2),
                        round(float(current_gt_bbox[1]) + 0.5 * float(current_gt_bbox[3]), 2),
                    ]
                    output_iou = round(float(bbox_iou_xywh(self.state, current_gt_bbox)), 6)
                frame_diag.update({
                    'oracle_state_enabled': bool(self.oracle_state),
                    'oracle_state_applied': bool(oracle_state_applied),
                    'oracle_state_reason': oracle_state_reason,
                    'normal_search_center': [
                        round(normal_search_state[0] + 0.5 * normal_search_state[2], 2),
                        round(normal_search_state[1] + 0.5 * normal_search_state[3], 2),
                    ],
                    'oracle_search_center': [
                        round(search_state[0] + 0.5 * search_state[2], 2),
                        round(search_state[1] + 0.5 * search_state[3], 2),
                    ],
                    'prev_gt_center': prev_gt_center,
                    'gt_center': current_gt_center,
                    'gt_bbox': None if current_gt_bbox is None else [round(float(v), 2) for v in current_gt_bbox[:4]],
                    'output_bbox': [round(float(v), 2) for v in self.state],
                    'output_iou': output_iou,
                })
            if forced_action is not None:
                frame_diag['action'] = forced_action
            if self.edge_proxy_sra:
                frame_diag['edge_proxy_tau'] = self.tau_edge
            if self.oracle_action is not None:
                frame_diag['oracle_action'] = self.oracle_action
            if self.sh_controller is not None:
                frame_diag['soft_hann_state'] = action.state
                frame_diag['hann_alpha'] = round(action.hann_alpha, 4)
                frame_diag['next_search_factor'] = round(action.next_search_factor, 4)
                frame_diag['lost_streak'] = self.sh_controller.lost_streak
                frame_diag['recover_streak'] = self.sh_controller.recover_streak
            if self.hyst_controller is not None:
                frame_diag['hyst_is_lost'] = self.hyst_controller.is_lost
                frame_diag['hyst_lost_streak'] = self.hyst_controller.lost_streak
                frame_diag['hyst_recover_streak'] = self.hyst_controller.recover_streak
            if self.kf_controller is not None:
                frame_diag['kf_uncertainty'] = round(self.kf_controller._uncertainty, 4)
                frame_diag['kf_velocity'] = [round(v, 4) for v in self.kf_controller._velocity]
                frame_diag['kf_innovation'] = round(self.kf_controller._innovation_norm, 4)
                frame_diag['kf_is_lost'] = self.kf_controller.is_lost
                if kf_center is not None:
                    frame_diag['kf_predicted_center'] = [round(v, 2) for v in kf_center]
            if self.gmc_controller is not None:
                frame_diag['gmc_shift'] = round(self.gmc_controller.get_shift(), 2)
                frame_diag['gmc_inliers'] = self.gmc_controller.get_inliers()
                frame_diag['gmc_valid'] = self.gmc_controller.is_valid()
            if self.use_sra and sra_flag is not None:
                frame_diag['sra_flag'] = round(sra_flag, 6)
            if topk_diag is not None:
                frame_diag.update(topk_diag)
            if hr_diag is not None:
                frame_diag.update(hr_diag)
            if ltr_feat_diag is not None:
                frame_diag['ltr_features'] = ltr_feat_diag
            if lgbm_diag is not None:
                frame_diag.update(lgbm_diag)
            if tcr_diag is not None:
                frame_diag.update(tcr_diag)
            if e27_diag is not None:
                frame_diag.update(e27_diag)
        else:
            frame_diag = None

        # for debug
        if self.debug:
            if not self.use_visdom or self.visdom is None:
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

        if 'failure_logit' in out_dict:
            logit = out_dict['failure_logit']
            if logit.shape[-1] > 1:
                probs = torch.softmax(logit, dim=-1)
                result['failure_prob'] = probs[0].detach().cpu().tolist()
            else:
                result['failure_prob'] = torch.sigmoid(logit).item()

        if 'logits' in out_dict:
            sra_logits = out_dict['logits']
            sra_probs = torch.softmax(sra_logits, dim=-1)
            result['sra_flag'] = sra_probs[0][-1].item()

        if 'sf_pred' in out_dict:
            result['sf_pred'] = out_dict['sf_pred'].item()
            result['sf_cls_logits'] = out_dict['sf_cls_logits']

        if 'action_logits' in out_dict:
            result['action_logits'] = out_dict['action_logits'].detach()
            action_probs = torch.softmax(out_dict['action_logits'], dim=-1)
            result['action_prob'] = action_probs[0].detach().cpu().tolist()
            if frame_diag is not None:
                frame_diag['action_prob'] = [round(v, 6) for v in result['action_prob']]

        if frame_diag is not None:
            result['frame_diag'] = frame_diag

        self._prev_result = result
        self._prev_score_max = raw_score_max
        self._prev_edge_risk = current_edge_risk

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
    return ORTrack
