import os
import sys
import argparse

"""Run frozen ReSeT configurations and the bounded study variants.

The default ReSeT command-line values correspond to the implementation fields
documented in the supplementary parameter table. Optional flags support the
single-component studies listed in `configs/mechanism_studies.json`.
"""

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, dynamic_search=False, conf_thresh=0.4, enlarged_search_factor=6.0,
                hann_control=False, dump_frames=False, save_dir=None, fsr_controller=None, soft_hann=None,
                sra_hysteresis=False, hyst_enter=0.40, hyst_exit=0.50, hyst_frames=2,
                kalman_sra=False, kf_process_noise=1e-2, kf_measurement_noise=1e-1,
                kf_uncertainty_thresh=15.0, kf_velocity_thresh=5.0,
                kf_center_predict=True, kf_uncertainty_sra=True,
                use_sf_head=False, sf_head_sf_min=4.0, sf_head_sf_max=7.0,
                use_action_head=False, tau_action=0.7, oracle_action=None,
                edge_proxy_sra=False, tau_edge=0.75,
                sra_policy=None, severe_score_thresh=0.20, severe_search_factor=6.0,
                size_sf_thresh=60.0, gray_score_high=0.50,
                sra_t_score=None, sra_t_logits=None, sra_max_search_factor=None, sra_enlarge_step=None,
                use_gmc=False, gmc_downscale=2, gmc_max_shift_ratio=0.3,
                gmc_use_clahe=True, gmc_use_target_mask=True, gmc_mask_padding=50,
                gmc_safety_valve=True,
                topk_oracle=0, topk_oracle_diag_only=False, response_gamma=1.0,
                heuristic_rerank=False, rerank_lambda=0.2, rerank_mu=0.1, rerank_nu=0.0,
                rerank_ambig_low=0.25, rerank_ambig_high=0.45, rerank_margin_thresh=0.05, rerank_topk=5,
                use_lgbm_reranker=False, lgbm_model_path='', ltr_topk=5,
                ltr_ambiguous_only=True, ltr_fallback_top1=True,
                use_tcr_reranker=False, tcr_model_path='', tcr_gate_path='',
                tcr_topk=5, tcr_gate_thr=0.70, tcr_margin=0.05,
                e27_mode=None, e27_bank_size=10, e27_bank_topk=3,
                e27_use_offsets=True, e27_offset_topk=1, e27_max_candidates=10,
                e27_delta_score=0.05, e27_margin_thresh=0.08,
                e27_strict_delta_score=0.08, e27_strict_margin_thresh=0.12,
                e27_bank23_delta_score=None, e27_bank23_margin_thresh=None,
                e29_c1_fallback_only=False, e29_trigger_budget_rate=0.0,
                e29_relative_gray_min_frames=0, e29_relative_high_score_protect=1.0,
                e29_reject_fallback='ha', e29_delayed_bank_write=False,
                e30_confirmation_frames=0, e30_source_aware_confirmation=False,
                e30_confirm_frames_c1=2, e30_confirm_frames_bank23=2,
                e30_confirm_frames_offset=1, e30_rollback=False,
                e30_rollback_apce_ratio=0.70, e30_rollback_margin_thresh=0.04,
                e30_rollback_score_thresh=0.35,
                e27_apce_low=80.0, e27_trigger_policy='legacy', e27_fallback='last_high_conf',
                e27_reset_policy='B', e27_cooldown=0, e27_long_unreliable=10,
                e27_include_sources='',
                e27_exclude_sources='',
                e27_dynamic_history=80, e27_dynamic_min_frames=20,
                e27_score_low_q=0.20, e27_score_gray_q=0.35, e27_apce_low_q=0.25,
                e27_bank_ranker='legacy', e27_local_neighborhood='legacy',
                e27_candidate_source='history',
                e27_state_quarantine=False, e27_reset_confidence=False,
                e27_reset_confidence_thresh=0.12,
                use_e31_verifier=False, e31_model_path='', e31_prob_thresh=0.55,
                commit_protocol='legacy', commit_gate='score_apce',
                commit_delay_frames=1, pending_ttl_frames=0,
                pending_replace_policy='replace', relative_acceptance='none',
                anchor_policy='last_committed', commit_oracle='none'):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of config file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        dynamic_search: Enable dynamic search region adaptation.
        conf_thresh: Confidence threshold for search adaptation.
        enlarged_search_factor: Enlarged search factor when target is lost.
        hann_control: Disable Hann window when target is lost.
    """

    dataset = get_dataset(*dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    for tracker in trackers:
        tracker.dynamic_search = dynamic_search
        tracker.conf_thresh = conf_thresh
        tracker.enlarged_search_factor = enlarged_search_factor
        tracker.hann_control = hann_control
        tracker.dump_frames = dump_frames
        if fsr_controller:
            tracker.fsr_controller = fsr_controller
        if soft_hann:
            tracker.soft_hann = soft_hann
        if sra_hysteresis:
            tracker.sra_hysteresis = True
            tracker.hyst_enter = hyst_enter
            tracker.hyst_exit = hyst_exit
            tracker.hyst_frames = hyst_frames
        if kalman_sra:
            tracker.kalman_sra = True
            tracker.kf_process_noise = kf_process_noise
            tracker.kf_measurement_noise = kf_measurement_noise
            tracker.kf_uncertainty_thresh = kf_uncertainty_thresh
            tracker.kf_velocity_thresh = kf_velocity_thresh
            tracker.kf_center_predict = kf_center_predict
            tracker.kf_uncertainty_sra = kf_uncertainty_sra
        if use_sf_head:
            tracker.use_sf_head = True
            tracker.sf_head_sf_min = sf_head_sf_min
            tracker.sf_head_sf_max = sf_head_sf_max
        if use_action_head:
            tracker.use_action_head = True
            tracker.tau_action = tau_action
        if oracle_action:
            tracker.oracle_action = oracle_action
        if edge_proxy_sra:
            tracker.edge_proxy_sra = True
            tracker.tau_edge = tau_edge
        if sra_t_score is not None:
            tracker.sra_t_score = sra_t_score
        if sra_t_logits is not None:
            tracker.sra_t_logits = sra_t_logits
        if sra_max_search_factor is not None:
            tracker.sra_max_search_factor = sra_max_search_factor
        if sra_enlarge_step is not None:
            tracker.sra_enlarge_step = sra_enlarge_step
        if use_gmc:
            tracker.use_gmc = True
            tracker.gmc_downscale = gmc_downscale
            tracker.gmc_max_shift_ratio = gmc_max_shift_ratio
            tracker.gmc_use_clahe = gmc_use_clahe
            tracker.gmc_use_target_mask = gmc_use_target_mask
            tracker.gmc_mask_padding = gmc_mask_padding
            tracker.gmc_safety_valve = gmc_safety_valve
        if sra_policy is not None:
            tracker.sra_policy = sra_policy
            tracker.severe_score_thresh = severe_score_thresh
            tracker.severe_search_factor = severe_search_factor
            tracker.size_sf_thresh = size_sf_thresh
            tracker.gray_score_high = gray_score_high
        if topk_oracle > 0:
            tracker.topk_oracle = topk_oracle
            tracker.topk_oracle_diag_only = topk_oracle_diag_only
        if response_gamma != 1.0:
            tracker.response_gamma = response_gamma
        if heuristic_rerank:
            tracker.heuristic_rerank = True
            tracker.rerank_lambda = rerank_lambda
            tracker.rerank_mu = rerank_mu
            tracker.rerank_nu = rerank_nu
            tracker.rerank_ambig_low = rerank_ambig_low
            tracker.rerank_ambig_high = rerank_ambig_high
            tracker.rerank_margin_thresh = rerank_margin_thresh
            tracker.rerank_topk = rerank_topk
        if use_lgbm_reranker:
            tracker.use_lgbm_reranker = True
            tracker.lgbm_model_path = lgbm_model_path
            tracker.ltr_topk = ltr_topk
            tracker.ltr_ambiguous_only = ltr_ambiguous_only
            tracker.ltr_fallback_top1 = ltr_fallback_top1
        if use_tcr_reranker:
            tracker.use_tcr_reranker = True
            tracker.tcr_model_path = tcr_model_path
            tracker.tcr_gate_path = tcr_gate_path
            tracker.tcr_topk = tcr_topk
            tracker.tcr_gate_thr = tcr_gate_thr
            tracker.tcr_margin = tcr_margin
        if e27_mode is not None:
            tracker.e27_mode = e27_mode
            tracker.e27_bank_size = e27_bank_size
            tracker.e27_bank_topk = e27_bank_topk
            tracker.e27_use_offsets = e27_use_offsets
            tracker.e27_offset_topk = e27_offset_topk
            tracker.e27_max_candidates = e27_max_candidates
            tracker.e27_delta_score = e27_delta_score
            tracker.e27_margin_thresh = e27_margin_thresh
            tracker.e27_strict_delta_score = e27_strict_delta_score
            tracker.e27_strict_margin_thresh = e27_strict_margin_thresh
            tracker.e27_bank23_delta_score = e27_bank23_delta_score
            tracker.e27_bank23_margin_thresh = e27_bank23_margin_thresh
            tracker.e29_c1_fallback_only = e29_c1_fallback_only
            tracker.e29_trigger_budget_rate = e29_trigger_budget_rate
            tracker.e29_relative_gray_min_frames = e29_relative_gray_min_frames
            tracker.e29_relative_high_score_protect = e29_relative_high_score_protect
            tracker.e29_reject_fallback = e29_reject_fallback
            tracker.e29_delayed_bank_write = e29_delayed_bank_write
            tracker.e30_confirmation_frames = e30_confirmation_frames
            tracker.e30_source_aware_confirmation = e30_source_aware_confirmation
            tracker.e30_confirm_frames_c1 = e30_confirm_frames_c1
            tracker.e30_confirm_frames_bank23 = e30_confirm_frames_bank23
            tracker.e30_confirm_frames_offset = e30_confirm_frames_offset
            tracker.e30_rollback = e30_rollback
            tracker.e30_rollback_apce_ratio = e30_rollback_apce_ratio
            tracker.e30_rollback_margin_thresh = e30_rollback_margin_thresh
            tracker.e30_rollback_score_thresh = e30_rollback_score_thresh
            tracker.e27_apce_low = e27_apce_low
            tracker.e27_trigger_policy = e27_trigger_policy
            tracker.e27_fallback = e27_fallback
            tracker.e27_reset_policy = e27_reset_policy
            tracker.e27_cooldown = e27_cooldown
            tracker.e27_long_unreliable = e27_long_unreliable
            tracker.e27_include_sources = e27_include_sources
            tracker.e27_exclude_sources = e27_exclude_sources
            tracker.e27_dynamic_history = e27_dynamic_history
            tracker.e27_dynamic_min_frames = e27_dynamic_min_frames
            tracker.e27_score_low_q = e27_score_low_q
            tracker.e27_score_gray_q = e27_score_gray_q
            tracker.e27_apce_low_q = e27_apce_low_q
            tracker.e27_bank_ranker = e27_bank_ranker
            tracker.e27_local_neighborhood = e27_local_neighborhood
            tracker.e27_candidate_source = e27_candidate_source
            tracker.e27_state_quarantine = e27_state_quarantine
            tracker.e27_reset_confidence = e27_reset_confidence
            tracker.e27_reset_confidence_thresh = e27_reset_confidence_thresh
            tracker.use_e31_verifier = use_e31_verifier
            tracker.e31_model_path = e31_model_path
            tracker.e31_prob_thresh = e31_prob_thresh
            tracker.commit_protocol = commit_protocol
            tracker.commit_gate = commit_gate
            tracker.commit_delay_frames = commit_delay_frames
            tracker.pending_ttl_frames = pending_ttl_frames
            tracker.pending_replace_policy = pending_replace_policy
            tracker.relative_acceptance = relative_acceptance
            tracker.anchor_policy = anchor_policy
            tracker.commit_oracle = commit_oracle
        suffix = ''
        if dynamic_search:
            suffix += '_sra'
        if hann_control:
            suffix += '_hann'
        if dynamic_search and hann_control:
            suffix += f'_ct{conf_thresh}_sf{enlarged_search_factor}'
        if fsr_controller:
            suffix += '_' + fsr_controller
        if soft_hann:
            suffix += '_soft_hann_' + soft_hann
        if sra_hysteresis:
            suffix += f'_hyst_e{hyst_enter}_x{hyst_exit}_n{hyst_frames}'
        if kalman_sra:
            suffix += f'_kf_ct{conf_thresh}_sf{enlarged_search_factor}'
            if not kf_center_predict:
                suffix += '_nocp'
            if not kf_uncertainty_sra:
                suffix += '_nousra'
        if use_sf_head:
            suffix += f'_sfhead_min{sf_head_sf_min}_max{sf_head_sf_max}'
        if use_action_head:
            suffix += f'_action_tau{tau_action}'
        if oracle_action:
            suffix += f'_oracle_{oracle_action}'
        if edge_proxy_sra:
            suffix += f'_edge_proxy_tau{tau_edge}'
        if sra_policy is not None:
            suffix += f'_sra-{sra_policy}'
            if sra_policy in ['score_severity', 'score_size_hybrid', 'boundary_severe']:
                suffix += f'_st{severe_score_thresh}'
            if sra_policy in ['size_conditioned', 'score_size_hybrid']:
                suffix += f'_szt{size_sf_thresh}'
            if sra_policy in ['score_severity', 'score_size_hybrid', 'boundary_severe']:
                suffix += f'_ssf{severe_search_factor}'
        if any(v is not None for v in [sra_t_score, sra_t_logits, sra_max_search_factor, sra_enlarge_step]):
            suffix += f'_sra_ts{sra_t_score}_tl{sra_t_logits}_msf{sra_max_search_factor}'
            if sra_enlarge_step is not None:
                suffix += f'_step{sra_enlarge_step}'
        if use_gmc:
            suffix += '_gmc'
            if not gmc_safety_valve:
                suffix += '_nosv'
            if not gmc_use_clahe:
                suffix += '_noclahe'
        if topk_oracle > 0:
            suffix += f'_topk{topk_oracle}'
            if topk_oracle_diag_only:
                suffix += '_diag'
        if response_gamma != 1.0:
            suffix += f'_gamma{response_gamma}'
        if heuristic_rerank:
            suffix += f'_hr_l{rerank_lambda}_m{rerank_mu}_n{rerank_nu}'
            if rerank_topk != 5:
                suffix += f'_rtopk{rerank_topk}'
        if use_lgbm_reranker:
            suffix += f'_lgbm'
        if use_tcr_reranker:
            suffix += f'_tcr_gate{tcr_gate_thr}_m{tcr_margin}_topk{tcr_topk}'
        if e27_mode is not None:
            suffix += f'_e27_{e27_mode}_ds{e27_delta_score}_m{e27_margin_thresh}_bt{e27_bank_topk}'
            if not e27_use_offsets:
                suffix += '_nooff'
            if e27_fallback != 'last_high_conf':
                suffix += f'_fb{e27_fallback}'
            if e27_reset_policy != 'B':
                suffix += f'_reset{e27_reset_policy}'
            if e27_cooldown:
                suffix += f'_cd{e27_cooldown}'
            if e27_long_unreliable != 10:
                suffix += f'_lu{e27_long_unreliable}'
            if e27_bank23_delta_score is not None:
                suffix += f'_b23ds{e27_bank23_delta_score}'
            if e27_bank23_margin_thresh is not None:
                suffix += f'_b23m{e27_bank23_margin_thresh}'
            if e29_c1_fallback_only:
                suffix += '_c1fallbackonly'
            if e29_trigger_budget_rate:
                suffix += f'_tbud{e29_trigger_budget_rate}'
            if e29_relative_gray_min_frames:
                suffix += f'_rgmin{e29_relative_gray_min_frames}'
            if e29_relative_high_score_protect != 1.0:
                suffix += f'_rhsp{e29_relative_high_score_protect}'
            if e29_reject_fallback != 'ha':
                suffix += f'_rejfb{e29_reject_fallback}'
            if e29_delayed_bank_write:
                suffix += '_delaybank'
            if e30_confirmation_frames:
                suffix += f'_e30cf{e30_confirmation_frames}'
            if e30_source_aware_confirmation:
                suffix += f'_e30srcawarec1{e30_confirm_frames_c1}b23{e30_confirm_frames_bank23}off{e30_confirm_frames_offset}'
            if e30_rollback:
                suffix += f'_e30rollback{e30_rollback_score_thresh}'
            if e27_trigger_policy != 'legacy':
                suffix += f'_trig{e27_trigger_policy}'
            if e27_bank_ranker != 'legacy':
                suffix += f'_br{e27_bank_ranker}'
            if e27_candidate_source != 'history':
                suffix += f'_cs{e27_candidate_source}'
            if e27_local_neighborhood != 'legacy':
                suffix += f'_ln{e27_local_neighborhood}'
            if e27_state_quarantine:
                suffix += '_quarantine'
            if e27_reset_confidence:
                suffix += f'_rconf{e27_reset_confidence_thresh}'
            if e27_trigger_policy == 'relative_score_apce_gate':
                suffix += f'_q{e27_score_low_q}-{e27_score_gray_q}-{e27_apce_low_q}'
            if e27_include_sources:
                suffix += '_in' + str(e27_include_sources).replace(',', '-')
            if e27_exclude_sources:
                suffix += '_ex' + str(e27_exclude_sources).replace(',', '-')
            if use_e31_verifier:
                suffix += f'_e31verifier_thr{e31_prob_thresh}'
            if commit_protocol != 'legacy' or commit_oracle != 'none':
                suffix += f'_reset_{commit_protocol}_{commit_gate}_d{commit_delay_frames}'
                if relative_acceptance != 'none':
                    suffix += f'_rel{relative_acceptance}'
                if anchor_policy != 'last_committed':
                    suffix += f'_anc{anchor_policy}'
                if commit_oracle != 'none':
                    suffix += f'_oracle{commit_oracle}'
        if suffix:
            tracker.results_dir = tracker.results_dir + suffix

    if save_dir is not None:
        for tracker in trackers:
            tracker.results_dir = tracker.results_dir.replace(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'test', 'tracking_results'),
                save_dir)

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot, antiuav410).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--dynamic_search', action='store_true', help='Enable dynamic search region adaptation.')
    parser.add_argument('--conf_thresh', type=float, default=0.4, help='Confidence threshold for search adaptation.')
    parser.add_argument('--enlarged_search_factor', type=float, default=6.0, help='Enlarged search factor when target is lost.')
    parser.add_argument('--hann_control', action='store_true', help='Disable Hann window when target is lost.')
    parser.add_argument('--dump_frames', action='store_true', help='Dump per-frame diagnostic data.')
    parser.add_argument('--save_dir', type=str, default=None, help='Override results directory.')
    parser.add_argument('--fsr_controller', type=str, default=None,
                        choices=['fsr_only', 'score_fsr'],
                        help='FSR controller mode: fsr_only or score_fsr')
    parser.add_argument('--soft_hann', type=str, default=None,
                        choices=['h1', 'h2', 'h3'],
                        help='Soft-Hann mode: h1 (interpolation), h2 (+hysteresis), h3 (+search_factor)')
    parser.add_argument('--sra_hysteresis', action='store_true', help='Enable SRA hysteresis controller')
    parser.add_argument('--hyst_enter', type=float, default=0.40, help='Hysteresis enter threshold')
    parser.add_argument('--hyst_exit', type=float, default=0.50, help='Hysteresis exit threshold')
    parser.add_argument('--hyst_frames', type=int, default=2, help='Hysteresis frame count')
    parser.add_argument('--kalman_sra', action='store_true', help='Enable Kalman filter SRA controller')
    parser.add_argument('--kf_process_noise', type=float, default=1e-2, help='Kalman process noise (Q)')
    parser.add_argument('--kf_measurement_noise', type=float, default=1e-1, help='Kalman measurement noise (R)')
    parser.add_argument('--kf_uncertainty_thresh', type=float, default=15.0, help='Uncertainty threshold for SRA')
    parser.add_argument('--kf_velocity_thresh', type=float, default=5.0, help='Velocity threshold for SRA')
    parser.add_argument('--kf_center_predict', action='store_true', default=True, help='Use KF predicted center for search')
    parser.add_argument('--kf_no_center_predict', action='store_false', dest='kf_center_predict')
    parser.add_argument('--kf_uncertainty_sra', action='store_true', default=True, help='Use KF uncertainty for SRA')
    parser.add_argument('--kf_no_uncertainty_sra', action='store_false', dest='kf_uncertainty_sra')
    parser.add_argument('--use_sf_head', action='store_true', help='Use learned SF regression head')
    parser.add_argument('--sf_head_sf_min', type=float, default=4.0, help='SF head min search factor')
    parser.add_argument('--sf_head_sf_max', type=float, default=7.0, help='SF head max search factor')
    parser.add_argument('--use_action_head', action='store_true', help='Use learned NORMAL/RECOVERY action head')
    parser.add_argument('--tau_action', type=float, default=0.7, help='Recovery probability threshold in score+action gray zone')
    parser.add_argument('--oracle_action', type=str, default=None,
                        choices=['visibility', 'region', 'best_action'],
                        help='Oracle action controller for upper-bound experiments')
    parser.add_argument('--edge_proxy_sra', action='store_true',
                        help='Enable response peak edge-risk proxy for hard SRA control')
    parser.add_argument('--tau_edge', type=float, default=0.75,
                        help='Edge risk threshold for Edge-Proxy Hard SRA gray-zone control')
    parser.add_argument('--sra_policy', type=str, default=None,
                        choices=['score_severity', 'size_conditioned', 'score_size_hybrid', 'boundary_severe'],
                        help='Layered SRA policy for P1-P4/P7')
    parser.add_argument('--severe_score_thresh', type=float, default=0.20,
                        help='Score threshold for severe recovery (sf=6.0)')
    parser.add_argument('--severe_search_factor', type=float, default=6.0,
                        help='Search factor for severe recovery')
    parser.add_argument('--size_sf_thresh', type=float, default=60.0,
                        help='sqrt(w*h) threshold for size-conditioned SRA')
    parser.add_argument('--gray_score_high', type=float, default=0.50,
                        help='Upper bound of score gray zone')
    parser.add_argument('--sra_t_score', type=float, default=None,
                        help='Override FocusTrack-style SRA score threshold')
    parser.add_argument('--sra_t_logits', type=float, default=None,
                        help='Override FocusTrack-style SRA logits/probability threshold')
    parser.add_argument('--sra_max_search_factor', type=float, default=None,
                        help='Override FocusTrack-style SRA max search factor')
    parser.add_argument('--sra_enlarge_step', type=float, default=None,
                        help='Override FocusTrack-style SRA enlarge step')
    parser.add_argument('--use_gmc', action='store_true', help='Enable GMC-SCC (Search Center Compensation)')
    parser.add_argument('--gmc_downscale', type=int, default=2, help='GMC image downscale factor')
    parser.add_argument('--gmc_max_shift_ratio', type=float, default=0.3, help='GMC max shift ratio for safety valve')
    parser.add_argument('--gmc_no_clahe', action='store_false', dest='gmc_use_clahe', help='Disable CLAHE for GMC')
    parser.add_argument('--gmc_no_target_mask', action='store_false', dest='gmc_use_target_mask', help='Disable target mask for GMC')
    parser.add_argument('--gmc_mask_padding', type=int, default=50, help='GMC target mask padding in pixels')
    parser.add_argument('--gmc_no_safety_valve', action='store_false', dest='gmc_safety_valve', help='Disable GMC safety valve')
    parser.add_argument('--topk_oracle', type=int, default=0,
                        help='Q0: Enable top-k oracle diagnostic (k=3/5/10)')
    parser.add_argument('--topk_oracle_diag_only', action='store_true',
                        help='Collect top-k oracle diagnostics without overriding SSRM output state')
    parser.add_argument('--response_gamma', type=float, default=1.0,
                        help='Q1: Response exponent gamma (1.25/1.5/2.0)')
    parser.add_argument('--heuristic_rerank', action='store_true',
                        help='Q2: Enable heuristic top-k reranking on ambiguous frames')
    parser.add_argument('--rerank_lambda', type=float, default=0.2,
                        help='Q2: Weight for template_cosine in rerank score')
    parser.add_argument('--rerank_mu', type=float, default=0.1,
                        help='Q2: Weight for displacement_penalty in rerank score')
    parser.add_argument('--rerank_nu', type=float, default=0.0,
                        help='Q2: Weight for edge_penalty in rerank score')
    parser.add_argument('--rerank_ambig_low', type=float, default=0.25,
                        help='Q2: Lower bound of ambiguous score range')
    parser.add_argument('--rerank_ambig_high', type=float, default=0.45,
                        help='Q2: Upper bound of ambiguous score range')
    parser.add_argument('--rerank_margin_thresh', type=float, default=0.05,
                        help='Q2: Top1-top2 margin threshold for ambiguity')
    parser.add_argument('--rerank_topk', type=int, default=5,
                        help='Q2: Number of top-k candidates for reranking')
    parser.add_argument('--use_lgbm_reranker', action='store_true',
                        help='LTR: Enable LGBM learned reranking on ambiguous frames')
    parser.add_argument('--lgbm_model_path', type=str, default='',
                        help='LTR: Path to pickled LGBM model')
    parser.add_argument('--ltr_topk', type=int, default=5,
                        help='LTR: Number of top-k candidates for LGBM reranking')
    parser.add_argument('--ltr_ambiguous_only', action='store_true', default=True,
                        help='LTR: Only rerank on ambiguous frames')
    parser.add_argument('--ltr_no_ambiguous_only', action='store_false', dest='ltr_ambiguous_only',
                        help='LTR: Rerank on all frames')
    parser.add_argument('--ltr_fallback_top1', action='store_true', default=True,
                        help='LTR: Fall back to top-1 if LGBM score <= 0')
    parser.add_argument('--ltr_no_fallback_top1', action='store_false', dest='ltr_fallback_top1',
                        help='LTR: Always use LGBM choice')
    parser.add_argument('--use_tcr_reranker', action='store_true',
                        help='E22-T1: Enable Token Candidate Reranker with optional gate')
    parser.add_argument('--tcr_model_path', type=str, default='',
                        help='E22-T1: Path to tcr_v1.pt')
    parser.add_argument('--tcr_gate_path', type=str, default='',
                        help='E22-T1: Path to gate_mlp.pt')
    parser.add_argument('--tcr_topk', type=int, default=5,
                        help='E22-T1: Number of top-k candidates for TCR')
    parser.add_argument('--tcr_gate_thr', type=float, default=0.70,
                        help='E22-T1: Gate threshold')
    parser.add_argument('--tcr_margin', type=float, default=0.05,
                        help='E22-T1: Minimum TCR score margin over top1')
    parser.add_argument('--e27_mode', type=str, default=None,
                        choices=['C0', 'C1', 'C2', 'C2safe', 'C2-safe'],
                        help='E27 State-Quarantined Multi-center SRA mode')
    parser.add_argument('--e27_bank_size', type=int, default=10)
    parser.add_argument('--e27_bank_topk', type=int, default=3)
    parser.add_argument('--e27_no_offsets', action='store_false', dest='e27_use_offsets')
    parser.add_argument('--e27_offset_topk', type=int, default=1)
    parser.add_argument('--e27_max_candidates', type=int, default=10)
    parser.add_argument('--e27_delta_score', type=float, default=0.05)
    parser.add_argument('--e27_margin_thresh', type=float, default=0.08)
    parser.add_argument('--e27_strict_delta_score', type=float, default=0.08)
    parser.add_argument('--e27_strict_margin_thresh', type=float, default=0.12)
    parser.add_argument('--e27_bank23_delta_score', type=float, default=None,
                        help='Optional stricter score delta for C4_bank_top2/C4_bank_top3 candidates')
    parser.add_argument('--e27_bank23_margin_thresh', type=float, default=None,
                        help='Optional stricter margin threshold for C4_bank_top2/C4_bank_top3 candidates')
    parser.add_argument('--e29_c1_fallback_only', action='store_true',
                        help='E29-A3: C1_last_reliable can guide fallback search_state but cannot be accepted directly')
    parser.add_argument('--e29_trigger_budget_rate', type=float, default=0.0,
                        help='E29-B1: per-sequence trigger budget rate; 0 disables the budget')
    parser.add_argument('--e29_relative_gray_min_frames', type=int, default=0,
                        help='E29-B2: relative gray-zone trigger requires frames_since_last_reliable >= this value')
    parser.add_argument('--e29_relative_high_score_protect', type=float, default=1.0,
                        help='E29-B3: disable relative hard-lost trigger when raw_score >= this value')
    parser.add_argument('--e29_reject_fallback', type=str, default='ha',
                        choices=['ha', 'bank_top1', 'previous_search_state', 'last_reliable', 'last_high_conf'],
                        help='E29-C1/C2: search_state used after rejected recovery')
    parser.add_argument('--e29_delayed_bank_write', action='store_true',
                        help='E29-C3: write accepted recovery into reliable bank only after the next reliable frame')
    parser.add_argument('--e30_confirmation_frames', type=int, default=0,
                        help='E30-A: require N reliable frames before committing pending recovery into the bank')
    parser.add_argument('--e30_source_aware_confirmation', action='store_true',
                        help='E30-C: use source-specific confirmation frame counts')
    parser.add_argument('--e30_confirm_frames_c1', type=int, default=2,
                        help='E30-C: confirmation frames for C1_last_reliable')
    parser.add_argument('--e30_confirm_frames_bank23', type=int, default=2,
                        help='E30-C: confirmation frames for C4_bank_top2/top3')
    parser.add_argument('--e30_confirm_frames_offset', type=int, default=1,
                        help='E30-C: confirmation frames for local offset sources')
    parser.add_argument('--e30_rollback', action='store_true',
                        help='E30-B: rollback pending recovery when short-term reliability collapses')
    parser.add_argument('--e30_rollback_apce_ratio', type=float, default=0.70,
                        help='E30-B: rollback if APCE drops below pending APCE times this ratio')
    parser.add_argument('--e30_rollback_margin_thresh', type=float, default=0.04,
                        help='E30-B: rollback if raw margin falls below this value')
    parser.add_argument('--e30_rollback_score_thresh', type=float, default=0.35,
                        help='E30-B: rollback if raw score falls below this value')
    parser.add_argument('--e27_apce_low', type=float, default=80.0)
    parser.add_argument('--e27_trigger_policy', type=str, default='legacy',
                        choices=['legacy', 'score_apce_gate', 'relative_score_apce_gate'])
    parser.add_argument('--e27_dynamic_history', type=int, default=80)
    parser.add_argument('--e27_dynamic_min_frames', type=int, default=20)
    parser.add_argument('--e27_score_low_q', type=float, default=0.20)
    parser.add_argument('--e27_score_gray_q', type=float, default=0.35)
    parser.add_argument('--e27_apce_low_q', type=float, default=0.25)
    parser.add_argument('--e27_bank_ranker', type=str, default='legacy',
                        choices=['legacy', 'reliability', 'score', 'apce', 'score_apce',
                                 'freshness', 'reliability_freshness', 'stability_aware',
                                 'diversity_aware'])
    parser.add_argument('--e27_local_neighborhood', type=str, default='legacy',
                        choices=['legacy', 'local_lr', 'local_ring4_strict'])
    parser.add_argument('--e27_candidate_source', type=str, default='history',
                        choices=['history', 'current_ring', 'enlarged_current',
                                 'velocity_ring', 'uniform_grid',
                                 'random_history', 'last_reliable_ring'],
                        help='Support source under the matched trigger, verifier, and candidate-list budget.')
    parser.add_argument('--e27_state_quarantine', action='store_true')
    parser.add_argument('--e27_reset_confidence', action='store_true')
    parser.add_argument('--e27_reset_confidence_thresh', type=float, default=0.12)
    parser.add_argument('--commit_protocol', type=str, default='legacy',
                        choices=['legacy', 'immediate', 'fixed_delay', 'reliability_confirmed', 'three_state'],
                        help='ReSeT search-state commitment protocol')
    parser.add_argument('--commit_gate', type=str, default='score_apce',
                        choices=['score', 'apce', 'score_apce', 'score_apce_disp', 'score_apce_disp_scale'],
                        help='Reliability gate used by ReSeT commitment')
    parser.add_argument('--commit_delay_frames', type=int, default=1,
                        help='Frames required before committing a tentative recovery')
    parser.add_argument('--pending_ttl_frames', type=int, default=0,
                        help='Maximum pending age; 0 preserves the frozen no-TTL policy')
    parser.add_argument('--pending_replace_policy', type=str, default='replace',
                        choices=['replace', 'keep'],
                        help='Whether a new accepted recovery supersedes an active pending record')
    parser.add_argument('--relative_acceptance', type=str, default='none',
                        choices=['none', 'current', 'committed', 'both'],
                        help='Require accepted recovery to improve over current and/or committed state')
    parser.add_argument('--anchor_policy', type=str, default='last_committed',
                        choices=['last_committed', 'best_score', 'best_reliability',
                                 'reliability_freshness', 'topm', 'oracle_diag'],
                        help='Committed anchor selection for tentative/isolated frames')
    parser.add_argument('--commit_oracle', type=str, default='none',
                        choices=['none', 'commit', 'isolation', 'anchor'],
                        help='Diagnostic-only oracle for ReSeT upper-bound studies')
    parser.add_argument('--e27_fallback', type=str, default='last_high_conf',
                        choices=['last_high_conf', 'last_reliable', 'previous_search_state'])
    parser.add_argument('--e27_reset_policy', type=str, default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--e27_cooldown', type=int, default=0)
    parser.add_argument('--e27_long_unreliable', type=int, default=10)
    parser.add_argument('--e27_include_sources', type=str, default='',
                        help='Comma-separated E27 candidate source prefixes allowed by verifier')
    parser.add_argument('--e27_exclude_sources', type=str, default='',
                        help='Comma-separated E27 candidate source prefixes excluded by verifier')
    parser.add_argument('--use_e31_verifier', action='store_true',
                        help='E31: use learned candidate verifier as a reject gate after E27 accepts a recovery candidate')
    parser.add_argument('--e31_model_path', type=str, default='',
                        help='E31: path to candidate verifier pickle')
    parser.add_argument('--e31_prob_thresh', type=float, default=0.55,
                        help='E31: accept probability threshold')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    if args.dataset_name == 'otb':
        args.dataset_name = ['trackingnet', 'got10k_test', 'lasot']
    else:
        args.dataset_name = [args.dataset_name]

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus,
                dynamic_search=args.dynamic_search, conf_thresh=args.conf_thresh,
                enlarged_search_factor=args.enlarged_search_factor,
                hann_control=args.hann_control,
                dump_frames=args.dump_frames,
                save_dir=args.save_dir,
                fsr_controller=args.fsr_controller,
                soft_hann=args.soft_hann,
                sra_hysteresis=args.sra_hysteresis,
                hyst_enter=args.hyst_enter,
                hyst_exit=args.hyst_exit,
                hyst_frames=args.hyst_frames,
                kalman_sra=args.kalman_sra,
                kf_process_noise=args.kf_process_noise,
                kf_measurement_noise=args.kf_measurement_noise,
                kf_uncertainty_thresh=args.kf_uncertainty_thresh,
                kf_velocity_thresh=args.kf_velocity_thresh,
                kf_center_predict=args.kf_center_predict,
                kf_uncertainty_sra=args.kf_uncertainty_sra,
                use_sf_head=args.use_sf_head,
                sf_head_sf_min=args.sf_head_sf_min,
                sf_head_sf_max=args.sf_head_sf_max,
                use_action_head=args.use_action_head,
                tau_action=args.tau_action,
                oracle_action=args.oracle_action,
                edge_proxy_sra=args.edge_proxy_sra,
                tau_edge=args.tau_edge,
                sra_policy=args.sra_policy,
                severe_score_thresh=args.severe_score_thresh,
                severe_search_factor=args.severe_search_factor,
                size_sf_thresh=args.size_sf_thresh,
                gray_score_high=args.gray_score_high,
                sra_t_score=args.sra_t_score,
                sra_t_logits=args.sra_t_logits,
                sra_max_search_factor=args.sra_max_search_factor,
                sra_enlarge_step=args.sra_enlarge_step,
                use_gmc=args.use_gmc,
                gmc_downscale=args.gmc_downscale,
                gmc_max_shift_ratio=args.gmc_max_shift_ratio,
                gmc_use_clahe=args.gmc_use_clahe,
                gmc_use_target_mask=args.gmc_use_target_mask,
                gmc_mask_padding=args.gmc_mask_padding,
                gmc_safety_valve=args.gmc_safety_valve,
                topk_oracle=args.topk_oracle,
                topk_oracle_diag_only=args.topk_oracle_diag_only,
                response_gamma=args.response_gamma,
                heuristic_rerank=args.heuristic_rerank,
                rerank_lambda=args.rerank_lambda,
                rerank_mu=args.rerank_mu,
                rerank_nu=args.rerank_nu,
                rerank_ambig_low=args.rerank_ambig_low,
                rerank_ambig_high=args.rerank_ambig_high,
                rerank_margin_thresh=args.rerank_margin_thresh,
                rerank_topk=args.rerank_topk,
                use_lgbm_reranker=args.use_lgbm_reranker,
                lgbm_model_path=args.lgbm_model_path,
                ltr_topk=args.ltr_topk,
                ltr_ambiguous_only=args.ltr_ambiguous_only,
                ltr_fallback_top1=args.ltr_fallback_top1,
                use_tcr_reranker=args.use_tcr_reranker,
                tcr_model_path=args.tcr_model_path,
                tcr_gate_path=args.tcr_gate_path,
                tcr_topk=args.tcr_topk,
                tcr_gate_thr=args.tcr_gate_thr,
                tcr_margin=args.tcr_margin,
                e27_mode=args.e27_mode,
                e27_bank_size=args.e27_bank_size,
                e27_bank_topk=args.e27_bank_topk,
                e27_use_offsets=args.e27_use_offsets,
                e27_offset_topk=args.e27_offset_topk,
                e27_max_candidates=args.e27_max_candidates,
                e27_delta_score=args.e27_delta_score,
                e27_margin_thresh=args.e27_margin_thresh,
                e27_strict_delta_score=args.e27_strict_delta_score,
                e27_strict_margin_thresh=args.e27_strict_margin_thresh,
                e27_bank23_delta_score=args.e27_bank23_delta_score,
                e27_bank23_margin_thresh=args.e27_bank23_margin_thresh,
                e29_c1_fallback_only=args.e29_c1_fallback_only,
                e29_trigger_budget_rate=args.e29_trigger_budget_rate,
                e29_relative_gray_min_frames=args.e29_relative_gray_min_frames,
                e29_relative_high_score_protect=args.e29_relative_high_score_protect,
                e29_reject_fallback=args.e29_reject_fallback,
                e29_delayed_bank_write=args.e29_delayed_bank_write,
                e30_confirmation_frames=args.e30_confirmation_frames,
                e30_source_aware_confirmation=args.e30_source_aware_confirmation,
                e30_confirm_frames_c1=args.e30_confirm_frames_c1,
                e30_confirm_frames_bank23=args.e30_confirm_frames_bank23,
                e30_confirm_frames_offset=args.e30_confirm_frames_offset,
                e30_rollback=args.e30_rollback,
                e30_rollback_apce_ratio=args.e30_rollback_apce_ratio,
                e30_rollback_margin_thresh=args.e30_rollback_margin_thresh,
                e30_rollback_score_thresh=args.e30_rollback_score_thresh,
                e27_apce_low=args.e27_apce_low,
                e27_trigger_policy=args.e27_trigger_policy,
                e27_fallback=args.e27_fallback,
                e27_reset_policy=args.e27_reset_policy,
                e27_cooldown=args.e27_cooldown,
                e27_long_unreliable=args.e27_long_unreliable,
                e27_include_sources=args.e27_include_sources,
                e27_exclude_sources=args.e27_exclude_sources,
                e27_dynamic_history=args.e27_dynamic_history,
                e27_dynamic_min_frames=args.e27_dynamic_min_frames,
                e27_score_low_q=args.e27_score_low_q,
                e27_score_gray_q=args.e27_score_gray_q,
                e27_apce_low_q=args.e27_apce_low_q,
                e27_bank_ranker=args.e27_bank_ranker,
                e27_local_neighborhood=args.e27_local_neighborhood,
                e27_candidate_source=args.e27_candidate_source,
                e27_state_quarantine=args.e27_state_quarantine,
                e27_reset_confidence=args.e27_reset_confidence,
                e27_reset_confidence_thresh=args.e27_reset_confidence_thresh,
                use_e31_verifier=args.use_e31_verifier,
                e31_model_path=args.e31_model_path,
                e31_prob_thresh=args.e31_prob_thresh,
                commit_protocol=args.commit_protocol,
                commit_gate=args.commit_gate,
                commit_delay_frames=args.commit_delay_frames,
                pending_ttl_frames=args.pending_ttl_frames,
                pending_replace_policy=args.pending_replace_policy,
                relative_acceptance=args.relative_acceptance,
                anchor_policy=args.anchor_policy,
                commit_oracle=args.commit_oracle)


if __name__ == '__main__':
    main()
