import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_class_map = {
            'ortrack': 'ORTrack',
            'ortrack_reset': 'ORTrack',
        }
        if self.name in tracker_class_map:
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = getattr(tracker_module, tracker_class_map[self.name])
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        if hasattr(self, 'dynamic_search'):
            params.dynamic_search = self.dynamic_search
        if hasattr(self, 'conf_thresh'):
            params.conf_thresh = self.conf_thresh
        if hasattr(self, 'enlarged_search_factor'):
            params.enlarged_search_factor = self.enlarged_search_factor
        if hasattr(self, 'hann_control'):
            params.hann_control = self.hann_control
        if hasattr(self, 'dump_frames'):
            params.dump_frames = self.dump_frames
        if hasattr(self, 'soft_hann'):
            params.soft_hann = self.soft_hann
        if hasattr(self, 'sra_hysteresis'):
            params.sra_hysteresis = self.sra_hysteresis
        if hasattr(self, 'hyst_enter'):
            params.hyst_enter = self.hyst_enter
        if hasattr(self, 'hyst_exit'):
            params.hyst_exit = self.hyst_exit
        if hasattr(self, 'hyst_frames'):
            params.hyst_frames = self.hyst_frames
        if hasattr(self, 'kalman_sra'):
            params.kalman_sra = self.kalman_sra
        if hasattr(self, 'kf_process_noise'):
            params.kf_process_noise = self.kf_process_noise
        if hasattr(self, 'kf_measurement_noise'):
            params.kf_measurement_noise = self.kf_measurement_noise
        if hasattr(self, 'kf_uncertainty_thresh'):
            params.kf_uncertainty_thresh = self.kf_uncertainty_thresh
        if hasattr(self, 'kf_velocity_thresh'):
            params.kf_velocity_thresh = self.kf_velocity_thresh
        if hasattr(self, 'kf_center_predict'):
            params.kf_center_predict = self.kf_center_predict
        if hasattr(self, 'kf_uncertainty_sra'):
            params.kf_uncertainty_sra = self.kf_uncertainty_sra
        if hasattr(self, 'use_sf_head'):
            params.use_sf_head = self.use_sf_head
        if hasattr(self, 'sf_head_sf_min'):
            params.sf_head_sf_min = self.sf_head_sf_min
        if hasattr(self, 'sf_head_sf_max'):
            params.sf_head_sf_max = self.sf_head_sf_max
        if hasattr(self, 'use_action_head'):
            params.use_action_head = self.use_action_head
        if hasattr(self, 'tau_action'):
            params.tau_action = self.tau_action
        if hasattr(self, 'oracle_action'):
            params.oracle_action = self.oracle_action
        if hasattr(self, 'edge_proxy_sra'):
            params.edge_proxy_sra = self.edge_proxy_sra
        if hasattr(self, 'tau_edge'):
            params.tau_edge = self.tau_edge
        if hasattr(self, 'sra_policy'):
            params.sra_policy = self.sra_policy
        if hasattr(self, 'severe_score_thresh'):
            params.severe_score_thresh = self.severe_score_thresh
        if hasattr(self, 'severe_search_factor'):
            params.severe_search_factor = self.severe_search_factor
        if hasattr(self, 'size_sf_thresh'):
            params.size_sf_thresh = self.size_sf_thresh
        if hasattr(self, 'gray_score_high'):
            params.gray_score_high = self.gray_score_high
        if hasattr(self, 'sra_t_score'):
            params.sra_t_score = self.sra_t_score
        if hasattr(self, 'sra_t_logits'):
            params.sra_t_logits = self.sra_t_logits
        if hasattr(self, 'sra_max_search_factor'):
            params.sra_max_search_factor = self.sra_max_search_factor
        if hasattr(self, 'sra_enlarge_step'):
            params.sra_enlarge_step = self.sra_enlarge_step
        if hasattr(self, 'use_gmc'):
            params.use_gmc = self.use_gmc
        if hasattr(self, 'gmc_downscale'):
            params.gmc_downscale = self.gmc_downscale
        if hasattr(self, 'gmc_max_shift_ratio'):
            params.gmc_max_shift_ratio = self.gmc_max_shift_ratio
        if hasattr(self, 'gmc_use_clahe'):
            params.gmc_use_clahe = self.gmc_use_clahe
        if hasattr(self, 'gmc_use_target_mask'):
            params.gmc_use_target_mask = self.gmc_use_target_mask
        if hasattr(self, 'gmc_mask_padding'):
            params.gmc_mask_padding = self.gmc_mask_padding
        if hasattr(self, 'gmc_safety_valve'):
            params.gmc_safety_valve = self.gmc_safety_valve
        if hasattr(self, 'topk_oracle'):
            params.topk_oracle = self.topk_oracle
        if hasattr(self, 'response_gamma'):
            params.response_gamma = self.response_gamma
        if hasattr(self, 'heuristic_rerank'):
            params.heuristic_rerank = self.heuristic_rerank
        if hasattr(self, 'rerank_lambda'):
            params.rerank_lambda = self.rerank_lambda
        if hasattr(self, 'rerank_mu'):
            params.rerank_mu = self.rerank_mu
        if hasattr(self, 'rerank_nu'):
            params.rerank_nu = self.rerank_nu
        if hasattr(self, 'rerank_ambig_low'):
            params.rerank_ambig_low = self.rerank_ambig_low
        if hasattr(self, 'rerank_ambig_high'):
            params.rerank_ambig_high = self.rerank_ambig_high
        if hasattr(self, 'rerank_margin_thresh'):
            params.rerank_margin_thresh = self.rerank_margin_thresh
        if hasattr(self, 'rerank_topk'):
            params.rerank_topk = self.rerank_topk
        if hasattr(self, 'feature_extraction_mode'):
            params.feature_extraction_mode = self.feature_extraction_mode
        if hasattr(self, 'ltr_topk'):
            params.ltr_topk = self.ltr_topk
        if hasattr(self, 'use_lgbm_reranker'):
            params.use_lgbm_reranker = self.use_lgbm_reranker
        if hasattr(self, 'lgbm_model_path'):
            params.lgbm_model_path = self.lgbm_model_path
        if hasattr(self, 'ltr_ambiguous_only'):
            params.ltr_ambiguous_only = self.ltr_ambiguous_only
        if hasattr(self, 'ltr_fallback_top1'):
            params.ltr_fallback_top1 = self.ltr_fallback_top1
        if hasattr(self, 'use_tcr_reranker'):
            params.use_tcr_reranker = self.use_tcr_reranker
        if hasattr(self, 'tcr_model_path'):
            params.tcr_model_path = self.tcr_model_path
        if hasattr(self, 'tcr_gate_path'):
            params.tcr_gate_path = self.tcr_gate_path
        if hasattr(self, 'tcr_topk'):
            params.tcr_topk = self.tcr_topk
        if hasattr(self, 'tcr_gate_thr'):
            params.tcr_gate_thr = self.tcr_gate_thr
        if hasattr(self, 'tcr_margin'):
            params.tcr_margin = self.tcr_margin
        for name in [
            'e27_mode', 'e27_bank_size', 'e27_bank_topk', 'e27_use_offsets',
            'e27_offset_topk', 'e27_max_candidates', 'e27_delta_score',
            'e27_margin_thresh', 'e27_strict_delta_score',
            'e27_strict_margin_thresh', 'e27_bank23_delta_score',
            'e27_bank23_margin_thresh', 'e27_apce_low', 'e27_trigger_policy', 'e27_fallback',
            'e27_reset_policy', 'e27_cooldown', 'e27_long_unreliable',
            'e27_include_sources', 'e27_exclude_sources',
            'e27_dynamic_history', 'e27_dynamic_min_frames',
            'e27_score_low_q', 'e27_score_gray_q', 'e27_apce_low_q',
            'e27_bank_ranker', 'e27_local_neighborhood',
            'e27_state_quarantine', 'e27_reset_confidence',
            'e27_reset_confidence_thresh',
            'e29_c1_fallback_only', 'e29_trigger_budget_rate',
            'e29_relative_gray_min_frames', 'e29_relative_high_score_protect',
            'e29_reject_fallback', 'e29_delayed_bank_write',
            'e30_confirmation_frames', 'e30_source_aware_confirmation',
            'e30_confirm_frames_c1', 'e30_confirm_frames_bank23',
            'e30_confirm_frames_offset', 'e30_rollback',
            'e30_rollback_apce_ratio', 'e30_rollback_margin_thresh',
            'e30_rollback_score_thresh',
            'use_e31_verifier', 'e31_model_path', 'e31_prob_thresh'
        ]:
            if hasattr(self, name):
                setattr(params, name, getattr(self, name))

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []
        if getattr(tracker.params, 'dump_frames', False):
            output['frame_diag'] = []
        output['failure_prob'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        frame_iter = enumerate(seq.frames[1:], start=1)
        if HAS_TQDM:
            frame_iter = tqdm(frame_iter, total=len(seq.frames) - 1,
                              desc=f'  {seq.name}', unit='f', leave=False,
                              bar_format='{desc}: {percentage:3.0f}%|{bar:10}{r_bar}')
        for frame_num, frame_path in frame_iter:
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



