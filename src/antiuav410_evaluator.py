from __future__ import absolute_import, division, print_function

import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import six


def rect_iou(rects1, rects2, bound=None):
    assert rects1.shape == rects2.shape
    if bound is not None:
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])
    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)
    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter
    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)
    index = rects2 > 0
    idx = (np.sum(index, axis=1) == 4)
    ious[~idx] = -1
    return ious


def _intersection(rects1, rects2):
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2], rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3], rects2[..., 1] + rects2[..., 3])
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)
    return np.stack([x1, y1, w, h]).T


def center_error(rects1, rects2):
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))
    index = rects2 > 0
    idx = (np.sum(index, axis=1) == 4)
    errors[~idx] = -1
    return errors


class AntiUAV410(object):
    def __init__(self, root_dir):
        super(AntiUAV410, self).__init__()
        self.root_dir = root_dir
        self.anno_files = sorted(glob.glob(os.path.join(root_dir, '*/IR_label.json')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if index not in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)
        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg')))
        with open(self.anno_files[index], 'r') as f:
            label_res = json.load(f)
        assert len(img_files) == len(label_res['gt_rect'])
        assert len(label_res['gt_rect'][0]) == 4
        return img_files, label_res

    def __len__(self):
        return len(self.seq_names)


class ExperimentAntiUAV410(object):
    def __init__(self, root_dir, subset, annos_dir='annos',
                 report_dir='reports', start_idx=0, end_idx=None):
        super(ExperimentAntiUAV410, self).__init__()
        self.subset = subset
        self.dataset = AntiUAV410(os.path.join(root_dir, subset))
        self.annos_dir = annos_dir
        self.report_dir = os.path.join(report_dir, 'AntiUAV410', subset)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.att_name = ['Thermal Crossover', 'Out-of-View', 'Scale Variation',
                         'Fast Motion', 'Occlusion', 'Dynamic Background Clutter',
                         'Tiny Size', 'Small Size', 'Medium Size', 'Normal Size']
        self.att_fig_name = ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC', 'TS', 'SS', 'MS', 'NS']

    def iou(self, bbox1, bbox2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x0_1, y0_1, w1_1, h1_1) = bbox1
        (x0_2, y0_2, w1_2, h1_2) = bbox2
        x1_1 = x0_1 + w1_1
        x1_2 = x0_2 + w1_2
        y1_1 = y0_1 + h1_1
        y1_2 = y0_2 + h1_2
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        return size_intersection / size_union

    def not_exist(self, pred):
        if len(pred) == 1 or len(pred) == 0:
            return 1.0
        else:
            return 0.0

    def eval(self, out_res, label_res):
        measure_per_frame = []
        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
            if not _exist:
                measure_per_frame.append(self.not_exist(_pred))
            else:
                if len(_gt) < 4 or sum(_gt) == 0:
                    continue
                if len(_pred) == 4:
                    measure_per_frame.append(self.iou(_pred, _gt))
                else:
                    measure_per_frame.append(0.0)
        return np.mean(measure_per_frame)

    def report(self, trackers, plot_curves=True, plot_attcurves=True):
        assert isinstance(trackers, (list, tuple))
        if not isinstance(trackers[0], dict):
            trackers = [{'name': trackers[0], 'path': trackers[0], 'mode': 1}]

        report_dir = self.report_dir
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        tracker_names = []
        overall_SAs = []

        for trackerid, tracker in enumerate(trackers):
            name = tracker['name']
            mode = tracker['mode']
            tracker_names.append(name)
            print('Evaluating %d trackers on %d sequences' % (len(trackers), len(self.dataset)))
            print('Tracker: %s' % name)
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speeds = np.zeros(seq_num)

            performance.update({name: {
                'overall': {},
                'TC': {}, 'OV': {}, 'SV': {}, 'FM': {}, 'OC': {},
                'DBC': {}, 'TS': {}, 'SS': {}, 'MS': {}, 'NS': {},
                'seq_wise': {}}})

            overall_SA = []
            att_list = []

            for s, (_, label_res) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(tracker['path'], '%s.txt' % seq_name)

                att_file = os.path.join(self.annos_dir, self.subset, 'att', '%s.txt' % seq_name)
                if os.path.exists(att_file):
                    with open(att_file, 'r') as f:
                        att_temp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                else:
                    att_temp = np.zeros(len(self.att_name), dtype=float)
                att_list.append(att_temp)

                try:
                    with open(record_file, 'r') as f:
                        boxestemp = json.load(f)['res']
                except:
                    with open(record_file, 'r') as f:
                        boxestemp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

                if mode == 2:
                    boxestemp[:, 2:] = boxestemp[:, 2:] - boxestemp[:, :2] + 1

                SA_Score = self.eval(boxestemp, label_res)
                overall_SA.append(SA_Score)

                boxes = []
                for box in boxestemp:
                    if len(box) == 4:
                        boxes.append(box)
                    else:
                        boxes.append([0, 0, 0, 0])
                boxes = np.array(boxes)

                anno = []
                annotemp = label_res['gt_rect']
                for ann in annotemp:
                    if len(ann) == 4:
                        anno.append(ann)
                    else:
                        anno.append([0, 0, 0, 0])
                anno = np.array(anno)

                boxes[0] = anno[0]
                if not (len(boxes) == len(anno)):
                    print('warning: %s anno do not match boxes' % seq_name)
                    len_min = min(len(boxes), len(anno))
                    boxes = boxes[:len_min]
                    anno = anno[:len_min]
                assert len(boxes) == len(anno)

                ious, center_errors = self._calc_metrics(boxes, anno)
                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                performance[name]['seq_wise'].update({seq_name: {
                    'success_curve': succ_curve[s].tolist(),
                    'precision_curve': prec_curve[s].tolist(),
                    'success_score': np.mean(succ_curve[s]),
                    'precision_score': prec_curve[s][20],
                    'success_rate': succ_curve[s][self.nbins_iou // 2],
                    'speed_fps': -1}})

            overall_SAs.append(np.mean(overall_SA))

            all_succ_curve = succ_curve.copy()
            all_prec_curve = prec_curve.copy()

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)

            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]

            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': -1})

            att_array = np.array(att_list)
            for ii in range(len(self.att_name)):
                att_ids = np.where(att_array[:, ii] > 0)[0]
                if trackerid == 0:
                    self.att_name[ii] = self.att_name[ii] + '(' + str(len(att_ids)) + ')'
                att_succ_curve = all_succ_curve[att_ids, :]
                att_prec_curve = all_prec_curve[att_ids, :]
                att_succ_curve = np.mean(att_succ_curve, axis=0)
                att_prec_curve = np.mean(att_prec_curve, axis=0)
                att_succ_score = np.mean(att_succ_curve)
                att_prec_score = att_prec_curve[20]
                att_succ_rate = att_succ_curve[self.nbins_iou // 2]
                performance[name][self.att_fig_name[ii]].update({
                    'att_success_curve': att_succ_curve.tolist(),
                    'att_precision_curve': att_prec_curve.tolist(),
                    'att_success_score': att_succ_score,
                    'att_precision_score': att_prec_score,
                    'att_success_rate': att_succ_rate})

        sa_file = os.path.join(report_dir, 'state_accuracy_scores.txt')
        if os.path.exists(sa_file):
            os.remove(sa_file)
        text = '[Overall] %20s %20s: %s' % ('Tracker name', 'Experiments metric', 'Scores')
        with open(sa_file, 'a', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')
        print(text)

        for ii in range(len(overall_SAs)):
            text = '[Overall] %20s %20s: %.04f' % (trackers[ii]['name'], 'State accuracy', overall_SAs[ii])
            with open(sa_file, 'a', encoding='utf-8') as f:
                f.write(text)
                f.write('\n')
            print(text)

        print('Saving state accuracy scores to', sa_file)
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        if plot_curves:
            self.plot_curves(tracker_names)
        if plot_attcurves:
            for ii in range(len(self.att_name)):
                self.plot_attcurves(tracker_names, self.att_name[ii], self.att_fig_name[ii])

        return performance

    def _calc_metrics(self, boxes, anno):
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)
        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)
        return succ_curve, prec_curve

    def plot_curves(self, tracker_names):
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file)
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.pdf')
        prec_file = os.path.join(report_dir, 'precision_plots.pdf')
        key = 'overall'

        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})

        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['success_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0), title='Success plots of OPE')
        ax.set_title('Success plots of OPE', fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving success plots to', succ_file)
        fig.savefig(succ_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
        plt.close(fig)

        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['precision_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0), title='Precision plots of OPE')
        ax.set_title('Precision plots of OPE', fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_attcurves(self, tracker_names, att_name, att_key):
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file)
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots_of_' + att_key + '.pdf')
        prec_file = os.path.join(report_dir, 'precision_plots_of_' + att_key + '.pdf')

        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        tracker_names = list(performance.keys())
        succ = [t[att_key]['att_success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][att_key]['att_success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_success_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_success_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0), title='Success plots of OPE - ' + att_name)
        ax.set_title('Success plots of OPE - ' + att_name, fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving success plots to', succ_file)
        fig.savefig(succ_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
        plt.close(fig)

        tracker_names = list(performance.keys())
        prec = [t[att_key]['att_precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][att_key]['att_precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_precision_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_precision_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0), title='Precision plots of OPE - ' + att_name)
        ax.set_title('Precision plots of OPE - ' + att_name, fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
        plt.close(fig)
