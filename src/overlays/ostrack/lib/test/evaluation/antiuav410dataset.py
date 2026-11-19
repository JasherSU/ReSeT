import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class AntiUAV410Dataset(BaseDataset):
    """ Anti-UAV410 test dataset for infrared UAV tracking evaluation.

    Each sequence is a directory under <root>/<split>/ containing
    IR images (000001.jpg, 000002.jpg, ...) and IR_label.json with:
        {"gt_rect": [[x,y,w,h], ...], "exist": [1/0, ...]}
    """

    def __init__(self, split='test'):
        super().__init__()
        self.base_path = self.env_settings.antiuav410_path
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, self.split, sequence_name)
        anno_file = os.path.join(seq_path, 'IR_label.json')

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        gt_rect = np.array(anno['gt_rect'], dtype=np.float64)
        exist = np.array(anno['exist'], dtype=np.float64)

        # target_visible: exist=1 means target is visible
        target_visible = (exist > 0)

        frames_list = ['{}/{:06d}.jpg'.format(seq_path, frame_number + 1)
                       for frame_number in range(gt_rect.shape[0])]

        return Sequence(sequence_name, frames_list, 'antiuav410',
                        ground_truth_rect=gt_rect.reshape(-1, 4),
                        object_class='uav',
                        target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        split_dir = os.path.join(self.base_path, self.split)
        seqs = []
        if not os.path.isdir(split_dir):
            return seqs
        for entry in sorted(os.listdir(split_dir)):
            seq_path = os.path.join(split_dir, entry)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, 'IR_label.json')):
                seqs.append(entry)
        return seqs
