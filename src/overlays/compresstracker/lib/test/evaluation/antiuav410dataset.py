import numpy as np
import os
import json
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class AntiUAV410Dataset(BaseDataset):
    """ Anti-UAV410 test dataset for infrared UAV tracking.

    Each sequence directory contains IR images (000001.jpg, 000002.jpg, ...)
    and an IR_label.json file with format:
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

        # Read annotation
        anno_file = os.path.join(seq_path, 'IR_label.json')
        with open(anno_file, 'r') as f:
            data = json.load(f)
        gt_rect = np.array(data['gt_rect'], dtype=np.float64)
        exist = np.array(data.get('exist', [1] * len(gt_rect)), dtype=np.float64)

        # Build frame list
        num_frames = gt_rect.shape[0]
        frames_list = [os.path.join(seq_path, '{:06d}.jpg'.format(i + 1)) for i in range(num_frames)]

        # Target visible: exist==1 and valid bbox
        target_visible = (exist == 1) & (gt_rect[:, 2] > 0) & (gt_rect[:, 3] > 0)

        return Sequence(sequence_name, frames_list, 'antiuav410', gt_rect.reshape(-1, 4),
                        object_class='uav', target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        split_dir = os.path.join(self.base_path, self.split)
        seqs = sorted([d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))
                       and os.path.exists(os.path.join(split_dir, d, 'IR_label.json'))])
        return seqs
