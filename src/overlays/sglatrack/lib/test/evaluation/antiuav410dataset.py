import json
import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class AntiUAV410Dataset(BaseDataset):
    """ Anti-UAV410 test dataset for infrared UAV tracking evaluation. """

    def __init__(self, split='test'):
        super().__init__()
        self.base_path = self.env_settings.antiuav410_path
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, self.split, sequence_name)
        json_path = os.path.join(seq_path, 'IR_label.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        gt_rect = np.array(data['gt_rect'], dtype=np.float64)
        exist = np.array(data['exist'], dtype=np.float64)

        # Build target_visible from exist
        target_visible = exist > 0

        frames_list = [
            os.path.join(seq_path, '{:06}.jpg'.format(frame_number + 1))
            for frame_number in range(gt_rect.shape[0])
        ]

        return Sequence(sequence_name, frames_list, 'antiuav410',
                        gt_rect.reshape(-1, 4),
                        object_class='uav',
                        target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        split_dir = os.path.join(self.base_path, self.split)
        seq_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
            and os.path.exists(os.path.join(split_dir, d, 'IR_label.json'))
        ])
        return seq_names
