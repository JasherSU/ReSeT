"""Portable ORTrack workspace paths configured without machine-specific literals."""

import os
from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        repository = Path(__file__).resolve().parents[3]
        output = Path(os.environ.get("RESET_OUTPUT_ROOT", repository / "output")).expanduser()
        dataset = Path(os.environ.get("ANTIUAV410_ROOT", repository / "data"))

        self.workspace_dir = str(output)
        self.tensorboard_dir = str(output / "tensorboard")
        self.pretrained_networks = str(output / "pretrained_networks")
        self.antiuav410_dir = str(dataset.expanduser())
        self.lasot_dir = ""
        self.got10k_dir = ""
        self.got10k_val_dir = ""
        self.got10k_lmdb_dir = ""
        self.trackingnet_dir = ""
        self.trackingnet_lmdb_dir = ""
        self.coco_dir = ""
        self.coco_lmdb_dir = ""
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = ""
        self.imagenet_lmdb_dir = ""
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""
