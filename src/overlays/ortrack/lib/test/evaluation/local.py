"""Portable ORTrack test paths configured through environment variables."""

import os
from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    repository = Path(__file__).resolve().parents[3]
    output = Path(os.environ.get("RESET_OUTPUT_ROOT", repository / "output")).expanduser()
    dataset = os.environ.get("ANTIUAV410_ROOT")
    if not dataset:
        raise RuntimeError("Set ANTIUAV410_ROOT to the AntiUAV410 dataset root")

    settings.prj_dir = str(repository)
    settings.save_dir = str(output)
    settings.results_path = str(output / "test" / "tracking_results")
    settings.segmentation_path = str(output / "test" / "segmentation_results")
    settings.network_path = str(output / "test" / "networks")
    settings.result_plot_path = str(output / "test" / "result_plots")
    settings.antiuav410_dir = str(Path(dataset).expanduser())
    settings.antiuav410_path = settings.antiuav410_dir
    return settings
