"""Frozen ROMTrack + ReSeT transfer configuration."""

from lib.test.parameter.ROMTrack import parameters as base_parameters


def parameters(_: str):
    params = base_parameters("antiuav410_fair")
    params.dump_frames = True
    params.commit_protocol = "legacy"
    params.relative_acceptance = "none"
    params.calibration_mode = "fixed"
    params.calibration_params = {}
    return params
