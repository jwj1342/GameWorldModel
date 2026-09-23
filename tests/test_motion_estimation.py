"""Synthetic observations exercise motion evidence without loading any model."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from gwm.perception.motion import estimate_motion


CFG = {"perception": {"motion": {"min_support_frames": 5, "min_temporal_span_s": 0.75,
                                 "periodic_min_cycles": 1.5, "smooth_window": 1}}}


def _quats(times, degrees=None):
    if degrees is None:
        degrees = np.zeros(len(times))
    return R.from_euler("y", degrees, degrees=True).as_quat()


def _result(times, positions, degrees=None, **kwargs):
    return estimate_motion(times, positions, _quats(times, degrees), CFG, **kwargs)


def test_moving_camera_static_object_is_static_in_world():
    times = np.linspace(0, 4, 17)
    camera = np.stack([0.5 * times, np.zeros_like(times), np.zeros_like(times)], axis=1)
    local = np.tile([2.0, 0.5, -3.0], (len(times), 1)) - camera
    result = _result(times, local, coordinate_space="camera", camera_positions=camera,
                     camera_quaternions=_quats(times))
    assert result["motion_guess"]["type"] == "static"
    assert result["motion_guess"]["coordinate_space"] == "world"
    assert result["motion_guess"]["support_frames"] == 17


def test_rotating_camera_static_object_is_static_in_world():
    times = np.linspace(0, 3, 13)
    camera_quat = _quats(times, 15 * times)
    camera_rotation = R.from_quat(camera_quat)
    world_position = np.tile([2.0, 0.5, -3.0], (len(times), 1))
    local_position = camera_rotation.inv().apply(world_position)
    local_quat = camera_rotation.inv().as_quat()
    result = estimate_motion(times, local_position, local_quat, CFG, coordinate_space="camera",
                             camera_positions=np.zeros_like(local_position), camera_quaternions=camera_quat)
    assert result["motion_guess"]["type"] == "static"


def test_uniform_translation_and_noisy_translation():
    times = np.linspace(0, 4, 17)
    positions = np.stack([0.4 * times, np.ones_like(times), np.zeros_like(times)], axis=1)
    for noise in (0.0, 0.005):
        perturbed = positions + noise * np.random.default_rng(4).normal(size=positions.shape)
        guess = _result(times, perturbed)["motion_guess"]
        assert guess["type"] == "prismatic"
        assert guess["residual"] < 0.03
        assert guess["candidate_margin"] >= 0.1


def test_rotation_about_world_y_axis():
    times = np.linspace(0, 3, 17)
    positions = np.tile([1.0, 0.5, 0.0], (len(times), 1))
    guess = _result(times, positions, degrees=30 * times)["motion_guess"]
    assert guess["type"] == "spin"
    assert abs(guess["rate"] - 30) < 1
    assert abs(guess["axis"][1]) > 0.99


def test_revolute_arc_has_a_pivot_candidate():
    times = np.linspace(0, 3, 17)
    angle = 25 * times
    radians = np.radians(angle)
    positions = np.stack([1.2 * np.cos(radians), np.full_like(times, 0.5),
                          1.2 * np.sin(radians)], axis=1)
    guess = _result(times, positions, degrees=angle)["motion_guess"]
    assert guess["type"] == "revolute"
    assert np.allclose(guess["pivot"], [0.0, 0.5, 0.0], atol=0.1)
    assert guess["residual_unit"] == "m"


def test_complete_translation_and_rotation_periods():
    times = np.linspace(0, 8, 49)
    wave = np.sin(2 * np.pi * times / 4)
    positions = np.stack([wave, np.ones_like(times), np.zeros_like(times)], axis=1)
    guess = _result(times, positions)["motion_guess"]
    assert guess["type"] == "periodic_translate"
    assert abs(guess["period"] - 4) < 0.15
    fixed = np.tile([0.0, 0.5, 0.0], (len(times), 1))
    rotation = _result(times, fixed, degrees=35 * wave)["motion_guess"]
    assert rotation["type"] == "periodic_rotate"
    assert abs(rotation["period"] - 4) < 0.15


def test_incomplete_period_is_not_claimed_as_periodic():
    times = np.linspace(0, 2.5, 21)
    wave = np.sin(2 * np.pi * times / 4)
    positions = np.stack([wave, np.ones_like(times), np.zeros_like(times)], axis=1)
    result = _result(times, positions)
    assert result["motion_guess"]["type"] == "unknown"
    assert result["hypothesis"] is not None


def test_missing_frames_and_severe_occlusion_return_unknown():
    times = np.array([0, 0.25, 0.5, 0.75, 3, 3.25, 3.5, 3.75])
    positions = np.stack([0.5 * times, np.zeros_like(times), np.zeros_like(times)], axis=1)
    assert _result(times, positions)["motion_guess"]["type"] == "unknown"
    dense_times = np.linspace(0, 3, 13)
    dense_positions = np.stack([0.5 * dense_times, np.zeros_like(dense_times), np.zeros_like(dense_times)], axis=1)
    assert _result(dense_times, dense_positions, visible_fractions=np.full(len(dense_times), 0.2))["motion_guess"]["type"] == "unknown"


def test_camera_pose_missing_and_insufficient_evidence_are_unknown():
    times = np.linspace(0, 3, 13)
    positions = np.tile([0.0, 0.5, 0.0], (len(times), 1))
    missing = _result(times, positions, coordinate_space="camera")
    assert missing["motion_guess"]["type"] == "unknown"
    assert "camera pose missing" in missing["motion_guess"]["notes"]
    untrusted = _result(times, positions, camera_pose_available=False)
    assert untrusted["motion_guess"]["type"] == "unknown"
    short = _result(times[:3], positions[:3])
    assert short["motion_guess"]["type"] == "unknown"
    assert short["motion_guess"]["conf"] == "unknown"


def test_unreliable_orientation_does_not_force_static_or_spin():
    times = np.linspace(0, 3, 13)
    positions = np.tile([0.0, 0.5, 0.0], (len(times), 1))
    guess = _result(times, positions, degrees=30 * times, orientation_reliable=False)["motion_guess"]
    assert guess["type"] == "unknown"
    assert "orientation" in guess["notes"]
