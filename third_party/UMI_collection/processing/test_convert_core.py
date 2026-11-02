#!/usr/bin/env python
"""Hardware-free, dep-light logic tests for the UMI converter core.

Covers the numerically load-bearing pieces (resampling, quaternion convention,
gripper calibration, anti-alias wrench downsample, crop geometry, timeline
hygiene) using numpy + scipy only — runs in the collection .venv.

Run (no pytest needed — runs as a script too):
    .venv/bin/python processing/test_convert_core.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import umi_convert_core as core  # noqa: E402


def test_strictly_increasing_drops_stale_resend():
    # RealSense resends the last frame on reconnect -> duplicate/backward t.
    t = np.array([0.0, 0.1, 0.1, 0.05, 0.2])
    keep = core.make_strictly_increasing(t)
    kept = t[keep]
    assert np.all(np.diff(kept) > 0), kept
    assert kept.tolist() == [0.0, 0.1, 0.2]


def test_quat_wxyz_to_xyzw_roundtrip():
    # A known 90deg rotation about z.
    r = Rotation.from_euler("z", 90, degrees=True)
    xyzw = r.as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    back = core.quat_wxyz_to_xyzw(wxyz)
    assert np.allclose(back, xyzw, atol=1e-9) or np.allclose(back, -xyzw, atol=1e-9)


def test_deglitch_drops_isolated_teleport_and_keeps_neighbors():
    # A hand moving smoothly at ~0.5 m/s with ONE lighthouse teleport at index 3
    # that overshoots ~0.6 m in one 13ms sample (~46 m/s) then returns. The glitch
    # sample must be dropped; every real sample must be kept.
    t = np.arange(8) * 0.013                       # ~77 Hz
    pos = np.zeros((8, 3))
    pos[:, 0] = np.arange(8) * 0.0065              # 0.5 m/s along x, smooth
    pos[3, 0] += 0.6                               # teleport out at 3, back at 4
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert keep.tolist() == [0, 1, 2, 4, 5, 6, 7], keep.tolist()


def test_deglitch_noop_on_clean_motion():
    # Fast but PLAUSIBLE hand motion (2 m/s) must be untouched — no false positives.
    t = np.arange(20) * 0.013
    pos = np.zeros((20, 3))
    pos[:, 0] = np.arange(20) * (2.0 * 0.013)      # 2 m/s < 3 m/s gate
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert keep.tolist() == list(range(20))


def test_deglitch_keeps_genuine_fast_reposition():
    # A real fast move that ENDS SOMEWHERE NEW (net displacement stays high) must
    # NOT be dropped — the net-speed return test never rejoins the start position.
    t = np.arange(20) * 0.013
    pos = np.zeros((20, 3))
    disp = np.zeros(20)
    disp[5:] = 0.6                                 # steps to a new place and stays
    pos[:, 0] = disp
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0, max_run=30)
    assert keep.tolist() == list(range(20)), "a move to a new pose must be kept"


def test_deglitch_bridges_long_drift_excursion():
    # The real-data signature: jump off the path, WANDER far from the true position
    # for ~20 samples, then snap back near where it left. The whole excursion is
    # dropped and bridged, because sample r rejoins the pre-glitch position pos[k].
    t = np.arange(40) * 0.013
    pos = np.zeros((40, 3))
    pos[:, 0] = 0.5                                # sitting still at x=0.5
    pos[10:32, 0] = 0.9 + 0.02 * np.arange(22)     # drift out (0.9..1.32), all far
    # sample 32 returns to 0.5 (within rejoin_dist_m of the pre-glitch baseline)
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0, max_run=30)
    # samples 10..31 (the excursion) dropped; 0..9 and 32..39 kept
    assert keep.tolist() == list(range(10)) + list(range(32, 40)), keep.tolist()
    kp = pos[keep]
    sp = np.linalg.norm(np.diff(kp, axis=0), axis=1) / np.diff(t[keep])
    assert sp.max() < 3.0


def test_deglitch_slow_drift_that_parks_is_not_falsely_rejoined():
    # Guards the reason we use position-rejoin not net-speed: a SLOW drift that
    # jumps off and parks at a wrong place accumulates enough elapsed time that its
    # AVERAGE speed would fall under the gate — a net-speed test would wrongly
    # "rejoin" while still parked. Position-rejoin only bridges a TRUE return, so
    # here (never returns to 0.5) nothing is dropped.
    t = np.arange(40) * 0.013
    pos = np.zeros((40, 3))
    pos[:, 0] = 0.5
    pos[10:, 0] = 1.2                              # jumps off and parks, never back
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0, max_run=30)
    assert keep.tolist() == list(range(40)), "parked drift must not be bridged away"


def test_deglitch_leaves_unbridgeable_excursion_untouched():
    # An excursion LONGER than max_run with no return in the window has no clean
    # anchor — held as-is (the episode should be dropped elsewhere, not bridged).
    t = np.arange(40) * 0.013
    pos = np.zeros((40, 3))
    pos[:, 0] = 0.5
    pos[10:, 0] = 5.0                              # jumps away and never returns
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0, max_run=6)
    assert keep.tolist() == list(range(40)), "no return within max_run -> keep all"


def test_deglitch_never_drops_endpoints_and_short_input():
    # First sample anchors the clamp and is never dropped; <3 samples is a no-op.
    assert core.deglitch_tracker_positions(np.array([0.0, 1.0]),
                                           np.zeros((2, 3))).tolist() == [0, 1]
    # A huge jump AT the first step then return: sample 0 kept, sample 1 dropped.
    t = np.arange(5) * 0.013
    pos = np.zeros((5, 3)); pos[1, 0] = 0.6
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert 0 in keep and keep[-1] == 4 and 1 not in keep


def test_deglitch_row_drop_keeps_arrays_aligned():
    # The returned indices must apply identically to t, pos AND quat so all three
    # stay the same length and t stays strictly increasing after a drop.
    t = np.arange(6) * 0.013
    pos = np.zeros((6, 3)); pos[:, 0] = np.arange(6) * 0.0065
    pos[2, 0] += 0.6                               # single teleport at index 2
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (6, 1))
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    tk, pk, qk = t[keep], pos[keep], quat[keep]
    assert tk.shape[0] == pk.shape[0] == qk.shape[0] == 5
    assert np.all(np.diff(tk) > 0)                 # still strictly increasing
    sp = np.linalg.norm(np.diff(pk, axis=0), axis=1) / np.diff(tk)
    assert sp.max() < 3.0, sp.max()


def test_tracker_glitch_severity_reports_bridge_and_residual():
    # A single bridgeable spike (8 samples, spike at 3): pin the EXACT arithmetic.
    t = np.arange(8) * 0.013
    pos = np.zeros((8, 3)); pos[:, 0] = np.arange(8) * 0.0065
    pos[3, 0] += 0.6
    s = core.tracker_glitch_severity(t, pos, max_speed_mps=3.0)
    assert s["n"] == 8
    assert s["n_hot_steps"] == 2                 # the out-jump AND the return-jump
    assert s["n_dropped"] == 1 and s["n_unbridged"] == 0
    assert abs(s["dropped_frac"] - 0.125) < 1e-9
    assert 0.6 < s["max_jump_m"] < 0.61 and 46 < s["peak_speed_mps"] < 47
    # after bridging, the kept stream is clean (residual ~ the smooth 0.5 m/s)
    assert s["residual_max_jump_m"] < 0.05 and s["residual_peak_speed_mps"] < 3.0

    # An unbridgeable runaway -> nothing dropped; EXACTLY ONE unbridged excursion
    # (not a hot-edge count), and a big residual jump left on the kept stream — the
    # two signals a QA gate uses to DROP the whole episode instead of bridging.
    t2 = np.arange(40) * 0.013
    pos2 = np.zeros((40, 3)); pos2[:, 0] = 0.5; pos2[10:, 0] = 5.0
    s2 = core.tracker_glitch_severity(t2, pos2, max_speed_mps=3.0, max_run=6)
    assert s2["n_dropped"] == 0
    assert s2["n_unbridged"] == 1                 # one excursion, not 2 hot edges
    assert s2["residual_max_jump_m"] > 0.3        # 4.5m jump survives -> drop signal


def test_severity_bridged_spike_does_not_count_as_unbridged():
    # Regression for the review finding: a cleanly BRIDGED spike where the hand
    # truly moved during the occlusion leaves a fast bridge step, which must NOT be
    # miscounted as an unbridged excursion (that over-dropped good episodes before).
    t = np.arange(30) * 0.013
    pos = np.zeros((30, 3))
    pos[:, 0] = np.arange(30) * 0.0065
    for i in (5, 12, 20):                          # three isolated bridgeable spikes
        pos[i, 0] += 0.6
    s = core.tracker_glitch_severity(t, pos, max_speed_mps=3.0)
    assert s["n_dropped"] == 3
    assert s["n_unbridged"] == 0                   # all three bridged, none residual


def test_deglitch_return_stats_counts_excursions():
    # return_stats gives per-excursion counts straight from the scan.
    t = np.arange(16) * 0.013
    pos = np.zeros((16, 3)); pos[:, 0] = np.arange(16) * 0.0065
    pos[3, 0] += 0.6; pos[9, 0] += 0.6            # two separate bridgeable spikes
    keep, stats = core.deglitch_tracker_positions(t, pos, return_stats=True)
    assert stats["n_bridged_excursions"] == 2 and stats["n_unbridged_excursions"] == 0
    assert set(range(16)) - set(keep.tolist()) == {3, 9}


def test_bridged_spans_bracket_the_interpolated_hole():
    """REGRESSION (real labelling hole). Bridging is NOT repairing: the dropped samples
    leave a gap that interp1d/Slerp cross with a STRAIGHT LINE, so wherever the hand
    really moved during the occlusion the label is a fabricated ramp. That has to be
    reported, or nothing downstream can mask it.

    This was live on the shipped clean94 dataset: episodes 30/31 (takes 033/034) carry
    118 mm and 127 mm ONE-FRAME jumps in robot0_eef_pos (1.76 / 1.91 m/s, vs a 0.506 m/s
    max across the other 89 episodes) while robot0_pose_glitch == 0 on every frame,
    because ``unbridged_spans`` is empty for a bridged excursion BY CONSTRUCTION.
    """
    # A ~200 ms occlusion (16 samples at 13 ms) during which the hand moves 95 mm — the
    # exact ep30/ep31 signature. Teleport out, wander, then rejoin 95 mm along the path.
    t = np.arange(40) * 0.013
    pos = np.zeros((40, 3))
    pos[:, 0] = 0.5
    pos[10:26, 0] = 1.4                            # excursion: far away, then returns
    pos[26:, 0] = 0.595                            # rejoined, 95 mm further along
    keep, stats = core.deglitch_tracker_positions(t, pos, return_stats=True)
    assert stats["n_bridged_excursions"] == 1
    assert stats["n_unbridged_excursions"] == 0     # so unbridged_spans CANNOT flag it
    assert stats["unbridged_spans"] == []
    # ...but bridged_spans does, bracketing exactly the interpolated hole [t[9], t[26]].
    assert len(stats["bridged_spans"]) == 1
    lo, hi = stats["bridged_spans"][0]
    assert lo == t[9] and hi == t[26], (lo, hi)
    # The hole really is a fabricated ramp: the kept stream jumps 95 mm in one step.
    kp = pos[keep]
    assert abs(float(np.linalg.norm(kp[10] - kp[9])) - 0.095) < 1e-9
    # tracker_glitch_severity must forward it, since that is what the converter reads.
    sev = core.tracker_glitch_severity(t, pos)
    assert sev["bridged_spans"] == stats["bridged_spans"]
    assert sev["n_unbridged"] == 0


def test_clean_stream_reports_no_spans_at_all():
    # The flip side: a clean stream must not manufacture spans (they mask training data).
    t = np.arange(40) * 0.013
    pos = np.zeros((40, 3)); pos[:, 0] = np.arange(40) * 0.006
    sev = core.tracker_glitch_severity(t, pos)
    assert sev["bridged_spans"] == [] and sev["unbridged_spans"] == []
    assert sev["n_dropped"] == 0
    # ...including on inputs too short to scan (the early-return paths).
    _, st = core.deglitch_tracker_positions(t[:2], pos[:2], return_stats=True)
    assert st["bridged_spans"] == [] and st["unbridged_spans"] == []


def test_deglitch_two_separate_excursions():
    # The while-loop must catch a SECOND glitch after bridging the first (k=r resume).
    t = np.arange(16) * 0.013
    pos = np.zeros((16, 3)); pos[:, 0] = np.arange(16) * 0.0065
    pos[3, 0] += 0.6; pos[9, 0] += 0.6
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert keep.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]


def test_deglitch_multi_axis_glitch_uses_euclidean_norm():
    # A diagonal teleport whose PER-AXIS components (0.42, 0.42) are each large but
    # whose combined 3D norm (0.59 m) is the real jump — pins norm, not per-axis/L1.
    t = np.arange(8) * 0.013
    pos = np.zeros((8, 3))
    pos[3, 1] += 0.42; pos[3, 2] += 0.42          # y AND z jump together, x stays 0
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert keep.tolist() == [0, 1, 2, 4, 5, 6, 7], keep.tolist()


def test_deglitch_last_sample_glitch_is_unbridged():
    # A glitch on the FINAL sample has no later sample to rejoin -> cannot be
    # bridged (documented limitation). Pin that QA's drop signal fires for it.
    t = np.arange(8) * 0.013
    pos = np.zeros((8, 3)); pos[:, 0] = np.arange(8) * 0.0065
    pos[7, 0] += 0.6                               # glitch at the last sample
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    assert keep.tolist() == list(range(8))         # last sample never dropped
    s = core.tracker_glitch_severity(t, pos, max_speed_mps=3.0)
    assert s["n_hot_steps"] == 1 and s["n_dropped"] == 0
    assert s["n_unbridged"] == 1                   # QA flags it for episode drop


def test_deglitch_rejoin_distance_boundary():
    # rejoin_dist_m is the bridge-vs-keep discriminator: an excursion returning
    # 0.15 m off the departure point is KEPT at 0.10 but BRIDGED at 0.20.
    t = np.arange(12) * 0.013
    pos = np.zeros((12, 3)); pos[:, 0] = 0.5
    pos[4:7, 0] = 1.5                              # excursion out
    pos[7:, 0] = 0.5 + 0.15                        # returns 0.15 m off baseline
    keep_tight = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0,
                                                 rejoin_dist_m=0.10)
    keep_loose = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0,
                                                 rejoin_dist_m=0.20)
    assert keep_tight.tolist() == list(range(12))          # 0.15 > 0.10 -> not bridged
    assert 4 not in keep_loose and 5 not in keep_loose     # 0.15 <= 0.20 -> bridged


def test_deglitch_cleans_camera_grid_eef_pos_end_to_end():
    # The whole point: after keep-index + resample onto the CAMERA GRID, the
    # resulting eef_pos (the action/pose LABEL) must no longer carry the glitch.
    t = np.arange(20) * 0.013
    pos = np.zeros((20, 3)); pos[:, 0] = np.arange(20) * 0.0065
    pos[8, 0] += 0.6                               # teleport spike
    cam_t = np.linspace(t[0], t[-1], 8)
    raw = core.resample_position(t, pos, cam_t)
    raw_speed = np.linalg.norm(np.diff(raw, axis=0), axis=1) / np.diff(cam_t)
    assert raw_speed.max() > 10.0                  # glitch present in the raw label
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    fixed = core.resample_position(t[keep], pos[keep], cam_t)
    fixed_speed = np.linalg.norm(np.diff(fixed, axis=0), axis=1) / np.diff(cam_t)
    assert fixed_speed.max() < 3.0                 # label is clean after deglitch


def test_deglitch_row_drop_removes_bad_rotation_from_slerp():
    # Non-tautological rotation test: the glitch sample carries a DISTINCT wrong
    # rotation. Dropping the row must remove it from the Slerp knots so the resampled
    # rotation at the glitch time reads the neighbours' orientation, not the bad one.
    t = np.arange(6) * 0.013
    pos = np.zeros((6, 3)); pos[:, 0] = np.arange(6) * 0.0065
    pos[2, 0] += 0.6                               # position teleport at index 2
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (6, 1))  # identity everywhere
    # sample 2 additionally has a 90deg-about-z rotation (wxyz)
    quat[2] = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    t_spike = t[2:3]
    raw_rot = core.resample_rotation_wxyz(t, quat, t_spike)
    assert abs(raw_rot.as_euler("zyx", degrees=True)[0, 0] - 90.0) < 1e-3  # bad rot present
    keep = core.deglitch_tracker_positions(t, pos, max_speed_mps=3.0)
    fixed_rot = core.resample_rotation_wxyz(t[keep], quat[keep], t_spike)
    assert abs(fixed_rot.as_euler("zyx", degrees=True)[0, 0]) < 1e-3  # bad rot removed


def _rot_stream(n=40, dt=0.013, drift_deg_per_sample=0.3):
    """A slowly rotating tracker stream (wxyz), the QA baseline for the scan below."""
    t = np.arange(n) * dt
    ang = np.deg2rad(np.arange(n) * drift_deg_per_sample)
    quat = np.stack([np.cos(ang / 2), np.zeros(n), np.zeros(n), np.sin(ang / 2)], axis=1)
    return t, quat


def test_rotation_excursions_flags_there_and_back_and_ignores_clean_motion():
    t, quat = _rot_stream()
    assert core.rotation_excursions(t, quat) == []          # slow drift: nothing
    # One sample knocked 20 deg off-path about a DIFFERENT axis, then back = a glitch.
    bad = quat.copy()
    a = np.deg2rad(20.0)
    bad[20] = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    exc = core.rotation_excursions(t, bad)
    assert len(exc) == 1, exc
    assert 19.0 < exc[0]["amp_deg"] < 22.0, exc
    assert exc[0]["n_samples"] == 1, exc
    assert exc[0]["rate_dps"] > core.ROT_EXCURSION_RATE_DPS, exc


def test_rotation_excursions_keeps_genuine_fast_rotation():
    # A real fast wrist twist: ~1540 deg/s for 8 samples = 160 deg net, then it HOLDS
    # there. Far above the rate gate, but it ends somewhere NEW and never rejoins, so it
    # must NOT be flagged -- the same rejoin discriminator the position deglitch uses.
    t, _ = _rot_stream(n=40)
    n = len(t)
    deg = np.concatenate([np.arange(20) * 0.3, 6.0 + np.arange(1, 9) * 20.0,
                          np.full(n - 28, 6.0 + 8 * 20.0)])
    ang = np.deg2rad(deg)
    fast = np.stack([np.cos(ang / 2), np.zeros(n), np.zeros(n), np.sin(ang / 2)], axis=1)
    assert core.rotation_excursions(t, fast) == []


def test_rotation_excursions_known_limit_a_full_turn_reads_as_a_rejoin():
    # DOCUMENTED LIMITATION, not a bug: orientation alone cannot tell "jumped away and
    # snapped back" from "spun a whole revolution" -- both END at the departure
    # orientation. A sustained spin only reaches that inside max_run samples above
    # ~360deg/(30*0.013s) = 920 deg/s, i.e. ~2.6 rev/s of the wrist; the fastest
    # NON-glitch take measured on 0822+0823 peaks at 367 deg/s, so the gap is ~2.5x and
    # this stays theoretical. If a task ever does spin the tool that fast, unwrap the
    # incremental rotation instead of raising the rate gate.
    t = np.arange(40) * 0.013
    deg = np.concatenate([np.zeros(10), np.arange(1, 21) * 18.0, np.full(10, 360.0)])
    ang = np.deg2rad(deg)                                     # 18 deg/sample = 1385 deg/s
    spin = np.stack([np.cos(ang / 2), np.zeros(40), np.zeros(40), np.sin(ang / 2)], axis=1)
    exc = core.rotation_excursions(t, spin)
    assert len(exc) == 1 and exc[0]["amp_deg"] > 170.0, exc    # reads as one big excursion


def test_rotation_excursions_amplitude_floor_and_short_input():
    t, quat = _rot_stream()
    small = quat.copy()
    a = np.deg2rad(6.0)                                     # below MIN_AMP_DEG (10)
    small[20] = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    assert core.rotation_excursions(t, small) == []                # reported: nothing
    # ... but it IS an excursion; only the amplitude floor suppresses it.
    assert len(core.rotation_excursions(t, small, min_amp_deg=1.0)) == 1
    assert core.rotation_excursions(t[:2], quat[:2]) == []         # too short, no crash


def test_rotation_excursions_is_quaternion_order_agnostic():
    # Only |dot| between rows is used, so any fixed component permutation agrees.
    t, quat = _rot_stream()
    a = np.deg2rad(25.0)
    quat[20] = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    wxyz = core.rotation_excursions(t, quat)
    xyzw = core.rotation_excursions(t, quat[:, [1, 2, 3, 0]])
    assert len(wxyz) == len(xyzw) == 1
    assert abs(wxyz[0]["amp_deg"] - xyzw[0]["amp_deg"]) < 1e-9


def test_rotation_excursions_survive_the_position_deglitch():
    # THE reason this scan exists: the excursion has a QUIET position, so the
    # position-only deglitch keeps every row and the bad orientation reaches the label.
    t, quat = _rot_stream()
    pos = np.zeros((len(t), 3)); pos[:, 0] = np.arange(len(t)) * 0.0013   # 0.1 m/s
    a = np.deg2rad(25.0)
    quat[20] = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    keep = core.deglitch_tracker_positions(t, pos)
    assert keep.shape[0] == len(t)                       # deglitch drops NOTHING
    assert len(core.rotation_excursions(t[keep], quat[keep])) == 1


def test_resample_position_clamps_out_of_range():
    src_t = np.array([0.0, 1.0])
    src_pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    # Query beyond both ends + midpoint; must clamp, never raise.
    q = np.array([-5.0, 0.5, 99.0])
    out = core.resample_position(src_t, src_pos, q)
    assert np.allclose(out[0], [0.0, 0.0, 0.0])      # clamped to first
    assert np.allclose(out[1], [0.5, 1.0, 1.5])      # midpoint
    assert np.allclose(out[2], [1.0, 2.0, 3.0])      # clamped to last


def test_slerp_rejects_nonmonotonic_but_sanitizer_fixes_it():
    # B1 regression: scipy Slerp raises on duplicate/backward knots; the converter
    # must sanitize EVERY source stream with make_strictly_increasing first. Here
    # we prove (a) raw non-monotonic times would break Slerp, and (b) feeding the
    # make_strictly_increasing-filtered subset to resample_rotation_wxyz works.
    t = np.array([0.0, 0.1, 0.1, 0.05, 0.2])  # dup + backward
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
    raised = False
    try:
        core.resample_rotation_wxyz(t, quats, np.array([0.15]))
    except ValueError:
        raised = True
    assert raised, "expected Slerp to reject non-monotonic source times"
    keep = core.make_strictly_increasing(t)
    out = core.resample_rotation_wxyz(t[keep], quats[keep], np.array([0.15]))
    assert out.as_quat().shape == (1, 4)


def test_resample_rotation_slerp_midpoint():
    src_t = np.array([0.0, 1.0])
    r0 = Rotation.from_euler("z", 0, degrees=True)
    r1 = Rotation.from_euler("z", 90, degrees=True)
    wxyz = np.stack([r0.as_quat()[[3, 0, 1, 2]], r1.as_quat()[[3, 0, 1, 2]]])
    out = core.resample_rotation_wxyz(src_t, wxyz, np.array([0.5]))
    ang = out.as_euler("zyx", degrees=True)[0, 0]
    assert abs(ang - 45.0) < 1e-6, ang


def test_pose_to_6d_layout():
    pos = np.array([[1.0, 2.0, 3.0]])
    rot = Rotation.from_rotvec([[0.1, 0.2, 0.3]])
    p6 = core.pose_to_6d(pos, rot)
    assert p6.shape == (1, 6)
    assert np.allclose(p6[0, :3], [1.0, 2.0, 3.0])
    assert np.allclose(p6[0, 3:], [0.1, 0.2, 0.3])


def test_tcp_extrinsic_identity_is_noop():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Two rotations about z (use rotvec to avoid scipy's single-axis shape ambiguity).
    rot = Rotation.from_rotvec([[0, 0, np.deg2rad(30)], [0, 0, np.deg2rad(60)]])
    tcp_pos, tcp_rot = core.apply_tcp_extrinsic(pos, rot, np.eye(4))
    assert np.allclose(tcp_pos, pos)
    assert np.allclose(tcp_rot.as_matrix(), rot.as_matrix())


def test_tcp_extrinsic_translation_couples_with_rotation():
    # A pure +x offset of 0.1 m in the tracker frame: when the tracker is rotated
    # 90deg about z, the world TCP position must shift along +y, NOT +x. This is
    # exactly why an identity placeholder silently poisons rotated poses.
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]
    pos = np.array([[0.0, 0.0, 0.0]])
    rot = Rotation.from_euler("z", [90], degrees=True)
    tcp_pos, _ = core.apply_tcp_extrinsic(pos, rot, T)
    assert np.allclose(tcp_pos[0], [0.0, 0.1, 0.0], atol=1e-9), tcp_pos[0]


def test_gripper_width_absolute_meters_and_clamp():
    cal = core.GripperWidthCalib(enc_closed=0.0, enc_open=4.90,
                                 width_closed_m=0.0, width_open_m=0.085)
    w = cal.to_width_m(np.array([-1.0, 0.0, 2.45, 4.90, 10.0]))
    assert abs(w[0] - 0.0) < 1e-9      # below closed -> clamp 0
    assert abs(w[1] - 0.0) < 1e-9
    assert abs(w[2] - 0.0425) < 1e-6   # half travel
    assert abs(w[3] - 0.085) < 1e-9    # open
    assert abs(w[4] - 0.085) < 1e-9    # above open -> clamp


def test_gripper_width_handles_inverted_encoder():
    # DM4310 re-references on power-cycle: open may be < closed in rad.
    cal = core.GripperWidthCalib(enc_closed=-1.35, enc_open=-6.40,
                                 width_closed_m=0.0, width_open_m=0.085)
    assert abs(cal.to_width_m(np.array([-1.35]))[0] - 0.0) < 1e-9
    assert abs(cal.to_width_m(np.array([-6.40]))[0] - 0.085) < 1e-6
    mid = cal.to_width_m(np.array([-3.875]))[0]
    assert abs(mid - 0.0425) < 1e-6, mid


def test_effective_gripper_calib_prefers_recorded_endpoints():
    # The repo JSON ships nominal endpoints (0..4.9); a real take's encoder is
    # all-negative after a power-cycle re-reference, with the LIVE endpoints stored
    # per-sample in gripper.npy cols 3/4. Mapping with the JSON would clamp every
    # width to a rail; effective_gripper_calib must prefer the recorded endpoints.
    base = core.GripperWidthCalib(enc_closed=0.0, enc_open=4.90,
                                  width_closed_m=0.0, width_open_m=0.095)
    n = 50
    enc = np.linspace(-4.373, -1.303, n)              # the real session's range
    gripper = np.zeros((n, 5))
    gripper[:, 0] = np.arange(n) * 0.03               # t
    gripper[:, 1] = enc                               # encoder_rad
    gripper[:, 3] = -1.254                             # recorded open_rad
    gripper[:, 4] = -6.524                             # recorded closed_rad

    # JSON calib zeroes everything (regression we are fixing).
    assert float(base.to_width_m(enc).max()) == 0.0

    cal, src = core.effective_gripper_calib(gripper, base)
    assert src == "recorded"
    assert cal.enc_open == -1.254 and cal.enc_closed == -6.524
    assert cal.width_open_m == 0.095                  # physical width kept from JSON
    w = cal.to_width_m(enc)
    assert w.min() > 0.0 and w.max() <= 0.095 + 1e-9
    assert w.min() < 0.08 < w.max()                   # actually crosses the grasp threshold


def test_effective_gripper_calib_falls_back_when_recorded_absent():
    base = core.GripperWidthCalib(enc_closed=0.0, enc_open=4.90,
                                  width_closed_m=0.0, width_open_m=0.095)
    # Endpoints never reported -> cols 3/4 are 0.0 (degenerate span) -> use JSON.
    gripper = np.zeros((10, 5))
    gripper[:, 1] = np.linspace(0.0, 4.9, 10)
    cal, src = core.effective_gripper_calib(gripper, base)
    assert src == "json" and cal is base
    # No calib at all -> (None, "none").
    cal2, src2 = core.effective_gripper_calib(gripper, None)
    assert cal2 is None and src2 == "none"
    # Old 4-col gripper.npy (no endpoint columns) -> JSON fallback, no crash.
    cal3, src3 = core.effective_gripper_calib(np.zeros((10, 4)), base)
    assert src3 == "json"


def test_antialias_wrench_averages_in_interval():
    # 10 raw samples between two grid points; the grid value should be the mean
    # of the samples that fall in each interval, not a single point sample.
    src_t = np.linspace(0.0, 1.0, 11)
    wrench = np.zeros((11, 6))
    wrench[:, 0] = np.arange(11)  # Fx = 0..10
    grid = np.array([0.0, 1.0])
    out = core.antialias_resample_wrench(src_t, wrench, grid)
    assert out.shape == (2, 6)
    # Edge at midpoint 0.5; searchsorted(side="right") puts t==0.5 in bin1, so
    # bin0 = samples t<0.5 (Fx 0..4), bin1 = t>=0.5 (Fx 5..10).
    assert abs(out[0, 0] - np.mean([0, 1, 2, 3, 4])) < 1e-5, out[0, 0]
    assert abs(out[1, 0] - np.mean([5, 6, 7, 8, 9, 10])) < 1e-5, out[1, 0]


def test_antialias_wrench_no_cross_window_bleed():
    # Regression for the marker-split cross-demo F/T bleed: a take's full F/T spans
    # [0,3]s with a huge contact spike in demo-0 (t<1.0). Episode 2's camera grid
    # is the window [2.0, 3.0]. The episode-2 wrench must NOT inherit demo-0's spike
    # in its first bin — only window-[2,3] samples (all ~0) may contribute.
    src_t = np.linspace(0.0, 3.0, 301)          # 100 Hz over the whole take
    wrench = np.zeros((301, 6))
    wrench[src_t < 1.0, 0] = 100.0               # big Fx spike only in demo-0
    grid = np.linspace(2.0, 3.0, 16)             # episode-2 camera grid
    out = core.antialias_resample_wrench(src_t, wrench, grid)
    assert out.shape == (16, 6)
    # No bin may carry the demo-0 spike; window [2,3] is all zeros.
    assert out[:, 0].max() < 1e-6, out[:, 0].max()


def test_antialias_wrench_empty_interval_falls_back():
    # Sparse source, dense grid -> some grid intervals have no sample; must fall
    # back to interp (clamped), never NaN/raise.
    src_t = np.array([0.0, 1.0])
    wrench = np.array([[0.0] * 6, [6.0] + [0.0] * 5])
    grid = np.linspace(0.0, 1.0, 7)
    out = core.antialias_resample_wrench(src_t, wrench, grid)
    assert out.shape == (7, 6)
    assert np.all(np.isfinite(out))


def test_hires_wrench_shape_dtype_and_causal_order():
    # K most-recent raw samples AT OR BEFORE each grid point; newest at slot K-1.
    # grid starts at 0.0 so the window [0,1] spans all samples; test an INTERIOR
    # frame (0.5) and the last frame (1.0), both of which have >= K samples behind.
    src_t = np.linspace(0.0, 1.0, 101)           # 100 Hz
    wrench = np.zeros((101, 6))
    wrench[:, 0] = np.arange(101)                # Fx = sample index
    grid = np.array([0.0, 0.5, 1.0])             # three steps
    out = core.resample_wrench_hires(src_t, wrench, grid, K=4)
    assert out.shape == (3, 4, 6) and out.dtype == np.float32, (out.shape, out.dtype)
    # At grid=0.5: last sample <= 0.5 is index 50 (t=0.50); 4 newest = 47,48,49,50.
    assert list(out[1, :, 0]) == [47, 48, 49, 50], out[1, :, 0]
    # At grid=1.0: last <= 1.0 is index 100; 4 newest = 97,98,99,100.
    assert list(out[2, :, 0]) == [97, 98, 99, 100], out[2, :, 0]


def test_hires_wrench_overlapping_windows():
    # Consecutive steps' backward windows OVERLAP — this is the whole point (K is
    # decoupled from per-interval sample count). Two close INTERIOR grid points
    # (window starts at 0.0 so both have full backward windows) share samples.
    src_t = np.linspace(0.0, 1.0, 101)
    wrench = np.zeros((101, 6)); wrench[:, 0] = np.arange(101)
    grid = np.array([0.0, 0.50, 0.52])           # steps 1&2 are ~2 samples apart
    out = core.resample_wrench_hires(src_t, wrench, grid, K=4)
    # newest slots: 50 (t=0.50) and 52 (t=0.52); windows {47..50} and {49..52} overlap.
    assert out[1, -1, 0] == 50 and out[2, -1, 0] == 52
    assert set(out[1, :, 0]).intersection(out[2, :, 0]), "windows should overlap"


def test_hires_wrench_left_pads_episode_start_window_confined():
    # A frame with < K in-window samples at/before it is left-padded with the OLDEST
    # IN-WINDOW sample. Window-confinement to [grid_t[0], grid_t[-1]] EXCLUDES samples
    # before the episode's first camera frame (marker-split bleed guard + tare-settling
    # transient exclusion), so the pad value is the oldest sample AT/AFTER grid_t[0],
    # never a pre-window sample.
    src_t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])           # 5 samples over the take
    wrench = np.zeros((5, 6)); wrench[:, 0] = [10.0, 20.0, 30.0, 40.0, 50.0]
    grid = np.array([0.2, 0.4])                            # episode window = [0.2, 0.4]
    out = core.resample_wrench_hires(src_t, wrench, grid, K=5)
    assert out.shape == (2, 5, 6)
    # In-window samples are Fx=30,40,50 (t=0.2,0.3,0.4); the pre-window 10,20 are EXCLUDED.
    # frame grid=0.2: only sample at 0.2 (Fx=30) is <= it -> all K slots = 30 (oldest in-win).
    assert list(out[0, :, 0]) == [30.0, 30.0, 30.0, 30.0, 30.0], out[0, :, 0]
    # frame grid=0.4: 3 in-window samples 30,40,50 -> slots [30,30,30,40,50] (left-pad 30).
    assert list(out[1, :, 0]) == [30.0, 30.0, 30.0, 40.0, 50.0], out[1, :, 0]


def test_hires_wrench_no_cross_window_bleed():
    # Same marker-split regression as the antialias test: a demo-0 spike (t<1.0) must
    # NOT leak into episode-2's window [2,3] via the backward window.
    src_t = np.linspace(0.0, 3.0, 301)
    wrench = np.zeros((301, 6)); wrench[src_t < 1.0, 0] = 100.0
    grid = np.linspace(2.0, 3.0, 16)
    out = core.resample_wrench_hires(src_t, wrench, grid, K=16)
    assert out.shape == (16, 16, 6)
    assert out[:, :, 0].max() < 1e-6, out[:, :, 0].max()   # window [2,3] is all zeros


def test_hires_wrench_k_zero_and_empty_guards():
    src_t = np.linspace(0.0, 1.0, 11); wrench = np.zeros((11, 6))
    grid = np.linspace(0.0, 1.0, 5)
    # K<=0 -> shaped (T,0,6), never crashes (caller omits the key).
    z = core.resample_wrench_hires(src_t, wrench, grid, K=0)
    assert z.shape == (5, 0, 6) and z.dtype == np.float32
    # Empty source -> shaped zeros, no raise.
    e = core.resample_wrench_hires(np.array([]), np.zeros((0, 6)), grid, K=4)
    assert e.shape == (5, 4, 6) and np.all(e == 0)


def test_hires_wrench_all_values_are_real_in_window_samples():
    # Every emitted vector must be an actual in-window raw sample (or a repeat of one),
    # never an interpolated/fabricated value — the "K real samples" contract.
    rng = np.random.default_rng(0)
    src_t = np.sort(rng.uniform(0.0, 5.0, 400))
    wrench = rng.standard_normal((400, 6))
    grid = np.linspace(0.3, 4.7, 40)
    out = core.resample_wrench_hires(src_t, wrench, grid, K=16).astype(np.float64)
    # Compare in float32 (the output dtype) with a tolerance: every emitted vector must
    # match SOME in-window raw sample. (Exact set membership would false-fail on the
    # float64->float32 cast, not on any real fabrication.)
    inwin = wrench[(src_t >= grid[0]) & (src_t <= grid[-1])].astype(np.float32).astype(np.float64)
    bad = 0
    for v in out.reshape(-1, 6):
        if np.abs(inwin - v).sum(axis=1).min() > 1e-4:
            bad += 1
    assert bad == 0, f"{bad} emitted vectors are not real in-window samples"


def test_compute_crop_resize_cover_and_center():
    # 640x480 -> 224x224: cover-fit then center crop the wider axis.
    p = core.compute_crop_resize(640, 480, 224, 224)
    assert p.resize_h == 224
    assert p.resize_w >= 224
    # crop region must be exactly the output size and centered.
    assert p.out_w == 224 and p.out_h == 224
    assert p.crop_x == (p.resize_w - 224) // 2
    assert p.crop_y == (p.resize_h - 224) // 2
    # No distortion: aspect preserved by scale, so resize keeps 4:3 ratio.
    assert abs(p.resize_w / p.resize_h - 640 / 480) < 0.02


def test_split_no_markers_is_single_episode():
    # No markers -> the whole take is one episode (DECISION-E).
    cam_t = np.linspace(0.0, 1.0, 16)
    segs = core.split_take_into_episodes(cam_t, np.zeros((0,)))
    assert segs == [(0, 16)]


def test_split_two_markers_yields_three_episodes():
    # 30 frames at 0..2.9s; markers at 1.0s and 2.0s -> 3 episodes.
    cam_t = np.arange(30) * 0.1
    markers = np.array([1.0, 2.0])
    segs = core.split_take_into_episodes(cam_t, markers)
    assert len(segs) == 3
    # Marker at 1.0 -> first frame >= 1.0 is index 10; at 2.0 -> index 20.
    assert segs == [(0, 10), (10, 20), (20, 30)]
    # Episodes tile the timeline with no gaps/overlap.
    assert segs[0][1] == segs[1][0] and segs[1][1] == segs[2][0]
    assert segs[0][0] == 0 and segs[-1][1] == 30


def test_split_drops_too_short_segments():
    # A marker right before the last frame leaves a 1-frame tail -> dropped.
    cam_t = np.arange(10) * 0.1            # 0.0 .. 0.9
    markers = np.array([0.85])            # cut at index 9 -> tail [9,10) is 1 frame
    segs = core.split_take_into_episodes(cam_t, markers, min_frames=2)
    assert segs == [(0, 9)]              # the 1-frame tail is dropped
    # Unsorted / duplicate markers are tolerated (sorted internally; dup -> empty seg dropped).
    segs2 = core.split_take_into_episodes(cam_t, np.array([0.45, 0.45]), min_frames=2)
    assert all((b - a) >= 2 for a, b in segs2)
    # Frames still fully covered by the union of kept + dropped (monotone bounds).
    assert segs2[0][0] == 0


def test_split_marker_outside_window_is_safe():
    cam_t = np.arange(10) * 0.1
    # Markers before first / after last frame -> no real cut, single episode.
    segs = core.split_take_into_episodes(cam_t, np.array([-5.0, 99.0]), min_frames=2)
    assert segs == [(0, 10)]


def test_validate_episode_flags_jumps_and_coverage():
    cam_t = np.array([0.0, 0.5, 0.6])          # 0.5s gap -> time jump
    tr_t = np.array([0.1, 0.2, 0.3])           # does NOT cover cam window
    tr_p = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0, 0]], dtype=float)  # 0.5m jump
    rep = core.validate_episode_timeline(cam_t, tr_t, tr_p)
    assert not rep["ok"]
    joined = " ".join(rep["warnings"])
    assert "camera time jump" in joined
    assert "position jump" in joined
    assert "does not fully cover" in joined


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed}/{len(tests)} FAILED")
        return 1
    print(f"\nLOGIC OK ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
