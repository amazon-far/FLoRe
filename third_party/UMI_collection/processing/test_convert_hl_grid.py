#!/usr/bin/env python
"""Logic tests for the HL-v2 fixed-dt uniform grids in ``umi_convert_core``.

These pin the properties the high-level S1 policy contract depends on, and that a
frame-index window silently violated: exact 1-second spans on a jittery camera grid,
strict causality of the force history, and a force-target bin that describes the SAME
time interval as the action step it is paired with.

Run (no pytest needed — runs as a script too):
    .venv/bin/python processing/test_convert_hl_grid.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import umi_convert_core as core  # noqa: E402


# The real recorded rates, so the tests exercise the actual regime rather than a
# convenient synthetic one: F/T ~178 Hz, camera ~11.9 Hz mean with heavy jitter.
FT_HZ = 178.0
CAM_HZ_MEAN = 11.9


@pytest.fixture
def multibin_grid():
    """Run a test under a MULTI-BIN history grid (8 bins over 1 s), then restore v3.

    The v3 contract has ONE history bin, which makes several history properties
    untestable on the live grid — "oldest-first ordering", "an interior dropout is
    detected", "a window must not bleed across an episode boundary" all need at least
    two bins, and a 33 ms window cannot reach far enough back to bleed anywhere. The
    properties themselves are grid-INDEPENDENT (the binning code takes n_bins/span as
    parameters), so they are exercised on the grid that can actually express them
    rather than weakened to fit the default. ``test_contract_constants_...`` and
    ``test_..._shape_stable...`` pin the live v3 grid separately.
    """
    try:
        yield core.set_grid(force_hist_len=8, action_horizon=16, action_hz=16.0,
                            force_hist_span_s=1.0)
    finally:
        core.set_grid(force_hist_len=1, action_horizon=50, action_hz=30.0)

# A discrete bin mean of a linear signal differs from the signal's value at the
# continuous bin CENTER by at most half the raw sample spacing (the bin's samples are
# not centered in it). So this is the exact analytic tolerance for the ramp checks —
# asserting it (rather than a hand-picked epsilon) also pins that the error is pure
# discretization: it shrinks proportionally with FT_HZ, verified to 17.8 kHz.
RAMP_TOL = 0.5 / FT_HZ


def _jittery_cam_grid(n: int, seed: int = 0) -> np.ndarray:
    """Camera timestamps reproducing the REAL recorded non-uniformity.

    Not uniform jitter: the shipped dataset's grid alternates between 2 and 3
    capture periods of the 30 fps sensor (66.7 / 100 ms) — see the repo's
    "published camera grid must be UNIFORM" invariant, which this data violates.
    Measured on muse_force_20260725: dt min 63.0 / median 97.5 / max 103.9 ms,
    and a 16-frame window spans 1.166..1.568 s (mean 1.345), with 0% of windows
    within 1.0 +/- 0.05 s.
    """
    rng = np.random.default_rng(seed)
    dt = rng.choice([2.0 / 30.0, 3.0 / 30.0], size=n - 1, p=[0.2, 0.8])
    return np.concatenate([[100.0], 100.0 + np.cumsum(dt)])


def _ramp_wrench(t: np.ndarray) -> np.ndarray:
    """A wrench that is an exact linear function of time, per channel.

    Linearity makes every expected bin mean analytic: the mean of a linear signal
    over a bin equals its value at the bin CENTER. That turns "is the anti-aliasing
    right" into an exact equality check instead of a tolerance guess.
    """
    t = np.asarray(t, dtype=np.float64)
    return np.stack([t, 2.0 * t, 3.0 * t, -t, -2.0 * t, -3.0 * t], axis=1)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
def test_contract_constants_are_the_documented_ones():
    """The v3 contract (2026-08-07): NO history, 30 Hz, 50-step chunk.

    Pinned as LITERALS on purpose — this is the one test whose job is to fail when the
    contract moves, so that moving it is a deliberate act with a matching FLoRe-side
    change (flore/s1_policy/data/layout.py + hl_dataset._check_file gate on these exact
    numbers, and a mismatch mis-times every training label)."""
    assert core.FORCE_HIST_LEN == 1                    # t0 only, no look-back
    assert core.ACTION_HORIZON == 50
    assert core.ACTION_HZ == 30.0                      # == CAMERA_PUBLISH_HZ
    assert abs(core.ACTION_DT - 1.0 / 30.0) < 1e-12
    assert abs(core.ACTION_SPAN_S - 50.0 / 30.0) < 1e-12          # 1.6667 s
    # One input bin is exactly one action step wide, so the force the model SEES at t0
    # is the same average the first predicted step is scored against.
    assert abs(core.FORCE_HIST_SPAN_S - core.ACTION_DT) < 1e-12
    assert abs(core.FORCE_HIST_DT - core.ACTION_DT) < 1e-12


def test_the_v2_grid_is_still_reproducible_via_set_grid():
    """The PREVIOUS contract (8 bins over 1 s -> 16 steps over 1 s @ 16 Hz) must stay
    exactly producible, because that is what ``--hl-force-hist-len 8 --hl-action-horizon
    16 --hl-action-hz 16 --hl-force-hist-span-s 1.0`` promises. Note the span must be
    passed: 8 bins is NOT 8*action_dt at 16 Hz, so deriving it would silently give 0.5 s.
    """
    try:
        core.set_grid(force_hist_len=8, action_horizon=16, action_hz=16.0,
                      force_hist_span_s=1.0)
        assert core.FORCE_HIST_LEN == 8
        assert abs(core.FORCE_HIST_SPAN_S - 1.0) < 1e-12
        assert abs(core.FORCE_HIST_DT - 0.125) < 1e-12
        assert core.ACTION_HORIZON == 16
        assert abs(core.ACTION_SPAN_S - 1.0) < 1e-12
        assert abs(core.ACTION_DT - 0.0625) < 1e-12
        # ...and the override REACHES the resamplers (a def-time default would not).
        assert core.future_step_times(np.array([10.0])).shape == (1, 16)
        assert core.past_bin_edges(np.array([10.0])).shape == (1, 9)
    finally:
        core.set_grid(force_hist_len=1, action_horizon=50, action_hz=30.0)


def test_future_times_span_the_action_window_including_the_anchor():
    anchor = np.array([100.0, 250.5])
    times = core.future_step_times(anchor)
    assert times.shape == (2, core.ACTION_HORIZON)
    # T0-INCLUSIVE (DECISION-T0): step 0 lands ON the anchor (UMI parity — the
    # current action is the first predicted step).
    assert np.allclose(times[:, 0] - anchor, 0.0)
    # Last step is one dt SHORT of the full span: [t0, t0+span).
    assert np.allclose(times[:, -1] - anchor, core.ACTION_SPAN_S - core.ACTION_DT)
    # Uniform spacing, independent of where the anchor sits.
    assert np.allclose(np.diff(times, axis=1), core.ACTION_DT)


def test_future_span_is_time_based_not_frame_based():
    """The whole point of the grid: a chunk counted in FRAMES spans a variable
    wall-clock duration on a jittery camera timeline, while the STEP grid is exact.

    Still load-bearing on the v3 grid even though ACTION_HZ now equals the nominal
    camera rate: 'nominal' is not 'actual'. A dropped frame, a reconnect, or a session
    recorded at a different CAMERA_PUBLISH_HZ all break frame-counting, and the offline
    window must stay the same object as the online one."""
    H = core.ACTION_HORIZON
    cam_t = _jittery_cam_grid(400 + H, seed=3)
    frame_spans = cam_t[H:] - cam_t[:-H]
    # Frame-index chunks are both far from the intended span and VARIABLE...
    assert float(frame_spans.max() - frame_spans.min()) > 0.1
    assert np.mean(np.abs(frame_spans - core.ACTION_SPAN_S) < 0.05) == 0.0
    # ...whereas the time grid is exact for every anchor, to float precision.
    times = core.future_step_times(cam_t[:100])
    assert np.allclose(times[:, -1] - cam_t[:100],
                       core.ACTION_SPAN_S - core.ACTION_DT, atol=1e-12)


def test_past_edges_end_at_the_anchor_and_span_the_history_window():
    anchor = np.array([500.0])
    edges = core.past_bin_edges(anchor)
    assert edges.shape == (1, core.FORCE_HIST_LEN + 1)
    assert edges[0, -1] == 500.0                      # last edge IS the anchor
    # The window reaches back exactly FORCE_HIST_SPAN_S (one bin on the v3 grid).
    assert abs(edges[0, 0] - (500.0 - core.FORCE_HIST_SPAN_S)) < 1e-12
    assert np.allclose(np.diff(edges[0]), core.FORCE_HIST_DT)


def test_bad_grid_params_raise():
    for bad in (0, -1):
        try:
            core.future_step_times(np.array([1.0]), horizon=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"horizon={bad} should raise")
        try:
            core.past_bin_edges(np.array([1.0]), n_bins=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_bins={bad} should raise")


# ---------------------------------------------------------------------------
# Force history (input): causality, anti-aliasing, ordering
# ---------------------------------------------------------------------------
def test_history_is_strictly_causal(multibin_grid):
    """No future sample may leak into any history bin — the single most important
    property of an input feature. Verified by making the future enormous."""
    src_t = np.arange(0.0, 20.0, 1.0 / FT_HZ)
    w = np.zeros((src_t.shape[0], 6))
    anchor = np.array([10.0])
    w[src_t > 10.0] = 1e6                              # future is huge
    out = core.resample_wrench_uniform_past(src_t, w, anchor)
    assert out.shape == (1, 8, 6)
    assert np.all(np.abs(out) < 1e-6), out.max()


def test_history_includes_the_sample_exactly_at_the_anchor(multibin_grid):
    """Half-open (lo, hi] bins: a sample AT the anchor is the most recent
    observation and belongs in the last history bin, not dropped and not in a
    future bin."""
    src_t = np.array([9.9, 10.0])
    w = np.array([[0.0] * 6, [5.0] * 6])
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.0]))
    # Last bin covers (9.875, 10.0] -> contains BOTH samples above? 9.9 > 9.875 yes.
    assert np.allclose(out[0, -1], 2.5), out[0, -1]
    # T0-INCLUSIVE: future step 0's bin is (t0-dt, t0] = (9.9375, 10.0], which contains
    # the anchor sample (10.0) but not 9.9 -> its mean is the 5.0 sample alone. The
    # current wrench IS the first action target (UMI parity), not a held/clamped value.
    fut = core.resample_wrench_uniform_future(src_t, w, np.array([10.0]))
    assert np.allclose(fut[0, 0], 5.0), fut[0, 0]


def test_history_causal_through_the_EMPTY_BIN_FALLBACK():
    """REGRESSION. The dense-stream causality test above passes trivially because
    every bin has real samples. The leak lived in the EMPTY-bin fallback: with bins
    (0.125 s) narrower than the sample spacing, an empty bin fell back to a linear
    interpolation over the full stream, which blends the neighbouring sample on EACH
    side — and for the bins near the anchor the right-hand neighbour is in the FUTURE.

    Original observed failure: samples at t=9.0,9.4,9.8,10.2,10.6 with the two
    post-anchor samples set to 1e6 and anchor=10.0 produced 3.4e5 in the last history
    bin. The fallback is now a zero-order hold of the newest sample at or before the
    bin's right edge, so the answer must be the t=9.8 value (0.0)."""
    src_t = np.array([9.0, 9.4, 9.8, 10.2, 10.6])   # 0.4 s apart >> the 0.125 s bin
    w = np.zeros((5, 6))
    w[3:] = 1e6                                      # the FUTURE is enormous
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.0]))
    assert np.abs(out).max() < 1e-6, f"future leaked into history: {out[0, :, 0]}"

    # Non-degenerate version: the hold must return the correct PAST sample, not zero
    # and not the future one.
    w2 = np.array([[1.0] * 6, [2.0] * 6, [3.0] * 6, [999.0] * 6, [999.0] * 6])
    out2 = core.resample_wrench_uniform_past(src_t, w2, np.array([10.0]))
    assert np.allclose(out2[0, -1], 3.0), out2[0, -1]   # newest sample <= 10.0 is t=9.8
    assert out2.max() < 999.0
    # Older bins hold their own most-recent past sample, so the series is monotone
    # non-decreasing and drawn only from {1,2,3}.
    assert set(np.round(out2[0, :, 0], 6)) <= {1.0, 2.0, 3.0}, out2[0, :, 0]
    assert np.all(np.diff(out2[0, :, 0]) >= 0)


def test_history_fallback_holds_rather_than_interpolates_across_a_dropout():
    """A mid-stream sensor dropout must be held at the last real value, not smoothly
    interpolated across (which would invent a measurement AND, near the anchor, pull
    from the future)."""
    src_t = np.concatenate([np.arange(9.0, 9.5, 1.0 / FT_HZ),
                            np.arange(10.5, 11.0, 1.0 / FT_HZ)])   # 1 s hole
    w = np.zeros((src_t.shape[0], 6))
    w[src_t > 10.0] = 50.0                                          # post-gap regime
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.4]))
    # Every history bin sits inside the dropout -> all hold the pre-gap value.
    assert np.allclose(out, 0.0), out[0, :, 0]
    # ...and the window is correctly reported unusable, so it is masked out anyway.
    assert not core.force_history_valid(src_t, np.array([10.4]))[0]


def test_future_targets_still_interpolate_across_empty_bins(multibin_grid):
    """The causal hold applies ONLY to the history input. A target label may use the
    more accurate interpolation, because looking at both neighbours of a FUTURE bin
    leaks nothing."""
    src_t = np.array([10.0, 11.0])            # 1 s apart: the interior bins are empty
    w = np.stack([np.zeros(6), np.full(6, 8.0)])
    out = core.resample_wrench_uniform_future(src_t, w, np.array([10.0]))
    ch0 = out[0, :, 0]
    # T0-INCLUSIVE: step 0's bin (t0-dt, t0] contains the real sample at t0 (value 0.0),
    # so it is that measurement, not an interpolation. Steps 1..15 fall in empty bins and
    # interpolate a rising ramp toward the 11.0 s sample.
    assert ch0[0] == 0.0, ch0[0]
    assert np.all(np.diff(ch0) > 0), ch0            # still strictly rising after step 0
    assert 0.0 < ch0[-1] <= 8.0


def test_history_does_not_bleed_across_an_episode_boundary(multibin_grid):
    """REGRESSION. A take split at markers holds SEPARATE demonstrations. Without
    episode confinement an anchor just after a boundary inherits the previous demo's
    contact (measured: 7 of 8 bins), and the policy learns 'fresh scene + sustained
    contact -> do X' from a recording artifact."""
    src_t = np.arange(100.0, 110.0, 1.0 / FT_HZ)
    w = np.zeros((src_t.shape[0], 6))
    w[src_t < 103.6, 0] = 99.0          # the PREVIOUS episode was pressing hard
    boundary = 103.7                     # ep1's first camera frame
    anchor = np.array([103.8])           # 0.1 s into ep1

    leaked = core.resample_wrench_uniform_past(src_t, w, anchor)
    assert leaked[0, :, 0].max() > 50.0, "expected the un-confined case to bleed"

    clean = core.resample_wrench_uniform_past(src_t, w, anchor, episode_start_t=boundary)
    assert np.allclose(clean, 0.0), clean[0, :, 0]
    # And the confined window is correctly reported unusable (its bins are holds),
    # so these anchors are masked rather than silently trusted.
    assert not core.force_history_valid(src_t, anchor, episode_start_t=boundary)[0]
    # Deeper into ep1, once a full second of THIS episode exists, it is usable again.
    assert core.force_history_valid(src_t, np.array([104.8]), episode_start_t=boundary)[0]


def test_history_ordering_is_chronological_oldest_first(multibin_grid):
    src_t = np.arange(0.0, 20.0, 1.0 / FT_HZ)
    w = _ramp_wrench(src_t)
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.0]))
    ch0 = out[0, :, 0]
    assert np.all(np.diff(ch0) > 0), ch0               # monotone increasing in time
    # Slot -1 is the newest (closest to the anchor), slot 0 the oldest.
    assert ch0[-1] > ch0[0]
    assert abs(ch0[-1] - (10.0 - 0.5 * core.FORCE_HIST_DT)) <= RAMP_TOL


def test_history_bin_mean_matches_the_analytic_bin_center():
    """Anti-aliasing check: the mean of a linear signal over a bin == its value at
    the bin center. Exact to the raw sample spacing."""
    src_t = np.arange(0.0, 20.0, 1.0 / FT_HZ)
    w = _ramp_wrench(src_t)
    anchor = np.array([12.3456])
    out = core.resample_wrench_uniform_past(src_t, w, anchor)
    edges = core.past_bin_edges(anchor)
    centers = 0.5 * (edges[0, :-1] + edges[0, 1:])
    for j, c in enumerate(centers):
        assert abs(out[0, j, 0] - c) <= RAMP_TOL, (j, out[0, j, 0], c)
        # channel 2 is 3x the ramp, so its discretization error scales 3x too.
        assert abs(out[0, j, 2] - 3.0 * c) <= 3.0 * RAMP_TOL, (j, out[0, j, 2])


def test_history_antialiases_rather_than_point_samples():
    """A high-frequency contact oscillation must AVERAGE toward its mean, not be
    point-sampled (which would alias it into a spurious low-frequency signal)."""
    src_t = np.arange(0.0, 20.0, 1.0 / FT_HZ)
    # 40 Hz oscillation about a DC offset of 3.0: well above the 8 Hz bin rate.
    w = np.zeros((src_t.shape[0], 6))
    w[:, 0] = 3.0 + 10.0 * np.sin(2 * np.pi * 40.0 * src_t)
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.0]))
    assert np.all(np.abs(out[0, :, 0] - 3.0) < 1.0), out[0, :, 0]


def test_history_near_episode_start_holds_a_real_value_never_zero():
    """A zero wrench means 'no contact' downstream, so an uncovered bin must hold a
    real measurement (and be flagged invalid separately), never be fabricated as 0."""
    src_t = np.arange(10.0, 12.0, 1.0 / FT_HZ)   # stream STARTS at t=10
    w = np.full((src_t.shape[0], 6), 7.0)
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.1]))
    assert np.allclose(out, 7.0), out.min()      # every bin holds the real 7.0
    assert not np.any(out == 0.0)


def test_history_empty_stream_is_zero_and_flagged_invalid(multibin_grid):
    src_t = np.zeros(0)
    w = np.zeros((0, 6))
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.0, 11.0]))
    assert out.shape == (2, 8, 6) and np.all(out == 0.0)
    valid = core.force_history_valid(src_t, np.array([10.0, 11.0]))
    assert valid.shape == (2,) and not valid.any()


def test_history_single_sample_stream_holds_it():
    src_t = np.array([10.0])
    w = np.full((1, 6), 2.5)
    out = core.resample_wrench_uniform_past(src_t, w, np.array([10.5]))
    assert np.allclose(out, 2.5)


# ---------------------------------------------------------------------------
# Force targets (output): phase alignment with the action step
# ---------------------------------------------------------------------------
def test_future_force_bin_k_ends_at_action_step_k():
    """THE alignment contract: force step k and action step k describe the same time
    bin. The pre-v2 hires docstring flagged this as an off-by-one landmine; pin it."""
    src_t = np.arange(0.0, 20.0, 1.0 / FT_HZ)
    w = _ramp_wrench(src_t)
    anchor = np.array([10.0])
    out = core.resample_wrench_uniform_future(src_t, w, anchor)
    times = core.future_step_times(anchor)
    for k in range(core.ACTION_HORIZON):
        # bin k covers (t_k - dt, t_k], so its mean (linear signal) is t_k - dt/2.
        expect = times[0, k] - 0.5 * core.ACTION_DT
        assert abs(out[0, k, 0] - expect) <= RAMP_TOL, (k, out[0, k, 0], expect)


def test_future_and_past_both_end_at_the_anchor(multibin_grid):
    """T0-INCLUSIVE (DECISION-T0): history bin -1 covers (t-dt_h, t] and future step 0's
    bin covers (t-dt_a, t] — BOTH end exactly at the anchor. The step-0 force target is
    therefore the current contact (the most recent dt_a of wrench), and it re-reads the
    tail of the history window rather than describing a disjoint future interval. That
    overlap is intended: history is the causal INPUT, the step-0 force is the first
    TARGET, and 'the action now' is anchored to 'the force now'."""
    anchor = np.array([10.0])
    past = core.past_bin_edges(anchor)
    fut_t = core.future_step_times(anchor)
    assert past[0, -1] == anchor[0]                  # last history edge is the anchor
    assert abs(fut_t[0, 0] - anchor[0]) < 1e-12      # future step 0 lands on the anchor
    # Future step 0's bin lower edge sits INSIDE the last history bin (they overlap).
    fut_first_lo = fut_t[0, 0] - core.ACTION_DT
    assert past[0, -2] < fut_first_lo < past[0, -1]


def test_future_steps_past_the_stream_hold_the_last_value():
    src_t = np.arange(0.0, 10.2, 1.0 / FT_HZ)       # stream ENDS at ~10.2
    w = np.zeros((src_t.shape[0], 6))
    w[:, 0] = np.arange(src_t.shape[0])              # last value is large
    out = core.resample_wrench_uniform_future(src_t, w, np.array([10.0]))
    last_real = w[-1, 0]
    # The tail steps (beyond 10.2 s) all hold the final sample...
    assert np.allclose(out[0, -4:, 0], last_real, atol=1.0), out[0, -4:, 0]
    # ...and must be flagged as padding rather than trained on.
    valid = core.future_step_valid(src_t, np.array([10.0]), episode_end_t=10.5)
    assert valid[0, 0] and not valid[0, -1]


# ---------------------------------------------------------------------------
# Pose / gripper targets on the future grid
# ---------------------------------------------------------------------------
def _synth_tracker(t: np.ndarray, *, omega: float = 0.7) -> np.ndarray:
    """A raw-tracker array (TRACKER_COLS layout) on a known analytic trajectory:
    position linear in t, rotation a constant-rate spin about +z."""
    from scipy.spatial.transform import Rotation

    t = np.asarray(t, dtype=np.float64)
    n = t.shape[0]
    pos = np.stack([0.1 * t, 0.2 * t, 0.3 * t], axis=1)
    # from_rotvec (not from_euler) so a length-n angle array is unambiguous.
    rotvec = np.zeros((n, 3))
    rotvec[:, 2] = omega * t
    q_xyzw = Rotation.from_rotvec(rotvec).as_quat()                 # (n, 4) xyzw
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    out = np.zeros((n, 10))
    out[:, 0] = t
    out[:, 1] = np.arange(n)
    out[:, 2:5] = pos
    out[:, 5:9] = q_wxyz
    return out


def test_pose_targets_land_on_the_future_grid_times(multibin_grid):
    """A linear trajectory sampled on the 16 Hz grid must reproduce the analytic
    position at each step time — i.e. the resample really queries those times."""
    tr_t = np.arange(0.0, 30.0, 1.0 / 77.0)        # the real ~77 Hz tracker rate
    tracker = _synth_tracker(tr_t)
    anchor = np.array([10.0, 12.5])
    out = core.resample_pose_uniform_future(tracker, anchor, None)
    assert out.shape == (2, 16, 6)
    times = core.future_step_times(anchor)
    for i in range(2):
        for k in range(core.ACTION_HORIZON):
            tk = times[i, k]
            assert np.allclose(out[i, k, :3], [0.1 * tk, 0.2 * tk, 0.3 * tk], atol=1e-6)


def test_pose_targets_come_from_the_raw_tracker_not_the_camera_grid():
    """Deriving 16 Hz targets from the ~11.9 Hz camera grid would UPSAMPLE from the
    slower stream and lose real motion. Feed a trajectory with structure between
    camera frames and show the raw-stream resample tracks it while a camera-grid
    resample does not."""
    tr_t = np.arange(0.0, 30.0, 1.0 / 77.0)
    # A 4 Hz wiggle: resolvable at 77 Hz, ALIASED at 11.9 Hz.
    tracker = _synth_tracker(tr_t)
    tracker[:, 2] += 0.05 * np.sin(2 * np.pi * 4.0 * tr_t)
    anchor = np.array([10.0])
    from_raw = core.resample_pose_uniform_future(tracker, anchor, None)

    cam_t = _jittery_cam_grid(400, seed=1) - 100.0        # ~11.9 Hz, starts at 0
    cam_pos = core.resample_position(tr_t, tracker[:, 2:5], cam_t)
    cam_rot = core.resample_rotation_wxyz(tr_t, tracker[:, 5:9], cam_t)
    cam_tracker = np.zeros((cam_t.shape[0], 10))
    cam_tracker[:, 0] = cam_t
    cam_tracker[:, 2:5] = cam_pos
    cam_tracker[:, 5:9] = cam_rot.as_quat()[:, [3, 0, 1, 2]]
    from_cam = core.resample_pose_uniform_future(cam_tracker, anchor, None)

    # The two disagree materially on the wiggling axis: information was lost.
    assert np.abs(from_raw[0, :, 0] - from_cam[0, :, 0]).max() > 0.01


def test_pose_targets_apply_the_tcp_extrinsic_like_the_camera_grid_path():
    """Same extrinsic composition as the anchor path, so a target and its anchor are
    numerically consistent. Rotation must couple into translation."""
    tr_t = np.arange(0.0, 30.0, 1.0 / 77.0)
    tracker = _synth_tracker(tr_t)
    T = np.eye(4)
    T[0, 3] = 0.21                                  # the rig's puck->TCP offset
    anchor = np.array([10.0])
    raw = core.resample_pose_uniform_future(tracker, anchor, None)
    tcp = core.resample_pose_uniform_future(tracker, anchor, T)
    # Offsets differ per step because the hand is SPINNING (so +x rotates).
    delta = tcp[0, :, :3] - raw[0, :, :3]
    assert np.allclose(np.linalg.norm(delta, axis=1), 0.21, atol=1e-6)
    assert np.abs(delta[0] - delta[-1]).max() > 1e-3, "rotation must couple in"
    # Identity extrinsic is a strict no-op.
    same = core.resample_pose_uniform_future(tracker, anchor, np.eye(4))
    assert np.allclose(same, raw, atol=1e-9)


def test_pose_targets_use_slerp_for_rotation():
    """A constant-rate spin must come back as a constant-rate spin (Slerp), not the
    component-wise quaternion lerp that would bend the rate."""
    from scipy.spatial.transform import Rotation

    tr_t = np.arange(0.0, 30.0, 1.0 / 77.0)
    omega = 0.7
    tracker = _synth_tracker(tr_t, omega=omega)
    anchor = np.array([10.0])
    out = core.resample_pose_uniform_future(tracker, anchor, None)
    times = core.future_step_times(anchor)[0]
    for k, tk in enumerate(times):
        ang = Rotation.from_rotvec(out[0, k, 3:]).as_euler("xyz")[2]
        expect = np.arctan2(np.sin(omega * tk), np.cos(omega * tk))   # wrapped
        assert abs(np.arctan2(np.sin(ang - expect), np.cos(ang - expect))) < 1e-6, k


def test_pose_target_validity_gates_on_tracker_coverage(multibin_grid):
    """REGRESSION. resample_pose_uniform_future CLAMPS past the tracker span, so a
    mid-episode dropout yields 16 identical frozen positions. Those steps must be
    masked, or the policy is trained (at full loss weight) to predict 'stop moving'."""
    tr_t = np.arange(100.0, 102.0, 1.0 / 77.0)     # tracker DIES at 102.0
    tracker = _synth_tracker(tr_t - 100.0)
    tracker[:, 0] = tr_t
    anchor = np.array([103.5])                      # 1.5 s after the tracker died
    out = core.resample_pose_uniform_future(tracker, anchor, None)
    # Every step is the same frozen endpoint...
    assert np.allclose(out[0, :, :3], out[0, 0, :3], atol=1e-9)
    # ...and the tracker-coverage term marks them all invalid.
    valid = core.future_step_valid(tracker[:, 0], anchor, episode_end_t=105.16)
    assert not valid.any(), valid
    # An anchor inside the tracker's coverage stays valid.
    assert core.future_step_valid(tracker[:, 0], np.array([100.5]),
                                  episode_end_t=105.16)[0].all()


def test_implausible_target_steps_are_flagged():
    """REGRESSION. deglitch BRIDGES an isolated teleport, which repairs the ANCHOR label
    but leaves a physically impossible ramp in the interpolated TARGET trajectory —
    covered by neither the take-level gate nor the per-anchor robot0_pose_glitch mask.
    Measured before this mask: 84 of 10,067 supervised steps implied >3.2 m/s."""
    T, H = 4, core.ACTION_HORIZON
    pose = np.zeros((T, H, 6))
    # A calm trajectory: 6 mm/step, the measured real median.
    pose[:, :, 0] = np.arange(H) * 0.006
    assert not core.implausible_step_mask(pose, core.ACTION_DT).any()

    # Inject a 40 cm teleport between steps 7 and 8 of anchor 2 (6.4 m/s).
    pose[2, 8:, 0] += 0.40
    mask = core.implausible_step_mask(pose, core.ACTION_DT)
    # BOTH endpoints of the offending interval are flagged (either may be corrupt)...
    assert mask[2, 7] and mask[2, 8]
    # ...and nothing else is.
    assert mask.sum() == 2, np.argwhere(mask)
    assert not mask[[0, 1, 3]].any()

    # The threshold is in m/s, so it scales with dt, and is tunable.
    assert core.implausible_step_mask(pose, core.ACTION_DT, max_speed=100.0).sum() == 0
    # Pinned against the MEASURED camera-grid step distribution, not free-hand speed.
    # The old 4.0 was reasoned from how fast a hand can move and was consequently DEAD
    # CODE: it flagged 0 of 634,928 clean94 steps while 5 episodes carried 118-201 mm
    # one-frame ramps (1.76-3.01 m/s). The 89 known-clean episodes never exceed 0.506
    # m/s, so 1.0 catches every residual ramp with zero false positives. Any change here
    # must be re-measured against a real session — see core.MAX_TARGET_SPEED_MPS.
    assert core.MAX_TARGET_SPEED_MPS == 1.0
    # ...and the gate must actually BITE at the real-data scale. A 120 mm one-frame jump
    # on the camera grid (the ep30/ep31 signature) implies 1.8 m/s: flagged now, missed
    # by the old 4.0 gate.
    ramp = np.zeros((1, H, 6))
    ramp[0, 4:, 0] = 0.120
    assert core.implausible_step_mask(ramp, core.ACTION_DT).sum() == 2
    assert core.implausible_step_mask(ramp, core.ACTION_DT, max_speed=4.0).sum() == 0


def test_implausible_step_mask_handles_degenerate_horizons():
    assert core.implausible_step_mask(np.zeros((3, 1, 6)), core.ACTION_DT).shape == (3, 1)
    assert not core.implausible_step_mask(np.zeros((3, 1, 6)), core.ACTION_DT).any()


def test_pose_targets_clamp_past_the_tracker_span():
    tr_t = np.arange(0.0, 10.2, 1.0 / 77.0)         # tracker ENDS at 10.2
    tracker = _synth_tracker(tr_t)
    out = core.resample_pose_uniform_future(tracker, np.array([10.0]), None)
    # Tail steps hold the endpoint value rather than extrapolating.
    assert np.allclose(out[0, -1, :3], out[0, -2, :3], atol=1e-9)
    assert np.isfinite(out).all()


def test_gripper_targets_track_the_encoder_stream(multibin_grid):
    src_t = np.arange(0.0, 30.0, 1.0 / 29.0)        # the real ~29 Hz gripper rate
    width = 0.02 + 0.001 * src_t                     # metres, linear ramp
    anchor = np.array([10.0, 20.0])
    out = core.resample_gripper_uniform_future(src_t, width, anchor)
    assert out.shape == (2, 16) and out.dtype == np.float32
    times = core.future_step_times(anchor)
    expect = 0.02 + 0.001 * times
    assert np.allclose(out, expect, atol=1e-6)
    # Absolute METRES (UMI's 0.08 m grasp threshold depends on the scale).
    assert 0.02 < out.min() < 0.06


def test_gripper_targets_clamp_and_stay_finite():
    src_t = np.arange(0.0, 10.1, 1.0 / 29.0)
    width = np.linspace(0.0, 0.095, src_t.shape[0])
    out = core.resample_gripper_uniform_future(src_t, width, np.array([10.0]))
    assert np.isfinite(out).all()
    assert np.allclose(out[0, -1], 0.095, atol=1e-6)   # endpoint-held


# ---------------------------------------------------------------------------
# Validity masks
# ---------------------------------------------------------------------------
def test_force_history_valid_detects_an_INTERIOR_dropout():
    """REGRESSION. A mid-window sensor dropout passes the span-coverage test trivially
    (samples exist on both sides) while every bin inside the hole is a stale hold.
    Originally reported valid; must now be invalid."""
    src_t = np.concatenate([np.arange(9.0, 9.5, 1.0 / FT_HZ),
                            np.arange(10.5, 11.0, 1.0 / FT_HZ)])   # 1 s hole
    assert not core.force_history_valid(src_t, np.array([10.4]))[0]
    # A window entirely inside a continuous stretch is still fine.
    assert core.force_history_valid(src_t, np.array([9.5]))[0] is not None
    dense = np.arange(0.0, 30.0, 1.0 / FT_HZ)
    assert core.force_history_valid(dense, np.array([20.0]))[0]


def test_force_history_valid_is_bin_exact_by_default_and_tunable(multibin_grid):
    """Default ``max_unmeasured_bins=0``: EVERY one of the 8 bins must contain a real
    sample. A hole wide enough to empty a bin invalidates the window; a hole narrower
    than the raw sample spacing cannot empty one."""
    dense = np.arange(0.0, 30.0, 1.0 / FT_HZ)
    # A 0.3 s hole empties >=2 bins (bins are 0.125 s).
    holed = dense[(dense < 19.5) | (dense > 19.8)]
    assert not core.force_history_valid(holed, np.array([20.0]))[0]
    # Explicit opt-in tolerance accepts it; never the default.
    assert core.force_history_valid(holed, np.array([20.0]), max_unmeasured_bins=4)[0]
    # A 0.1 s hole is shorter than a bin, so at most one bin can be emptied and with
    # this alignment none is -> still valid under the strict default.
    tiny = dense[(dense < 19.53) | (dense > 19.63)]
    assert core.force_history_valid(tiny, np.array([20.0]))[0]


def test_force_history_valid_catches_samples_bunched_at_one_end():
    """A window whose only samples sit far from the anchor leaves the recent bins —
    the ones that matter most — unmeasured. The edge-reach term catches it."""
    src_t = np.arange(9.0, 9.35, 1.0 / FT_HZ)      # nothing in the last 0.65 s
    assert not core.force_history_valid(src_t, np.array([10.0]))[0]


def test_force_history_valid_tracks_coverage_of_the_window(multibin_grid):
    src_t = np.arange(10.0, 30.0, 1.0 / FT_HZ)
    # Fully covered anchor deep inside the stream.
    assert core.force_history_valid(src_t, np.array([20.0]))[0]
    # Anchor 0.1 s after the stream starts: only 10% of a 1 s window is covered.
    assert not core.force_history_valid(src_t, np.array([10.1]))[0]
    # An anchor entirely before the stream is invalid, not silently ok.
    assert not core.force_history_valid(src_t, np.array([5.0]))[0]
    # An anchor at exactly 1 s in is the first fully-measured one.
    assert core.force_history_valid(src_t, np.array([11.01]))[0]


def test_episode_start_anchor_is_invalid_even_at_high_span_coverage(multibin_grid):
    """REGRESSION. An anchor 0.8 s into the stream passes the 0.75 span-coverage test
    (80% of its window overlaps the recording) while its FIRST bins lie entirely before
    the stream and are backfilled from the earliest sample. The bin-exact check is what
    catches it; span coverage alone called this valid."""
    src_t = np.arange(100.0, 130.0, 1.0 / FT_HZ)
    anchor = np.array([100.8])
    # Span coverage alone would pass...
    overlap = min(anchor[0], src_t[-1]) - max(anchor[0] - 1.0, src_t[0])
    assert overlap / 1.0 >= 0.75
    # ...but bins 0-1 precede the stream, so the window is not measured.
    assert not core.force_history_valid(src_t, anchor)[0]
    # Prove the leading bins really are holds: a ramping signal shows bin 0 carrying
    # the value of the stream's FIRST sample, not of its own time interval.
    w = np.zeros((src_t.shape[0], 6))
    w[:, 0] = np.arange(src_t.shape[0], dtype=float)
    out = core.resample_wrench_uniform_past(src_t, w, anchor)
    assert out[0, 0, 0] == 0.0, out[0, 0, 0]      # held from sample 0 (t=100.0)
    assert out[0, -1, 0] > 100.0                 # the newest bin is real


def test_future_step_valid_stops_at_the_episode_boundary(multibin_grid):
    """A chunk must never predict across an episode boundary into the next demo."""
    src_t = np.arange(0.0, 100.0, 1.0 / FT_HZ)
    anchor = np.array([10.0])
    valid = core.future_step_valid(src_t, anchor, episode_end_t=10.5)
    assert valid.shape == (1, 16)
    # Steps at or before 10.5 s are valid; the rest are padding.
    times = core.future_step_times(anchor)[0]
    assert np.array_equal(valid[0], times <= 10.5 + 1e-9)
    # T0-INCLUSIVE: steps land on t0 + {0, dt, ..}, so t0..t0+0.5 s is 9 steps (0.5/dt + 1),
    # one more than the pre-v2 grid that started at t0+dt.
    assert valid[0].sum() == 9
    # A full-length episode leaves every step valid.
    assert core.future_step_valid(src_t, anchor, episode_end_t=99.0)[0].all()


def test_future_step_valid_also_gated_by_the_ft_stream_end():
    src_t = np.arange(0.0, 10.30, 1.0 / FT_HZ)     # F/T ends before the episode does
    valid = core.future_step_valid(src_t, np.array([10.0]), episode_end_t=11.0)
    assert valid[0, :4].all()                       # up to 0.25 s: covered
    assert not valid[0, -1]                         # 1.0 s out: past the F/T stream


# ---------------------------------------------------------------------------
# Tare-transient masking
# ---------------------------------------------------------------------------
def test_tare_transient_spans_point_BACKWARD_from_the_tare():
    """REGRESSION on the span DIRECTION. The new offset takes effect AT tare_wall, so
    the contaminated samples are the ones BEFORE it, still carrying the previous
    offset. Measured on session_2026.07.25: mean |F| in the 300 ms before the tare is
    0.474 N vs 0.245 N after, pre > post in 82/97 takes.

    A forward span (the original bug) masked the CLEAN side and left 143 pre-tare
    force-target bins flagged valid with up to 1.45 N of offset error."""
    spans = core.tare_transient_spans(np.array([100.0, 131.9]))
    assert spans.shape == (2, 2)
    assert np.allclose(spans[:, 1], [100.0, 131.9])          # span ENDS at the tare
    assert np.allclose(spans[:, 0], [99.5, 131.4])           # ...and starts before it
    assert np.allclose(spans[:, 1] - spans[:, 0], core.TARE_SETTLE_S)
    # Absent / empty / non-finite provenance degrades to "no known transient".
    assert core.tare_transient_spans(None).shape == (0, 2)
    assert core.tare_transient_spans(np.zeros(0)).shape == (0, 2)
    assert core.tare_transient_spans(np.array([np.nan])).shape == (0, 2)


def test_window_overlap_is_intersection_not_containment():
    spans = np.array([[10.0, 10.5]])
    # A window that merely CLIPS the transient must still count as hit: a window is
    # only as trustworthy as its worst sample.
    assert core.window_overlaps_spans(np.array([9.6]), np.array([10.1]), spans)[0]
    assert core.window_overlaps_spans(np.array([10.4]), np.array([11.4]), spans)[0]
    assert core.window_overlaps_spans(np.array([9.0]), np.array([12.0]), spans)[0]
    # Strictly outside on either side is clean.
    assert not core.window_overlaps_spans(np.array([8.0]), np.array([9.99]), spans)[0]
    assert not core.window_overlaps_spans(np.array([10.51]), np.array([11.5]), spans)[0]


def test_history_invalidated_when_its_window_touches_a_tare(multibin_grid):
    """A tare at 10.0 contaminates [9.5, 10.0], so any anchor whose 1 s history window
    reaches into that band is unusable."""
    src_t = np.arange(0.0, 40.0, 1.0 / FT_HZ)
    tare = core.tare_transient_spans(np.array([10.0]))       # -> [9.5, 10.0]
    # A window is [anchor-1, anchor]; it overlaps [9.5, 10.0] while anchor-1 <= 10.0,
    # i.e. for every anchor up to and including 11.0.
    anchors = np.array([9.8, 10.2, 10.6, 11.0, 11.01, 12.0])
    valid = core.force_history_valid(src_t, anchors, tare_spans=tare)
    assert not valid[:4].any(), valid
    # Once the window's lower edge clears the tare, it is usable again.
    assert valid[4:].all(), valid
    # Without the provenance the same anchors all look fine -> the silent case.
    assert core.force_history_valid(src_t, anchors).all()


def test_future_steps_inside_a_tare_transient_are_invalid():
    """A tare at 10.0 contaminates [9.5, 10.0], so an anchor just BEFORE it has its
    early future steps masked (they fall in the dirty band) and its later steps kept."""
    src_t = np.arange(0.0, 40.0, 1.0 / FT_HZ)
    tare = core.tare_transient_spans(np.array([10.0]))       # -> [9.5, 10.0]
    anchor = np.array([9.6])
    valid = core.future_step_valid(src_t, anchor, episode_end_t=39.0, tare_spans=tare)
    times = core.future_step_times(anchor)[0]
    dt = core.ACTION_DT
    # A step whose own (t-dt, t] bin overlaps [9.5, 10.0] is masked...
    dirty = (times - dt <= 10.0) & (times >= 9.5)
    assert not valid[0][dirty].any(), times[dirty]
    # ...and steps entirely past the tare survive.
    assert valid[0][(times - dt) > 10.0].all()


# ---------------------------------------------------------------------------
# Vectorization equivalence (the prefix-sum path vs a naive per-bin loop)
# ---------------------------------------------------------------------------
def test_vectorized_bin_means_match_a_naive_loop():
    """The O(1)-per-bin prefix-sum implementation must agree exactly with the
    obvious slow version, including on empty bins."""
    rng = np.random.default_rng(11)
    src_t = np.sort(rng.uniform(0.0, 20.0, 3000))
    w = rng.normal(size=(src_t.shape[0], 6))
    anchors = _jittery_cam_grid(40, seed=5) - 100.0 + 5.0     # inside the span
    out = core.resample_wrench_uniform_past(src_t, w, anchors)
    edges = core.past_bin_edges(anchors)
    for i in range(anchors.shape[0]):
        for j in range(core.FORCE_HIST_LEN):
            lo, hi = edges[i, j], edges[i, j + 1]
            sel = (src_t > lo) & (src_t <= hi)
            if sel.any():
                assert np.allclose(out[i, j], w[sel].mean(axis=0), atol=1e-5), (i, j)


def test_grids_are_shape_stable_over_a_whole_episode():
    """A realistic per-episode call: 220 anchors, both grids, no ragged shapes."""
    cam_t = _jittery_cam_grid(221, seed=7)
    src_t = np.arange(cam_t[0] - 2.0, cam_t[-1] + 2.0, 1.0 / FT_HZ)
    w = _ramp_wrench(src_t)
    past = core.resample_wrench_uniform_past(src_t, w, cam_t)
    fut = core.resample_wrench_uniform_future(src_t, w, cam_t)
    # Shapes come from the LIVE grid (v3: 1 history bin, 50 steps) — the point is that
    # every anchor gets the same shape, not what the numbers happen to be.
    assert past.shape == (221, core.FORCE_HIST_LEN, 6) and past.dtype == np.float32
    assert fut.shape == (221, core.ACTION_HORIZON, 6) and fut.dtype == np.float32
    assert np.isfinite(past).all() and np.isfinite(fut).all()
    # Coverage: with 2 s of pre/post padding every anchor's history is valid.
    assert core.force_history_valid(src_t, cam_t).all()


def _run_all() -> int:
    """Standalone runner (this file also runs under pytest).

    It must INJECT the ``multibin_grid`` fixture, because 15 of the 44 tests take it as
    a parameter: calling those with no argument raises TypeError, which the loop below
    counted as a FAILURE. So this runner reported "15/44 FAILED" while the pytest run of
    the same file was green — i.e. those 15 history/target tests were never actually
    executed here, they only ever recorded their own call-convention error. The fixture
    is a generator (set the multi-bin grid, restore v3 in `finally`), so drive it as a
    context manager and hand the yielded grid in."""
    import contextlib
    import inspect

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    # The @pytest.fixture decorator wraps the generator; reach the raw function so this
    # runner does not need pytest's fixture machinery.
    _raw = getattr(multibin_grid, "__wrapped__", None) or \
        getattr(getattr(multibin_grid, "__pytest_wrapped__", None), "obj", None)
    if _raw is None:  # pragma: no cover - decorator internals changed
        print("  WARN: cannot unwrap the multibin_grid fixture; its tests will be skipped")
    failed = 0
    for t in tests:
        try:
            if _raw is not None and "multibin_grid" in inspect.signature(t).parameters:
                with contextlib.contextmanager(_raw)() as grid:
                    t(grid)
            else:
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
