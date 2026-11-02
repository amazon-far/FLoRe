#!/usr/bin/env python
"""Hardware-free, dep-light logic tests for the offline marker-review safety net.

These cover the PURE decision logic behind ``recording/review_take.py`` — the
tool that lets an operator see and fix episode-split markers BEFORE the
irreversible UMI Zarr conversion. The viser UI layer is intentionally thin and
untested; all the load-bearing logic (segment summary, plausibility flags,
grasp-cycle detection, and that an edited marker set re-splits to the intended
episode count) lives in ``umi_convert_core`` and is exercised here with numpy
only — runs in the collection .venv.

The cascade under test (the operator's stated fear): a single MISSED marker
silently fuses two demonstrations into one over-long episode (and renumbers all
later ones); a DOUBLED marker can leave a spurious sliver. The review helpers
must make both VISIBLE (auto-flag) and the fix VERIFIABLE (re-split reproduces
the intended count) without any declared episode count.

Run (no pytest needed — runs as a script too):
    .venv/bin/python processing/test_review_logic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import umi_convert_core as core  # noqa: E402


# --- summarize_segments -----------------------------------------------------
def test_summary_matches_split_and_fields():
    # 30 frames at 0..2.9s, markers at 1.0/2.0 -> 3 episodes (mirrors the
    # converter test). The summary must agree frame-for-frame with the splitter.
    cam_t = np.arange(30) * 0.1
    markers = np.array([1.0, 2.0])
    segs = core.split_take_into_episodes(cam_t, markers)
    summ = core.summarize_segments(cam_t, markers)
    assert [(s["start"], s["end"]) for s in summ] == segs
    assert [s["n_frames"] for s in summ] == [10, 10, 10]
    # dur_s is (t_end - t_start) over the kept frames (9 steps of 0.1s).
    assert all(abs(s["dur_s"] - 0.9) < 1e-9 for s in summ)
    assert [s["index"] for s in summ] == [0, 1, 2]


def test_summary_no_markers_single_segment():
    cam_t = np.linspace(0.0, 5.0, 51)
    summ = core.summarize_segments(cam_t, np.zeros((0,)))
    assert len(summ) == 1
    assert summ[0]["start"] == 0 and summ[0]["end"] == 51


# --- flag_segments: missed marker (fused, over-long) ------------------------
def test_missed_marker_flags_outlier_long_then_fixable():
    # Five demos, each ~2 s, at 10 Hz (the realistic case: a median over several
    # normal episodes makes the fused one a clear outlier). Correct markers sit
    # at the four boundaries.
    cam_t = np.arange(0.0, 10.0, 0.1)           # 100 frames, 0..9.9s
    correct = np.array([2.0, 4.0, 6.0, 8.0])    # -> 5 episodes ~2 s each
    flagged_ok = core.flag_segments(core.summarize_segments(cam_t, correct))
    assert len(flagged_ok) == 5
    assert all(not f["flags"] for f in flagged_ok), [f["flags"] for f in flagged_ok]

    # MISS the boundary at 4.0: demos 1+2 fuse into a ~4 s episode among the
    # ~2 s ones. With 4 segments the median (~1.9 s) is meaningful, so the fused
    # blob (~3.9 s > 1.8*median) trips outlier_long under the DEFAULT guard.
    missed = np.array([2.0, 6.0, 8.0])
    flagged_bad = core.flag_segments(core.summarize_segments(cam_t, missed))
    assert len(flagged_bad) == 4
    fused = [f for f in flagged_bad if "outlier_long" in f["flags"]]
    assert len(fused) == 1, [f["flags"] for f in flagged_bad]
    assert abs(fused[0]["dur_s"] - 3.9) < 1e-6        # the two-demo blob
    assert "MISSED marker" in fused[0]["note"]

    # FIX: re-add the missing marker -> back to the intended 5 episodes, no flags.
    flagged_fixed = core.flag_segments(core.summarize_segments(cam_t, correct))
    assert len(flagged_fixed) == 5
    assert all(not f["flags"] for f in flagged_fixed)


# --- flag_segments: doubled marker (sliver) ---------------------------------
def test_doubled_marker_flags_too_short_then_fixable():
    # Two demos ~2 s; a doubled marker near the boundary leaves a short sliver
    # episode that survives the 2-frame drop (the markers are >2 frames apart).
    cam_t = np.arange(0.0, 4.0, 0.1)            # 40 frames, 0..3.9s
    doubled = np.array([2.0, 2.4])              # sliver of ~0.4 s between them
    summ = core.summarize_segments(cam_t, doubled)
    assert len(summ) == 3                       # ep0, sliver, ep2
    flagged = core.flag_segments(summ, short_s=0.5)
    shorts = [f for f in flagged if "too_short" in f["flags"]]
    assert len(shorts) == 1
    assert "doubled marker" in shorts[0]["note"]

    # FIX: delete the spurious second marker -> intended 2 episodes, no flags.
    summ_fixed = core.summarize_segments(cam_t, np.array([2.0]))
    flagged_fixed = core.flag_segments(summ_fixed, short_s=0.5)
    assert len(flagged_fixed) == 2
    assert all(not f["flags"] for f in flagged_fixed)


def test_no_false_flags_on_clean_even_take():
    # Five equal ~2 s episodes -> none too short, none an outlier (all == median).
    cam_t = np.arange(0.0, 10.0, 0.1)
    markers = np.array([2.0, 4.0, 6.0, 8.0])
    flagged = core.flag_segments(core.summarize_segments(cam_t, markers))
    assert len(flagged) == 5
    assert all(not f["flags"] for f in flagged), [f["flags"] for f in flagged]


def test_median_guard_off_for_few_segments():
    # With < min_segments_for_median segments the outlier-long test must NOT fire
    # (median of 1-2 samples is not a meaningful baseline) unless forced.
    cam_t = np.arange(0.0, 6.0, 0.1)
    summ = core.summarize_segments(cam_t, np.array([0.5]))   # 2 segments, very uneven (~0.4s vs ~5.4s)
    default = core.flag_segments(summ)                        # default min=3 -> off
    assert all("outlier_long" not in f["flags"] for f in default)
    forced = core.flag_segments(summ, min_segments_for_median=2)
    assert any("outlier_long" in f["flags"] for f in forced)


# --- grasp-cycle detection --------------------------------------------------
def test_grasp_close_times_counts_one_per_cycle():
    # Width trace: open-close-open-close-open (two grasps). Schmitt trigger must
    # count exactly two falling edges despite the smooth ramp through threshold.
    t = np.arange(0.0, 8.0, 0.05)
    open_m = 0.08
    # two triangular open->close->open cycles
    w = np.empty_like(t)
    for i, ti in enumerate(t):
        phase = ti % 4.0
        frac = 1.0 - abs(phase - 2.0) / 2.0      # 1 at phase0/4, 0 at phase2
        w[i] = open_m * frac
    closes = core.grasp_close_times(t, w, width_open_m=open_m)
    assert len(closes) == 2, (len(closes), closes)


def test_grasp_close_hysteresis_rejects_chatter():
    # Width dithers right around the close threshold: without hysteresis this
    # would fire many times; the Schmitt trigger must count it as ONE grasp.
    t = np.arange(0.0, 2.0, 0.01)
    open_m = 0.08
    th = 0.5 * open_m
    # start open, drop to just-below threshold and jitter there
    w = np.full_like(t, open_m)
    w[t >= 0.5] = th - 0.001 + 0.0005 * np.sin(50 * t[t >= 0.5])
    closes = core.grasp_close_times(t, w, width_open_m=open_m)
    assert len(closes) == 1, (len(closes), closes)


def test_grasps_per_segment_flags_fused_episode():
    # A fused episode contains TWO grasp cycles; flag_segments must mark it
    # multi_grasp even if its duration alone weren't an outlier.
    cam_t = np.arange(0.0, 8.0, 0.1)             # one big take
    # Marker only at 6.0: ep0 = [0,6) holds two grasps, ep1 = [6,8) holds one.
    summ = core.summarize_segments(cam_t, np.array([6.0]))
    # grasp closes at ~2 s, ~5 s (inside ep0) and ~7 s (inside ep1)
    closes = np.array([2.0, 5.0, 7.0])
    counts = core.grasps_per_segment(summ, closes)
    assert counts == [2, 1], counts
    flagged = core.flag_segments(summ, grasp_counts=counts)
    assert "multi_grasp" in flagged[0]["flags"]
    assert "multi_grasp" not in flagged[1]["flags"]


def test_grasp_close_empty_inputs_safe():
    assert core.grasp_close_times(np.zeros((0,)), np.zeros((0,)), 0.08).shape == (0,)
    assert core.grasp_close_times(np.array([0.0, 1.0]), np.array([0.04, 0.04]),
                                  width_open_m=0.0).shape == (0,)
    # A single sample has no edge to observe -> empty (range(1,1) is no iterations).
    assert core.grasp_close_times(np.array([0.5]), np.array([0.02]), 0.08).shape == (0,)


# --- the core promise: an edited marker set re-splits to the intended count --
def test_edited_markers_reproduce_intended_episode_count():
    # This is the whole point of the offline net: the split is a pure function of
    # the marker array, so fixing markers.npy deterministically fixes the episodes.
    cam_t = np.arange(0.0, 10.0, 0.1)
    intended = np.array([2.0, 4.0, 6.0, 8.0])         # 5 episodes
    # operator missed one and doubled another at capture: net same marker count,
    # but the segmentation is WRONG — a ~4 s fused blob plus a ~0.3 s sliver.
    captured = np.array([2.0, 6.0, 6.3, 8.0])         # miss @4.0, double @6.x
    bad = core.flag_segments(core.summarize_segments(cam_t, captured))
    # Same episode COUNT, so a count check alone would miss it — but the flags
    # catch it: a fused outlier AND a too-short sliver are both present.
    assert any("outlier_long" in f["flags"] for f in bad)
    assert any("too_short" in f["flags"] for f in bad)
    # after editing markers.npy back to intended: 5 clean, unflagged episodes.
    fixed = core.flag_segments(core.summarize_segments(cam_t, intended))
    assert len(fixed) == 5
    assert all(not f["flags"] for f in fixed)
    assert all(f["n_frames"] >= 2 for f in fixed)


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
