#!/usr/bin/env python
"""Logic tests for the per-take QC sheet (processing/plot_take_qc.py).

Guards the parts that must NOT silently drift from the converter/audit:
  * ``sanitize_tracker`` reproduces ``umi_convert_core`` severity exactly,
  * ``rotation_label_glitch`` uses the audit's label-grid path/net discriminator
    (so genuine fast rotation is NOT flagged, a there-and-back glitch IS),
  * ``verdict_for`` mirrors convert_to_umi's KEEP/DROP gate,
  * the small signal helpers (median filter, quat canonicalization, geodesic).

Dep-light (numpy + scipy only — matplotlib is imported lazily inside plot_take,
so this whole file runs in the collection .venv). Run:
    .venv/bin/python processing/test_plot_take_qc.py
The real-session cross-check auto-skips when the recordings aren't present.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recording"))

import plot_take_qc as q  # noqa: E402
import umi_convert_core as core  # noqa: E402
import recording_format as fmt  # noqa: E402

_SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "data", "recordings", "session_2026.08.03_21.23.19")
# clean94 shipped exclusions (the 2026-08-03 QC session): every one must auto-flag.
_EXCLUDED = ["011", "013", "025", "037", "048", "058", "072", "074", "075", "076", "083", "088"]


# ---- tiny helpers ---------------------------------------------------------


def test_canon_matches_converter():
    assert q._canon("5") == "take_005"
    assert q._canon("076") == "take_076"
    assert q._canon("take_037") == "take_037"


def test_medfilt_kills_lone_spike_and_keeps_shape():
    x = np.zeros((20, 3))
    x[10] = 5.0
    m = q._medfilt(x, 7)
    assert m.shape == (20, 3)
    assert abs(m[10].max()) < 1e-9, "rolling median must remove a single-sample spike"
    # 1-D passthrough
    assert q._medfilt(np.arange(5.0), 3).shape == (5,)


def test_quat_canonicalization_resolves_double_cover():
    q_wxyz = np.array([[1, 0, 0, 0.0]] * 3 + [[-1, 0, 0, 0.0]] * 3)  # sign flip halfway
    qc = q._canonical_quat_xyzw(q_wxyz)
    dots = np.sum(qc[:-1] * qc[1:], axis=1)
    assert (dots >= -1e-9).all(), "consecutive quats must share a hemisphere after canon"
    assert np.allclose(np.linalg.norm(qc, axis=1), 1.0)


def test_geodesic_deg_endpoints():
    a = np.array([[0, 0, 0, 1.0]])
    b = np.array([[1, 0, 0, 0.0]])  # 180deg about x
    assert abs(q._geodesic_deg(a, a)[0]) < 1e-6
    assert abs(q._geodesic_deg(a, b)[0] - 180.0) < 1e-3


# ---- verdict gate ---------------------------------------------------------


def test_verdict_mirrors_converter_drop_gate():
    clean = {"n_unbridged": 0, "residual_max_jump_m": 0.02}
    noglitch = {"is_glitch": False, "max_step_deg": 1.0, "ratio": 1.0}
    glitch = {"is_glitch": True, "max_step_deg": 154.0, "ratio": 11.8}
    assert q.verdict_for(clean, noglitch)[0] == "KEEP"
    assert q.verdict_for({"n_unbridged": 1, "residual_max_jump_m": 0.05}, noglitch)[0] == "KEEP*"
    assert q.verdict_for(clean, glitch)[0] == "CHECK"
    assert q.verdict_for({"n_unbridged": q._DROP_N, "residual_max_jump_m": 0.05}, noglitch)[0] == "DROP"
    assert q.verdict_for({"n_unbridged": 0, "residual_max_jump_m": q._DROP_J + 0.01}, noglitch)[0] == "DROP"


def test_secondary_verdict_says_the_drop_gate_does_NOT_cover_it():
    """A DROP-level glitch on a 2pt/3pt SECONDARY keypoint must not claim the converter's
    drop gate handled it. That gate is primary-only: for right/head the converter writes
    no pose_glitch mask and drops nothing, so the take ships with a corrupt robot1_*/head_*
    label. Same VERDICT letter (the take is still unusable for that keypoint), different
    reason — and the PRIMARY text stays byte-identical so the recorded clean94 verdicts
    still reproduce."""
    noglitch = {"is_glitch": False, "max_step_deg": 1.0, "ratio": 1.0}
    bad = {"n_unbridged": 0, "residual_max_jump_m": q._DROP_J + 0.01}
    v_left, why_left = q.verdict_for(bad, noglitch, "left")
    assert (v_left, why_left) == q.verdict_for(bad, noglitch), \
        "role defaults to 'left', so every existing caller is unchanged"
    for role in ("right", "head"):
        v, why = q.verdict_for(bad, noglitch, role)
        assert v == "DROP", f"{role}: a DROP-level excursion is still a DROP"
        assert why != why_left, f"{role}: must not reuse the primary's wording"
        assert "PRIMARY-only" in why and "--exclude" in why, \
            f"{role}: must say the gate does not cover it and name the remedy, got {why!r}"
    assert ">= drop gate" in why_left, "the primary's original wording is preserved verbatim"


def test_verdict_needs_path_net_not_raw_step():
    """A big raw step with a LOW path/net (genuine fast rotation) is NOT a glitch —
    this is the fix for takes 030/031/034, which reach ~14-23 deg/frame but ratio<2."""
    clean = {"n_unbridged": 0, "residual_max_jump_m": 0.02}
    fast_genuine = {"is_glitch": False, "max_step_deg": 23.0, "ratio": 1.8}
    assert q.verdict_for(clean, fast_genuine)[0] == "KEEP"


# ---- real-session fidelity (auto-skip if recordings absent) --------------


def _have_session():
    return os.path.isdir(_SESSION) and any(
        d.startswith("take_") for d in os.listdir(_SESSION))


def test_sanitize_tracker_equals_converter_severity_on_take_037():
    if not _have_session():
        print("  (skip: recordings not present)")
        return
    tr = np.load(os.path.join(_SESSION, "take_037", fmt.TRACKER_FILE))
    s = q.sanitize_tracker(tr)
    # Recompute severity the converter's way (stale-drop -> incr -> severity).
    t = tr[:, 0].astype(float); pos = tr[:, 2:5].astype(float); st = tr[:, 9]
    live = st < 0.5; t, pos = t[live], pos[live]
    keep = core.make_strictly_increasing(t); t, pos = t[keep], pos[keep]
    ref = core.tracker_glitch_severity(t, pos)
    assert s["sev"]["n_unbridged"] == ref["n_unbridged"] == 3
    assert abs(s["sev"]["residual_max_jump_m"] - ref["residual_max_jump_m"]) < 1e-12
    assert s["sev"]["residual_max_jump_m"] >= q._DROP_J, "037 must trip the residual drop gate"
    assert s["drop_mask"].sum() > 0, "037 must have deglitch drops (the red dots)"


def _verdict_for_take(td):
    tr = q._load(td, fmt.TRACKER_FILE)
    fidx = q._load(td, os.path.join(fmt.CAMERA_DIR, fmt.CAMERA_INDEX_FILE))
    n = 0 if fidx is None else len(fidx)
    if n and n < q.MIN_FRAMES:
        return "JUNK"
    if tr is None:
        return "KEEP"
    s = q.sanitize_tracker(tr)
    cam_t = fidx[:, 0].astype(float) if fidx is not None else None
    return q.verdict_for(s["sev"], q.rotation_label_glitch(s, cam_t))[0]


#: Takes the raw-stream rotation scan (``core.rotation_excursions``, added 2026-08-23)
#: flags on clean94 that its 12-exclusion list does NOT contain. They are NOT false
#: positives waved away: their raw signature is INDISTINGUISHABLE from the exclusions
#: that shipped — out-jump 711/1195/928 deg/s against a 3.4 deg/s median, peak amplitude
#: 18.1/14.9/11.9 deg, rejoining to within 3.0/11.7/5.1 deg — while clean94's OWN
#: exclusions sit at 486-1108 deg/s and 10.2-15.9 deg. No threshold on either axis
#: separates the two groups. What separates them is WHEN they were judged: clean94 is the
#: retired **v2** dataset on a 16 Hz camera grid, and the only detector at the time was the
#: label-grid path/net test, which a 13-51 ms excursion slips through *because* that grid
#: attenuates it (the same mechanism that hid take_366/371 in the 0823 session at 30 Hz).
#: So this list is read as "clean94's QC was incomplete", not as over-flagging — and it is
#: pinned rather than tolerated so any drift in the scan shows up as a test failure.
#: Nothing downstream changes: clean94 is retired (HLDataset refuses it under contract v3)
#: and CHECK is a badge asking for a human look, never an automatic drop.
_ROT_SCAN_EXTRA = ["030", "031", "034"]


def test_auto_verdict_matches_clean94_exclusions():
    """The QC auto-verdict must flag every one of the 12 shipped exclusions (no false
    negative), and flag NOTHING beyond those plus ``_ROT_SCAN_EXTRA``. This pins the
    whole pipeline to the real dataset."""
    if not _have_session():
        print("  (skip: recordings not present)")
        return
    excluded = {q._canon(x) for x in _EXCLUDED}
    allowed = excluded | {q._canon(x) for x in _ROT_SCAN_EXTRA}
    flagged, missed, false_pos = set(), [], []
    for d in sorted(os.listdir(_SESSION)):
        if not d.startswith("take_"):
            continue
        v = _verdict_for_take(os.path.join(_SESSION, d))
        bad = v in ("DROP", "CHECK", "JUNK")
        if bad:
            flagged.add(d)
        if d in excluded and not bad:
            missed.append((d, v))
        if d not in allowed and bad:
            false_pos.append((d, v))
    assert not missed, f"excluded takes the QC missed (read clean): {missed}"
    assert not false_pos, f"kept takes the QC wrongly flagged: {false_pos}"
    assert excluded <= flagged, f"not all exclusions flagged: {excluded - flagged}"
    print(f"  auto-verdict flagged all {len(excluded)} shipped exclusions "
          f"+ {len(flagged - excluded)} rotation-scan finds")


def test_rotation_scan_extras_are_flagged_for_the_documented_reason():
    """``_ROT_SCAN_EXTRA`` must be flagged by the RAW-stream scan specifically — if the
    label-grid test starts catching them instead, the comment above is stale."""
    if not _have_session():
        print("  (skip: recordings not present)")
        return
    for t in _ROT_SCAN_EXTRA:
        td = os.path.join(_SESSION, f"take_{t}")
        if not os.path.isdir(td):
            continue
        s = q.sanitize_tracker(q._load(td, fmt.TRACKER_FILE))
        keep = ~s["drop_mask"]
        exc = core.rotation_excursions(s["t"][keep], s["quat_wxyz"][keep])
        assert exc, f"take_{t}: raw scan found nothing; _ROT_SCAN_EXTRA is stale"
        assert _verdict_for_take(td) == "CHECK", f"take_{t} must read CHECK"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    n = 0
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
        n += 1
    print(f"\nALL {n} QC LOGIC TESTS PASS")
