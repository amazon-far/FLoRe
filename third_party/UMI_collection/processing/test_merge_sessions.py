#!/usr/bin/env python
"""Guard tests for ``merge_sessions.py``.

Merging sittings is the one pre-processing step that can corrupt a dataset **invisibly**:
every field it collapses (tracker-to-role map, world alignment, quaternion convention)
becomes a single zarr attr for the whole file, so two mismatched sittings do not fail —
they train. And take numbers are the only handle a QC exclusion has, so a collision
silently drops the wrong recording.

These pin the refusals, the numbering, and the two properties that make the merge safe to
re-run: raw bytes are never touched, and links stay relative so the tree can move.

Run (no pytest needed — runs as a script too):
    .venv/bin/python processing/test_merge_sessions.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "merge_sessions.py")

BASE_META = {
    "created_wall": 1786574316.0,
    "created_str": "2026.08.12_15.38.36",
    "clock": "time.time() wall-clock seconds (single rig clock)",
    "quaternion_convention": "wxyz",
    "record_model": "continuous_take_with_episode_split_markers",
    "mode": "1pt",
    "primary_role": "left",
    "serial_role_map": {"LHR-75106789": "left"},
    "world_align": {"rpy_deg": [0.0, 0.0, 180.0], "source": "world_rpy_deg"},
    "schema_version": 4,
}


def _sitting(root, name, n_takes, *, meta_over=None, uncommitted=()):
    """A minimal session dir: session_meta.json + take_NNN/take_meta.json + one payload byte.

    The payload exists so the "raw bytes untouched" assertion has something to check.
    ``uncommitted`` names take dirs written WITHOUT take_meta.json, i.e. an interrupted
    recording, which the converter ignores and so must the merge.
    """
    d = os.path.join(root, name)
    os.makedirs(d)
    meta = dict(BASE_META)
    meta["created_str"] = name.replace("session_", "")
    meta.update(meta_over or {})
    with open(os.path.join(d, "session_meta.json"), "w") as f:
        json.dump(meta, f)
    for i in range(n_takes):
        t = os.path.join(d, f"take_{i:03d}")
        os.makedirs(t)
        with open(os.path.join(t, "tracker.npy"), "wb") as f:
            f.write(b"raw")
        with open(os.path.join(t, "take_meta.json"), "w") as f:
            json.dump({"take_dir": f"take_{i:03d}"}, f)
    for name_ in uncommitted:
        t = os.path.join(d, name_)
        os.makedirs(t)
        with open(os.path.join(t, "tracker.npy"), "wb") as f:
            f.write(b"raw")
    return d


def _run(*args):
    return subprocess.run([sys.executable, _SCRIPT, *args], capture_output=True, text=True)


def _merged_meta(out):
    with open(os.path.join(out, "session_meta.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------
def test_offsets_default_to_the_next_free_hundred(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 69)
    b = _sitting(root, "session_B", 51)
    out = os.path.join(root, "session_merged")
    r = _run(a, b, "-o", out)
    assert r.returncode == 0, r.stderr

    m = _merged_meta(out)["merged_take_map"]
    assert m["take_000"] == "session_A/take_000"
    assert m["take_068"] == "session_A/take_068"
    assert m["take_100"] == "session_B/take_000"     # next hundred ABOVE 68
    assert m["take_150"] == "session_B/take_050"
    assert len(m) == 120


def test_a_sitting_over_100_takes_pushes_the_next_block_to_400(tmp_path):
    """The rule is "next free multiple of 100", not "100 * index" — a 168-take sitting
    starting at 200 ends at 367, so the block after it must start at 400, not 300."""
    root = str(tmp_path)
    a = _sitting(root, "session_A", 69)
    b = _sitting(root, "session_B", 51)
    c = _sitting(root, "session_C", 168)
    d = _sitting(root, "session_D", 3)
    out = os.path.join(root, "session_merged")
    assert _run(a, b, c, d, "-o", out).returncode == 0

    m = _merged_meta(out)["merged_take_map"]
    assert m["take_200"] == "session_C/take_000"
    assert m["take_367"] == "session_C/take_167"
    assert m["take_400"] == "session_D/take_000"
    assert _merged_meta(out)["merged_take_offsets"] == {
        "session_A": 0, "session_B": 100, "session_C": 200, "session_D": 400}


def test_explicit_offsets_that_collide_are_refused(tmp_path):
    """A collision would make ``--exclude take_050`` drop a take from the wrong sitting."""
    root = str(tmp_path)
    a = _sitting(root, "session_A", 69)
    b = _sitting(root, "session_B", 51)
    r = _run(a, b, "-o", os.path.join(root, "m"), "--offsets", "0", "50")
    assert r.returncode != 0
    assert "collision" in r.stderr
    assert not os.path.exists(os.path.join(root, "m", "take_050"))


def test_merging_preserves_the_earlier_numbering_so_exclusions_transfer(tmp_path):
    """Re-merging with a THIRD sitting appended must not renumber the first two, or every
    QC exclusion recorded against the old dataset silently points at a different take."""
    root = str(tmp_path)
    a = _sitting(root, "session_A", 69)
    b = _sitting(root, "session_B", 51)
    two = os.path.join(root, "m2")
    assert _run(a, b, "-o", two).returncode == 0
    c = _sitting(root, "session_C", 168)
    three = os.path.join(root, "m3")
    assert _run(a, b, c, "-o", three).returncode == 0

    old, new = _merged_meta(two)["merged_take_map"], _merged_meta(three)["merged_take_map"]
    assert all(old[t] == new[t] for t in old)
    assert len(new) == len(old) + 168


# ---------------------------------------------------------------------------
# The contract check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("world_align", {"rpy_deg": [0.0, 0.0, 0.0], "source": "world_rpy_deg"}),
    ("serial_role_map", {"LHR-4AE9DA27": "left"}),
    ("mode", "3pt"),
    ("primary_role", "right"),
    ("quaternion_convention", "xyzw"),
    ("schema_version", 3),
    ("record_model", "single_take"),
])
def test_a_contract_mismatch_is_refused(tmp_path, field, value):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 4)
    b = _sitting(root, "session_B", 4, meta_over={field: value})
    out = os.path.join(root, "m")
    r = _run(a, b, "-o", out)
    assert r.returncode != 0, f"{field} mismatch was accepted"
    assert field in r.stderr
    assert not os.path.exists(out), "a refused merge must not leave a partial dir"


def test_allow_field_overrides_one_named_field_only(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 4)
    b = _sitting(root, "session_B", 4, meta_over={"schema_version": 3, "mode": "3pt"})
    out = os.path.join(root, "m")
    # Allowing only schema_version leaves the mode mismatch fatal.
    r = _run(a, b, "-o", out, "--allow-field", "schema_version")
    assert r.returncode != 0 and "mode" in r.stderr
    # Allowing both proceeds, and still REPORTS what was overridden.
    r = _run(a, b, "-o", out, "--allow-field", "schema_version", "mode")
    assert r.returncode == 0, r.stderr
    assert "ALLOWED" in r.stderr
    assert _merged_meta(out)["schema_version"] == 4  # the FIRST sitting's, as documented


def test_a_merged_dir_cannot_be_used_as_a_source(tmp_path):
    """Chaining would make merged_from name a symlink farm instead of the real bytes."""
    root = str(tmp_path)
    a = _sitting(root, "session_A", 4)
    b = _sitting(root, "session_B", 4)
    first = os.path.join(root, "m1")
    assert _run(a, b, "-o", first).returncode == 0
    c = _sitting(root, "session_C", 4)
    r = _run(first, c, "-o", os.path.join(root, "m2"))
    assert r.returncode != 0
    assert "itself a merged session" in r.stderr
    assert "session_A" in r.stderr and "session_B" in r.stderr  # tells you what to pass


# ---------------------------------------------------------------------------
# Properties that make it safe to re-run
# ---------------------------------------------------------------------------
def test_links_are_relative_and_the_raw_bytes_are_untouched(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 3)
    b = _sitting(root, "session_B", 2)
    out = os.path.join(root, "m")
    assert _run(a, b, "-o", out).returncode == 0

    link = os.path.join(out, "take_000")
    assert os.path.islink(link)
    target = os.readlink(link)
    assert not os.path.isabs(target), "an absolute link breaks as soon as the tree moves"
    assert os.path.isdir(link)
    # The payload is reachable through the link and byte-identical.
    with open(os.path.join(link, "tracker.npy"), "rb") as f:
        assert f.read() == b"raw"
    # Nothing but links, session_meta.json and calibration/ exists in the merged dir.
    entries = sorted(os.listdir(out))
    assert set(entries) - {"session_meta.json", "calibration"} == {
        f"take_{i:03d}" for i in (0, 1, 2, 100, 101)}


def test_created_wall_is_the_first_sittings_so_episode_order_is_chronological(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 2, meta_over={"created_wall": 1000.0})
    b = _sitting(root, "session_B", 2, meta_over={"created_wall": 2000.0})
    out = os.path.join(root, "m")
    assert _run(a, b, "-o", out).returncode == 0
    assert _merged_meta(out)["created_wall"] == 1000.0
    assert _merged_meta(out)["created_str"] == "m"


def test_uncommitted_takes_are_ignored_exactly_as_the_converter_ignores_them(tmp_path):
    """An interrupted recording has no take_meta.json. Linking it would make the merge see
    a take the converter refuses to ingest, so the counts would disagree."""
    root = str(tmp_path)
    a = _sitting(root, "session_A", 3, uncommitted=("take_003", "take_004.tmp"))
    b = _sitting(root, "session_B", 2)
    out = os.path.join(root, "m")
    assert _run(a, b, "-o", out).returncode == 0
    m = _merged_meta(out)["merged_take_map"]
    assert len(m) == 5
    assert "take_003" not in m and "take_004" not in m


def test_force_rebuilds_links_but_refuses_to_eat_a_real_recording(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 2)
    b = _sitting(root, "session_B", 2)
    out = os.path.join(root, "m")
    assert _run(a, b, "-o", out).returncode == 0
    # Without --force: no clobber.
    assert _run(a, b, "-o", out).returncode != 0
    # With --force: fine, links are disposable.
    assert _run(a, b, "-o", out, "--force").returncode == 0

    # But a dir holding REAL take dirs is never overwritten, even with --force -- that is
    # the typo that would delete a raw session.
    raw = _sitting(root, "session_raw", 2)
    r = _run(a, b, "-o", raw, "--force")
    assert r.returncode != 0 and "real files" in r.stderr
    assert os.path.exists(os.path.join(raw, "take_000", "tracker.npy"))


def test_at_least_two_sources_are_required(tmp_path):
    root = str(tmp_path)
    a = _sitting(root, "session_A", 2)
    assert _run(a, "-o", os.path.join(root, "m")).returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
