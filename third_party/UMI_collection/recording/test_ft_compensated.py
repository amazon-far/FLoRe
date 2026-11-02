#!/usr/bin/env python3
"""Logic tests for the record-both F/T provenance (raw always + compensated
sidecar). Pure recorder buffer/flush behavior — no hardware, no WS.

Run:  PYTHONPATH=recording .venv/bin/python recording/test_ft_compensated.py
Prints "LOGIC OK (N checks)" on success; raises on the first failure.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recorder as R  # noqa: E402
import recording_format as fmt  # noqa: E402

_N = 0


def check(cond, msg):
    global _N
    _N += 1
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok  {msg}")


def _drive(rec, *, compensated, with_comp_samples, flip_to=None, tare=None, n=12,
           algo=None, sign_verified=None, flip_algo_to=None):
    """Feed a short take. flip_to: send a second batch with this compensated flag.
    algo/sign_verified: attach ft_comp_algo/ft_sign_verified to each batch.
    flip_algo_to: send a second batch with this ft_comp_algo (mid-take switch)."""
    rec.start(task="t")
    t0 = rec.buf.start_wall
    for k in range(n):
        t = t0 + 0.01 * k
        msg = {"compensated": compensated, "samples": [[t, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]}
        if with_comp_samples:
            msg["samples_comp"] = [[t, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01]]
        if algo is not None:
            msg["ft_comp_algo"] = algo
        if sign_verified is not None:
            msg["ft_sign_verified"] = sign_verified
        if tare is not None and k == 0:
            msg["tare_wrench"] = tare
            msg["tare_wall"] = t0
        rec.on_ft_batch(msg)
        if flip_to is not None and k == 0:
            rec.on_ft_batch({"compensated": flip_to, "samples": [[t + 0.005, 1, 1, 1, 0, 0, 0]]})
        if flip_algo_to is not None and k == 0:
            m2 = {"compensated": compensated, "samples": [[t + 0.005, 1, 1, 1, 0, 0, 0]],
                  "ft_comp_algo": flip_algo_to}
            if with_comp_samples:
                m2["samples_comp"] = [[t + 0.005, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0]]
            rec.on_ft_batch(m2)
        rec.on_tracker({"t": t, "seq": k, "pos": [0.1, 0.2, 0.3], "quat": [1, 0, 0, 0], "stale": False})
        rec.on_camera({"t": t, "seq": k, "w": 4, "h": 4,
                       "jpeg_b64": base64.b64encode(R._synth_jpeg(t - t0)).decode()})
    return rec.stop()


def _meta(td):
    return json.load(open(os.path.join(td, fmt.TAKE_META_FILE)))


def main() -> int:
    with tempfile.TemporaryDirectory() as root:
        # A) compensated take -> sidecar written, flag True, raw still present.
        td = _drive(R.Recorder(root, session_name="comp"), compensated=True,
                    with_comp_samples=True, tare=[0.5, 1.5, 2.5, 0.09, 0.19, 0.29])
        m = _meta(td)
        check(m["ft_compensated"] is True, "compensated take -> ft_compensated True")
        check(os.path.exists(os.path.join(td, fmt.FT_COMP_FILE)), "compensated take writes ft_wrench_comp.npy")
        check(os.path.exists(os.path.join(td, fmt.FT_FILE)), "compensated take still keeps raw ft_wrench.npy")
        check(m["ft_tare_wrench"][:3] == [0.5, 1.5, 2.5], "tare provenance recorded")
        raw = np.load(os.path.join(td, fmt.FT_FILE))
        comp = np.load(os.path.join(td, fmt.FT_COMP_FILE))
        check(abs(raw[0, 1] - 1.0) < 1e-9, "raw file holds RAW values (Fx=1)")
        check(abs(comp[0, 1] - 0.5) < 1e-9, "sidecar holds COMPENSATED values (Fx=0.5)")
        check(raw.shape == comp.shape, "raw + comp have matching shape")

        # B) raw-only take -> no sidecar, flag False (legacy/mock default path too).
        td = _drive(R.Recorder(root, session_name="raw"), compensated=False, with_comp_samples=False)
        m = _meta(td)
        check(m["ft_compensated"] is False, "raw take -> ft_compensated False")
        check(not os.path.exists(os.path.join(td, fmt.FT_COMP_FILE)), "raw take writes NO sidecar")

        # C) legacy frame with NO 'compensated' key -> defaults to raw (no KeyError).
        rec = R.Recorder(root, session_name="legacy")
        rec.start(task="t")
        t0 = rec.buf.start_wall
        for k in range(12):
            t = t0 + 0.01 * k
            rec.on_ft_batch({"samples": [[t, 1, 2, 3, 0, 0, 0]]})  # no 'compensated'
            rec.on_tracker({"t": t, "seq": k, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "stale": False})
            rec.on_camera({"t": t, "seq": k, "w": 4, "h": 4,
                           "jpeg_b64": base64.b64encode(R._synth_jpeg(t - t0)).decode()})
        m = _meta(rec.stop())
        check(m["ft_compensated"] is False, "legacy frame (no key) -> ft_compensated False")

        # D) mid-take flip -> 'mixed' (provenance conflict is loud, not silent).
        td = _drive(R.Recorder(root, session_name="mixed"), compensated=True,
                    with_comp_samples=True, flip_to=False)
        m = _meta(td)
        check(m["ft_compensated"] == "mixed", "mid-take flip -> ft_compensated 'mixed'")

        # E) no-FT take is never labeled compensated, even if a stray comp flag arrived.
        rec = R.Recorder(root, session_name="noft")
        rec.start(task="t")
        t0 = rec.buf.start_wall
        for k in range(12):  # tracker + camera only, no ft batches at all
            t = t0 + 0.01 * k
            rec.on_tracker({"t": t, "seq": k, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "stale": False})
            rec.on_camera({"t": t, "seq": k, "w": 4, "h": 4,
                           "jpeg_b64": base64.b64encode(R._synth_jpeg(t - t0)).decode()})
        m = _meta(rec.stop())
        check(m["ft_valid"] is False, "no-FT take -> ft_valid False")
        check(m["ft_compensated"] is False, "no-FT take -> ft_compensated False")
        check(m.get("ft_comp_algo") is None, "no-FT take -> ft_comp_algo null")

        # F) compensated take on RNEA (verified sign) -> ft_comp_algo 'rnea', sign True.
        td = _drive(R.Recorder(root, session_name="rnea"), compensated=True,
                    with_comp_samples=True, algo="rnea", sign_verified=True)
        m = _meta(td)
        check(m["ft_comp_algo"] == "rnea", "rnea take -> ft_comp_algo 'rnea'")
        check(m["ft_sign_verified"] is True, "rnea take (verified) -> ft_sign_verified True")

        # G) compensated take on RNEA with UNVERIFIED sign -> ft_sign_verified False.
        td = _drive(R.Recorder(root, session_name="rnea_unv"), compensated=True,
                    with_comp_samples=True, algo="rnea", sign_verified=False)
        m = _meta(td)
        check(m["ft_comp_algo"] == "rnea", "unverified rnea take -> ft_comp_algo 'rnea'")
        check(m["ft_sign_verified"] is False, "unverified rnea take -> ft_sign_verified False")

        # H) bota take -> ft_comp_algo 'bota'.
        td = _drive(R.Recorder(root, session_name="bota"), compensated=True,
                    with_comp_samples=True, algo="bota", sign_verified=True)
        m = _meta(td)
        check(m["ft_comp_algo"] == "bota", "bota take -> ft_comp_algo 'bota'")

        # I) mid-take ALGO switch (bota->rnea) -> ft_comp_algo 'mixed' (loud).
        td = _drive(R.Recorder(root, session_name="algo_mixed"), compensated=True,
                    with_comp_samples=True, algo="bota", sign_verified=True, flip_algo_to="rnea")
        m = _meta(td)
        check(m["ft_comp_algo"] == "mixed", "mid-take algo switch -> ft_comp_algo 'mixed'")

        # J) raw-only take -> ft_comp_algo null even if a stray algo key rode a raw batch.
        td = _drive(R.Recorder(root, session_name="raw_algo"), compensated=False,
                    with_comp_samples=False, algo="rnea")
        m = _meta(td)
        check(m.get("ft_comp_algo") is None, "raw-only take -> ft_comp_algo null (no sidecar)")

        # K) tare-window gap fix: at episode start the auto-tare drains frames and the
        # reader's frame_sink pushes RAW-ONLY compensated batches (compensated=True but
        # NO samples_comp) so raw stays gap-free; comp legitimately resumes post-tare.
        # This must NOT trip 'mixed' (the compensated FLAG is consistent) and RAW must
        # hold ALL samples while comp holds only the post-tare ones.
        rec = R.Recorder(root, session_name="tarewindow")
        rec.start(task="t")
        t0 = rec.buf.start_wall
        for k in range(4):  # tare-window: raw-only, compensated flag still True
            t = t0 + 0.01 * k
            rec.on_ft_batch({"compensated": True, "ft_comp_algo": "rnea", "ft_sign_verified": True,
                             "samples": [[t, 9.0, 0, 0, 0, 0, 0]]})  # NO samples_comp
            rec.on_tracker({"t": t, "seq": k, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "stale": False})
            rec.on_camera({"t": t, "seq": k, "w": 4, "h": 4,
                           "jpeg_b64": base64.b64encode(R._synth_jpeg(t - t0)).decode()})
        for k in range(4, 12):  # settled: raw + comp
            t = t0 + 0.01 * k
            rec.on_ft_batch({"compensated": True, "ft_comp_algo": "rnea", "ft_sign_verified": True,
                             "samples": [[t, 9.0, 0, 0, 0, 0, 0]], "samples_comp": [[t, 0.1, 0, 0, 0, 0, 0]]})
            rec.on_tracker({"t": t, "seq": k, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "stale": False})
            rec.on_camera({"t": t, "seq": k, "w": 4, "h": 4,
                           "jpeg_b64": base64.b64encode(R._synth_jpeg(t - t0)).decode()})
        td = rec.stop()
        m = _meta(td)
        raw = np.load(os.path.join(td, fmt.FT_FILE))
        comp = np.load(os.path.join(td, fmt.FT_COMP_FILE))
        check(m["ft_compensated"] is True, "tare-window raw-only batches -> NOT 'mixed' (flag consistent)")
        check(raw.shape[0] == 12, "tare-window: RAW is gap-free (all 12 samples)")
        check(comp.shape[0] == 8, "tare-window: comp holds only the 8 post-tare samples")

    print(f"\nLOGIC OK ({_N} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
