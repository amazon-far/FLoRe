#!/usr/bin/env python3
"""Offline marker review + edit tool — the safety net for continuous-take capture.

Runs BETWEEN ``recorder.py`` (Stop) and ``processing/convert_to_umi.py``, in the
single collection ``.venv`` (DECISION-I). It is a standalone OFFLINE process — NOT
a bridge, NOT part of the recorder, never opens the gripper/recorder WS — so it
touches none of the capture-time invariants (single-CAN-writer, the o+p gesture,
the atomic-flush take-commit path).

Why it exists
-------------
In the continuous-take model (DECISION-E) one Start..Stop is a single take and the
operator drops an episode-split MARKER (``o+p`` long-press, or the GUI button) at
each demo boundary; the OFFLINE converter splits the take into N+1 UMI episodes at
the N markers. The failure that motivated this tool: a single MISSED marker
silently FUSES two demonstrations into one over-long episode (and renumbers every
later one); a DOUBLED marker can leave a spurious sliver. Today nothing surfaces
that before the irreversible ``.zarr.zip`` write.

The fix leverages a structural fact: the split is a PURE, re-runnable function of
``markers.npy`` (a plain float64 ``[M,1]`` of absolute wall-clock seconds) and the
camera timeline. So a wrong marker is 100% recoverable by editing that one file
before conversion. This tool makes that practical:

  * It reproduces the EXACT split the converter will produce by calling the SAME
    ``umi_convert_core.split_take_into_episodes`` on the SAME
    ``make_strictly_increasing`` camera grid — the preview can never disagree with
    the real conversion.
  * It renders a take-relative timeline (gripper width + |F| + marker lines +
    shaded episode bands with frame/duration labels) so gripper-close cycles and
    F/T contact spikes give physical anchors for placing a boundary at a rest point.
  * It AUTO-FLAGS implausible segments (too-short sliver = doubled marker;
    outlier-long = missed marker; >=2 grasp cycles in one episode = fused demos)
    with NO declared episode count required.
  * It lets the operator add / move / delete markers and atomically rewrite
    ``markers.npy`` (``.tmp`` + ``os.replace`` INSIDE the already-committed take
    dir — a single-file swap that never re-enters the take-commit machinery), then
    patches ``take_meta.json``'s advisory ``n_markers``/``n_episodes``/``marker_wall``.

All decision logic lives in pure, unit-tested helpers in ``umi_convert_core``
(``summarize_segments`` / ``flag_segments`` / ``grasp_close_times`` /
``grasps_per_segment``); this module is a thin viser layer over them plus a PIL
timeline renderer (the repo has no matplotlib/plotly — PIL + viser ``add_image``).

Run:
    .venv/bin/python recording/review_take.py data/recordings/session_XXX
    .venv/bin/python recording/review_take.py data/recordings/session_XXX/take_000
    # headless (no UI): print the flagged summary, or apply a marker edit + exit
    .venv/bin/python recording/review_take.py <take_dir> --headless
    .venv/bin/python recording/review_take.py <take_dir> --headless \
        --set-markers 2.0,4.0,6.0   # take-RELATIVE seconds; writes markers.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processing"))

import recording_format as fmt  # noqa: E402
import umi_convert_core as core  # noqa: E402

# The converter's repo-default calibration fallback (so gripper width matches what
# convert_to_umi.py will use). Imported lazily-safe: it only needs numpy/json.
import convert_to_umi as conv  # noqa: E402


# ---------------------------------------------------------------------------
# Loading one committed take (mirrors convert_to_umi.build_episodes_from_take)
# ---------------------------------------------------------------------------
class TakeData:
    """One committed take loaded for review. All times are TAKE-RELATIVE seconds
    (camera frame 0 == 0.0) for display + editing; we convert to/from the absolute
    wall-clock that markers.npy stores at the I/O boundary only."""

    def __init__(self, take_dir: str, session_dir: str):
        self.take_dir = os.path.abspath(take_dir)
        self.session_dir = os.path.abspath(session_dir)
        self.name = os.path.basename(self.take_dir)

        cam_dir = os.path.join(self.take_dir, fmt.CAMERA_DIR)
        cam_index = np.load(os.path.join(cam_dir, fmt.CAMERA_INDEX_FILE))
        if cam_index.shape[0] == 0:
            raise ValueError(f"{self.name}: no camera frames")
        cam_t_all = cam_index[:, 0]
        keep = core.make_strictly_increasing(cam_t_all)
        # Absolute, strictly-increasing camera grid — EXACTLY what the converter cuts against.
        self.cam_t_abs = cam_t_all[keep].astype(np.float64)
        self.t0 = float(self.cam_t_abs[0])
        self.cam_t = self.cam_t_abs - self.t0            # take-relative
        self.dur = float(self.cam_t[-1])

        # markers.npy: absolute wall-clock -> take-relative for display/edit.
        markers_abs = conv._load_markers(self.take_dir)
        self.markers = np.sort(markers_abs.astype(np.float64) - self.t0)

        # Gripper width in metres (same calib resolution the converter uses).
        self.gripper_t = np.zeros((0,), dtype=np.float64)
        self.width_m = np.zeros((0,), dtype=np.float64)
        self.width_open_m = 0.085
        self.gripper_calibrated = False
        gripper = self._load_opt(fmt.GRIPPER_FILE)
        base_calib = conv.load_gripper_calib(self.session_dir)
        # Prefer the take's RECORDED homing endpoints over the stale nominal JSON
        # ones (DM4310 re-references per power-cycle), exactly like convert_to_umi —
        # so the previewed gripper-width trace matches what conversion will produce.
        calib, _ = core.effective_gripper_calib(gripper, base_calib)
        if gripper.shape[0] > 0 and calib is not None:
            gripper = gripper[core.make_strictly_increasing(gripper[:, 0])]
            self.gripper_t = gripper[:, 0] - self.t0
            self.width_m = np.asarray(calib.to_width_m(gripper[:, 1]), dtype=np.float64).reshape(-1)
            self.width_open_m = float(calib.width_open_m)
            self.gripper_calibrated = True

        # F/T magnitude |F| (advisory contact signal).
        self.ft_t = np.zeros((0,), dtype=np.float64)
        self.fmag = np.zeros((0,), dtype=np.float64)
        ft = self._load_opt(fmt.FT_FILE)
        if ft.shape[0] > 0:
            ft = ft[core.make_strictly_increasing(ft[:, 0])]
            self.ft_t = ft[:, 0] - self.t0
            self.fmag = np.linalg.norm(ft[:, 1:4], axis=1)

    def _load_opt(self, filename: str) -> np.ndarray:
        p = os.path.join(self.take_dir, filename)
        if not os.path.exists(p):
            return np.zeros((0, 0), dtype=np.float64)
        return np.asarray(np.load(p), dtype=np.float64)

    # -- review computation (pure helpers) ----------------------------------
    def grasp_close_rel(self) -> np.ndarray:
        if self.gripper_t.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return core.grasp_close_times(self.gripper_t, self.width_m, self.width_open_m)

    def flagged_summary(self) -> list[dict]:
        """The episodes the CURRENT markers would produce, with advisory flags.
        cam_t is take-relative here, so dur_s / t_start / t_end are too — and the
        grasp-close times are in the same base, so multi_grasp lines up."""
        summ = core.summarize_segments(self.cam_t, self.markers)
        closes = self.grasp_close_rel()
        counts = core.grasps_per_segment(summ, closes) if closes.shape[0] else None
        return core.flag_segments(summ, grasp_counts=counts)

    # -- editing -------------------------------------------------------------
    def add_marker(self, t_rel: float) -> None:
        t_rel = float(np.clip(t_rel, 0.0, self.dur))
        self.markers = np.sort(np.append(self.markers, t_rel))

    def delete_nearest(self, t_rel: float) -> bool:
        if self.markers.shape[0] == 0:
            return False
        i = int(np.argmin(np.abs(self.markers - t_rel)))
        self.markers = np.delete(self.markers, i)
        return True

    def set_markers_rel(self, ts: np.ndarray) -> None:
        ts = np.asarray(ts, dtype=np.float64).reshape(-1)
        ts = ts[(ts >= 0.0) & (ts <= self.dur)]
        self.markers = np.sort(ts)

    def reload_from_disk(self) -> None:
        markers_abs = conv._load_markers(self.take_dir)
        self.markers = np.sort(markers_abs.astype(np.float64) - self.t0)

    def save(self) -> str:
        """Atomically rewrite markers.npy (back to ABSOLUTE wall-clock) inside the
        already-committed take dir, then patch take_meta's advisory fields. Uses a
        ``.npy``-suffixed tmp + os.replace so the swap is a single-file atomic op
        that never re-enters the recorder's .tmp/rename take-commit path."""
        markers_abs = np.sort(self.markers.astype(np.float64) + self.t0).reshape(-1, len(fmt.MARKER_COLS))
        final = os.path.join(self.take_dir, fmt.MARKER_FILE)
        tmp = os.path.join(self.take_dir, ".markers.review.tmp.npy")
        np.save(tmp, markers_abs)
        os.replace(tmp, final)        # markers.npy is the critical artifact — commit it FIRST.
        # take_meta's n_markers/n_episodes/marker_wall are ADVISORY (the converter
        # reads markers.npy directly via _load_markers, never these fields), so a
        # patch failure must NOT propagate and undo the good markers.npy write. Do
        # not reorder before the os.replace above: that would block the critical
        # write on a meta-write failure, which is strictly worse.
        try:
            self._patch_meta(markers_abs)
        except OSError as e:  # noqa: BLE001 - advisory only; markers.npy already safe
            print(f"[review] WARNING: markers.npy written but take_meta patch failed: {e}", flush=True)
        return final

    def _patch_meta(self, markers_abs: np.ndarray) -> None:
        p = os.path.join(self.take_dir, fmt.TAKE_META_FILE)
        if not os.path.exists(p):
            return
        try:
            with open(p) as f:
                meta = json.load(f)
        except (ValueError, OSError):
            return
        meta["n_markers"] = int(markers_abs.shape[0])
        meta["n_episodes"] = int(markers_abs.shape[0] + 1)
        meta["marker_wall"] = [float(x) for x in markers_abs[:, 0]] if markers_abs.shape[0] else []
        meta["markers_reviewed"] = True   # provenance: edited offline, not the raw capture
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, p)


# ---------------------------------------------------------------------------
# PIL timeline render (no matplotlib/plotly in this venv; PIL + viser add_image)
# ---------------------------------------------------------------------------
_FLAG_COLOR = {
    "too_short": (220, 70, 70),
    "outlier_long": (220, 70, 70),
    "multi_grasp": (230, 150, 40),
}


def render_timeline(take: TakeData, width_px: int = 1100, height_px: int = 420,
                    cursor_t: float | None = None) -> np.ndarray:
    """Render the review timeline to an HxWx3 uint8 RGB image: gripper-width and
    |F| traces on a shared take-relative x-axis, shaded episode bands (flagged
    ones tinted red/orange) with 'ep k: N fr, D.Ds' labels, vertical marker lines,
    grasp-close ticks, and an optional add-marker cursor line."""
    from PIL import Image, ImageDraw

    W, H = width_px, height_px
    img = Image.new("RGB", (W, H), (24, 26, 30))
    d = ImageDraw.Draw(img)

    left, right, top, bot = 56, W - 14, 24, H - 28
    plot_w = max(1, right - left)
    plot_h = max(1, bot - top)
    dur = max(take.dur, 1e-6)

    def x_of(t: float) -> int:
        return int(left + plot_w * (np.clip(t, 0.0, dur) / dur))

    flagged = take.flagged_summary()

    # --- episode bands (drawn first, behind the traces) ---
    band_cols = [(40, 64, 52), (40, 52, 70)]   # alternating green/blue tints
    for k, seg in enumerate(flagged):
        x0, x1 = x_of(seg["t_start"]), x_of(seg["t_end"])
        col = band_cols[k % 2]
        if seg["flags"]:
            col = (78, 42, 42) if ("outlier_long" in seg["flags"] or "too_short" in seg["flags"]) else (84, 64, 34)
        d.rectangle([x0, top, max(x0 + 1, x1), bot], fill=col)
        label = f"ep{seg['index']}: {seg['n_frames']}fr {seg['dur_s']:.1f}s"
        d.text((x0 + 3, top + 2), label, fill=(210, 210, 210))
        if seg["flags"]:
            d.text((x0 + 3, top + 14), "⚠ " + ",".join(seg["flags"]), fill=(245, 180, 120))

    # --- axes baseline ---
    d.rectangle([left, top, right, bot], outline=(70, 74, 80))
    mid = (top + bot) // 2

    def draw_trace(t: np.ndarray, y: np.ndarray, y0: int, y1: int,
                   color, lo: float | None = None, hi: float | None = None):
        if t.shape[0] < 2:
            return
        # Sanitize NaN/inf from a corrupt sensor row BEFORE min/max + the int()
        # conversions below: a NaN would make rng=NaN (bool(NaN) is True, so the
        # `or 1.0` guard misses it) and int(np.clip(NaN,...)) raises ValueError,
        # crashing the whole render. Bound from the FINITE values only (np.nanmax
        # would still return +inf if any sample is +inf, defeating nan_to_num).
        y = np.asarray(y, dtype=np.float64)
        finite = y[np.isfinite(y)]
        ylo = float(finite.min()) if finite.size else 0.0
        yhi = float(finite.max()) if finite.size else 0.0
        y = np.nan_to_num(y, nan=ylo, posinf=yhi, neginf=ylo)
        lo = ylo if lo is None else lo
        hi = yhi if hi is None else hi
        rng = (hi - lo) or 1.0
        pts = []
        for ti, yi in zip(t, y):
            px = x_of(float(ti))
            py = int(y1 - (y1 - y0) * ((float(yi) - lo) / rng))
            pts.append((px, int(np.clip(py, y0, y1))))
        d.line(pts, fill=color, width=1)

    # gripper width on the TOP half (with the 0.08 m UMI grasp threshold line)
    if take.width_m.shape[0] >= 2:
        wlo, whi = 0.0, max(take.width_open_m, float(np.max(take.width_m)), 1e-3)
        draw_trace(take.gripper_t, take.width_m, top + 2, mid - 2, (90, 200, 120), wlo, whi)
        gy = int((mid - 2) - ((mid - 2) - (top + 2)) * (0.08 / (whi - wlo)))
        if top < gy < mid:
            d.line([(left, gy), (right, gy)], fill=(70, 120, 90), width=1)
        d.text((2, top + 2), "grip", fill=(90, 200, 120))
        d.text((2, top + 14), "0.08", fill=(70, 130, 100))
    else:
        d.text((2, top + 2), "grip n/a", fill=(150, 120, 90))

    # |F| on the BOTTOM half
    if take.fmag.shape[0] >= 2:
        draw_trace(take.ft_t, take.fmag, mid + 2, bot - 2, (120, 160, 230))
        d.text((2, mid + 2), "|F|", fill=(120, 160, 230))
    else:
        d.text((2, mid + 2), "|F| n/a", fill=(120, 120, 150))

    # grasp-close ticks (short orange marks at the bottom)
    for ct in take.grasp_close_rel():
        gx = x_of(float(ct))
        d.line([(gx, bot - 6), (gx, bot)], fill=(230, 150, 40), width=2)

    # marker lines (yellow, full height) + index labels
    for i, mt in enumerate(take.markers):
        mx = x_of(float(mt))
        d.line([(mx, top), (mx, bot)], fill=(240, 220, 70), width=2)
        d.text((mx + 2, bot - 14), f"m{i}", fill=(240, 220, 70))

    # add-marker cursor (cyan dashed-ish)
    if cursor_t is not None:
        cx = x_of(float(cursor_t))
        for yy in range(top, bot, 6):
            d.line([(cx, yy), (cx, min(yy + 3, bot))], fill=(80, 220, 220), width=1)

    # x ticks (seconds)
    nticks = 6
    for i in range(nticks + 1):
        tt = dur * i / nticks
        tx = x_of(tt)
        d.line([(tx, bot), (tx, bot + 4)], fill=(90, 94, 100), width=1)
        d.text((tx - 8, bot + 8), f"{tt:.1f}", fill=(150, 154, 160))

    return np.asarray(img, dtype=np.uint8)


def summary_markdown(take: TakeData) -> str:
    flagged = take.flagged_summary()
    nflag = sum(1 for s in flagged if s["flags"])
    lines = [
        f"**{take.name}** — take {take.dur:.1f}s, "
        f"**{len(take.markers)} markers → {len(flagged)} episodes**, "
        f"{nflag} flagged",
    ]
    if not take.gripper_calibrated:
        lines.append("_gripper width unavailable (no gripper_range.json) — grasp-cycle flags off_")
    for s in flagged:
        tag = (" ⚠ " + ", ".join(s["flags"])) if s["flags"] else ""
        lines.append(f"- ep{s['index']}: {s['n_frames']} fr, {s['dur_s']:.2f}s{tag}"
                     + (f" — {s['note']}" if s["note"] else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Take discovery
# ---------------------------------------------------------------------------
def find_takes(path: str) -> tuple[str, list[str]]:
    """Resolve a session dir or a single take dir. Returns (session_dir, [take_dirs]).
    Only COMMITTED takes (with take_meta.json) are listed, mirroring the converter."""
    path = os.path.abspath(path)
    base = os.path.basename(path)
    if base.startswith("take_") and os.path.exists(os.path.join(path, fmt.TAKE_META_FILE)):
        return os.path.dirname(path), [path]
    takes = sorted(
        os.path.join(path, x) for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x)) and x.startswith("take_")
        and not x.endswith(".tmp")
        and os.path.exists(os.path.join(path, x, fmt.TAKE_META_FILE)))
    return path, takes


# ---------------------------------------------------------------------------
# Headless mode (print summary, or apply a marker edit + exit; no viser)
# ---------------------------------------------------------------------------
def run_headless(args) -> int:
    session_dir, takes = find_takes(args.path)
    if not takes:
        print(f"[review] no committed take_* dirs in {args.path}", file=sys.stderr)
        return 1
    # Parse --set-markers ONCE, up front, so a bad token fails loudly with a clear
    # message instead of a raw traceback mid-loop (and never after a partial write).
    set_ts: np.ndarray | None = None
    if args.set_markers is not None:
        try:
            set_ts = (np.array([float(x) for x in args.set_markers.split(",") if x.strip() != ""],
                               dtype=np.float64) if args.set_markers.strip() else np.zeros((0,)))
        except ValueError as e:
            print(f"error: --set-markers is not a valid comma-separated number list: {e}",
                  file=sys.stderr)
            return 1
        # A write intent on a whole multi-take session is ambiguous: refuse loudly
        # rather than silently no-op'ing (the operator believes markers were saved).
        if len(takes) > 1 and not args.apply_all:
            print(f"error: --set-markers given but '{os.path.basename(session_dir)}' has "
                  f"{len(takes)} takes; nothing was written. Re-run on a single take_NNN dir, "
                  "or add --apply-all to write the SAME markers to every take.", file=sys.stderr)
            return 1
    rc = 0
    for tdir in takes:
        try:
            take = TakeData(tdir, session_dir)
        except ValueError as e:
            print(f"[review] skip {os.path.basename(tdir)}: {e}")
            continue
        if set_ts is not None:
            take.set_markers_rel(set_ts)
            path = take.save()
            print(f"[review] wrote {os.path.relpath(path)} "
                  f"({len(take.markers)} markers → {len(take.markers) + 1} episodes)")
        print(summary_markdown(take).replace("**", "").replace("_", ""))
        if any(s["flags"] for s in take.flagged_summary()):
            rc = 2  # advisory: flagged segments present
        print("")
    return rc


# ---------------------------------------------------------------------------
# Interactive viser UI
# ---------------------------------------------------------------------------
def run_viser(args) -> int:
    import time
    import viser

    session_dir, takes = find_takes(args.path)
    if not takes:
        print(f"[review] no committed take_* dirs in {args.path}", file=sys.stderr)
        return 1

    loaded: dict[str, TakeData] = {}

    def get_take(tdir: str) -> TakeData | None:
        if tdir not in loaded:
            try:
                loaded[tdir] = TakeData(tdir, session_dir)
            except ValueError as e:
                print(f"[review] {os.path.basename(tdir)}: {e}", file=sys.stderr)
                return None
        return loaded[tdir]

    server = viser.ViserServer(host=args.host, port=args.port, label="UMI take review")
    gui = server.gui
    state = {"tdir": takes[0], "cursor": None, "dirty": False}

    options = [os.path.basename(t) for t in takes]
    take_dd = gui.add_dropdown("Take", options, initial_value=options[0])
    status = gui.add_markdown("loading…")
    img_handle = gui.add_image(np.zeros((10, 10, 3), dtype=np.uint8), label="Timeline")

    with gui.add_folder("Place a marker"):
        cursor_num = gui.add_number("Cursor (s, take-relative)", 0.0, min=0.0, step=0.05)
        add_btn = gui.add_button("➕ Add marker at cursor")
        del_btn = gui.add_button("🗑 Delete marker nearest cursor")
    with gui.add_folder("Save / revert"):
        save_btn = gui.add_button("💾 Save markers.npy")
        reload_btn = gui.add_button("↺ Reload from disk (discard edits)")
        save_status = gui.add_text("Saved", initial_value="(no edits)", disabled=True)

    def refresh() -> None:
        take = get_take(state["tdir"])
        if take is None:
            cursor_num.max = 0.0     # reset so a stale bound from a prior take can't leak
            status.content = f"**{os.path.basename(state['tdir'])}**: unreadable (no camera?)"
            return
        cursor_num.max = round(take.dur, 3)
        state["cursor"] = float(np.clip(cursor_num.value, 0.0, take.dur))
        # Never let a render error (e.g. unexpectedly corrupt sensor data) leave a
        # silently STALE timeline image that disagrees with the real data — the one
        # way this tool could mislead. Surface it in the status panel instead.
        try:
            img_handle.image = render_timeline(take, cursor_t=state["cursor"])
            status.content = summary_markdown(take)
        except Exception as e:  # noqa: BLE001
            status.content = (f"**{take.name}**: render error — {e}. "
                              "Markers are still safe; fix the take or report this.")
            print(f"[review] render error on {take.name}: {e!r}", flush=True)
            return
        save_status.value = ("unsaved edits" if state["dirty"] else "(in sync with disk)")

    @take_dd.on_update
    def _(_e) -> None:
        idx = options.index(take_dd.value)
        state["tdir"] = takes[idx]
        state["dirty"] = False
        refresh()

    @cursor_num.on_update
    def _(_e) -> None:
        refresh()

    @add_btn.on_click
    def _(_e) -> None:
        take = get_take(state["tdir"])
        if take is None:
            return
        take.add_marker(float(cursor_num.value))
        state["dirty"] = True
        refresh()

    @del_btn.on_click
    def _(_e) -> None:
        take = get_take(state["tdir"])
        if take is None:
            return
        if take.delete_nearest(float(cursor_num.value)):
            state["dirty"] = True
        refresh()

    @save_btn.on_click
    def _(_e) -> None:
        take = get_take(state["tdir"])
        if take is None:
            return
        path = take.save()
        state["dirty"] = False
        save_status.value = f"saved {len(take.markers)} markers → {len(take.markers)+1} ep"
        print(f"[review] wrote {os.path.relpath(path)}", flush=True)
        refresh()

    @reload_btn.on_click
    def _(_e) -> None:
        take = get_take(state["tdir"])
        if take is None:
            return
        take.reload_from_disk()
        state["dirty"] = False
        refresh()

    refresh()
    print(f"[review] http://{args.host}:{args.port}  ({len(takes)} take(s) in "
          f"{os.path.basename(session_dir)})", flush=True)
    print("[review] add/delete markers at the cursor, then Save. The split preview "
          "is byte-for-byte what convert_to_umi.py will produce.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[review] bye", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="a session dir or a single take_NNN dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--headless", action="store_true",
                   help="print the flagged summary (and optionally apply --set-markers) without the UI")
    p.add_argument("--set-markers", default=None,
                   help="headless: comma-separated take-RELATIVE seconds to WRITE as markers.npy "
                        "(e.g. '2.0,4.0,6.0'; empty string clears all markers)")
    p.add_argument("--apply-all", action="store_true",
                   help="with --set-markers on a whole session, apply to every take (default: only a single take)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.headless or args.set_markers is not None:
        return run_headless(args)
    return run_viser(args)


if __name__ == "__main__":
    raise SystemExit(main())
