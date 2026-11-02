#!/usr/bin/env python3
"""Out-of-process UMI data recorder (hub-and-spoke consumer), continuous-take model.

Subscribes to the SAME localhost WebSocket bridges the viz consumes and writes
each modality to disk at its own native rate, with per-sample wall-clock
timestamps — the RAW recording the offline converter turns into a UMI Zarr
ReplayBuffer.

CONTINUOUS-TAKE model (DECISION-E)
----------------------------------
One Start..Stop span is a single "take", recorded UNBROKEN. The operator does not
gesture to begin/end each demonstration; they drop an episode-split MARKER at the
boundary between demonstrations (long-press ``o+p`` 1.5 s on the macropad -> a
``{"cmd":"marker"}`` on this recorder's control WS, or the GUI "Mark boundary"
button). Markers are stored as bare wall-clock timestamps in ``markers.npy``; the
OFFLINE converter splits each take into N+1 UMI episodes at the N marker times.
Capture stays dumb + lossless; all segmentation is post-processing.

Why a separate process (not a thread inside collection_viz.py): under CPython's
GIL an in-viz recorder doing JPEG/np.save work would contend with the Bota
SensorReader EKF — the viz ALREADY throttles its whole render loop during
payload recompensation for exactly this reason. A separate process removes the
recorder from the viz interpreter entirely and matches the rig's
one-singleton-per-process model.

Capture model (verified against the bridges): we record at MESSAGE ARRIVAL, not
by polling a latest-value register, so we losslessly keep every frame the bridge
emits. Per-stream notes:
  * tracker  (:8772) — broadcast SE3; we keep every frame, stamping the bridge's
                       wire ``t`` and ``seq``. RAW world pose, quaternion WXYZ
                       (see recording_format banner — converted to xyzw ONLY in
                       the converter).
  * camera   (:8873) — dedup by ``seq`` (the bridge bumps it only on a genuine
                       new capture), so this is lossless; JPEG bytes saved verbatim.
  * gripper  (:8774) — every encoder frame (t, encoder/cmd/open/closed rad).
  * F/T      (:8775) — BATCHES of full-rate (~250 Hz) samples from the viz's
                       opt-in F/T publisher (a per-sample socket can't keep up;
                       batches do). The RAW wrench is always saved (ft_wrench.npy);
                       when a payload config is active the COMPENSATED wrench is
                       saved too (ft_wrench_comp.npy) + ft_compensated in take_meta.

Controls (start/stop/marker/discard a take) arrive on the recorder's OWN control
WS (:8776) from the viz GUI buttons and the keyboard-gesture bridge; the recorder
also broadcasts a status frame there for live feedback.

Run (live):
    .venv/bin/python recording/recorder.py --out data/recordings
Run (self-contained smoke, no hardware/viz — synthesizes all streams + a scripted
take with 2 markers -> 3 episodes, then exits):
    .venv/bin/python recording/recorder.py --mock --out /tmp/rec_smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recording_format as fmt  # noqa: E402


# ---------------------------------------------------------------------------
# Per-take in-RAM buffers (flushed to disk on stop)
# ---------------------------------------------------------------------------
@dataclass
class FtChannel:
    """One hand's F/T stream + its provenance latches.

    There is one of these per HAND role (see recording_format's PER-HAND banner): the
    LEFT and RIGHT Bota sensors are independent devices with independent payload
    configs, tares and — crucially — independent ``ft_sign_verified``, so their
    provenance must NOT be merged. Latching them jointly is how an unverified
    right-hand RNEA sign would silently mark the left hand's trace suspect too."""

    raw: list = field(default_factory=list)    # rows [t, Fx,Fy,Fz, Tx,Ty,Tz]  (RAW; always)
    comp: list = field(default_factory=list)   # rows [t, Fx,Fy,Fz, Tx,Ty,Tz]  (COMPENSATED; if available)
    # Provenance, latched from the FIRST ft_batch of the take (None until then).
    # See recording_format's F/T invariant: every batch in a take must agree; a
    # disagreement sets `mixed` True (loud) so a conflict is never silently lost.
    compensated: "bool | None" = None
    mixed: bool = False
    # WHICH compensation engine produced the compensated wrench this take (bota|rnea),
    # latched from the FIRST batch that carries the key; a mid-take disagreement sets
    # algo_mixed True (loud) — same discipline as `compensated`, because switching the
    # GUI dropdown mid-episode blends two DIFFERENT compensation models (Bota
    # gravity-only vs RNEA full-dynamic) into one ft_wrench_comp.npy. None until seen.
    comp_algo: "str | None" = None
    algo_mixed: bool = False
    # Whether the active engine's global sign was verified on THIS hand (an RNEA
    # config seeded from the other hand carries an UNVERIFIED sign that can INVERT
    # the compensated wrench). Latched "last non-null wins"; False if any batch was
    # unverified so a suspect take is never mislabeled trustworthy.
    sign_verified: "bool | None" = None
    tare_wrench: "list | None" = None   # compensator DC offset (provenance), or None
    tare_wall: "float | None" = None    # time.time() of that tare
    # Every tare seen during the take, as [wall, Fx,Fy,Fz, Tx,Ty,Tz] rows (the
    # start-of-episode auto-tare, plus any manual/end tares whose batch lands
    # before the buffer detaches). Lets offline tooling + replay draw a vertical
    # marker at each tare (R5). `tare_wrench`/`tare_wall` above stay the single
    # "last non-null" provenance latch for back-compat.
    tare_events: list = field(default_factory=list)


@dataclass
class CamChannel:
    """One hand's camera stream + its timeline-health counters.

    Per-hand because each D405 is its own bridge process with its own pacing: a
    non-uniform grid on ONE camera must be attributable to that camera. The LEFT
    camera's grid is the MASTER timeline of the dataset (the converter resamples every
    other stream onto it), so its counters are the ones that gate a session."""

    jpegs: list = field(default_factory=list)   # bytes, verbatim from the bridge
    index: list = field(default_factory=list)   # rows [t, seq]
    last_seq: int = -1                          # dedup: the bridge re-broadcasts
    seq_gaps: int = 0                           # frames the bridge produced but we never saw
    capture_gaps: dict = field(default_factory=dict)  # histogram of SDK-frame-number deltas
    last_capture_seq: int = -1


@dataclass
class TakeBuffers:
    task: str = ""
    start_wall: float = 0.0
    tracker: list = field(default_factory=list)   # rows [t, seq, x,y,z, qw,qx,qy,qz, stale]  (PRIMARY = LEFT)
    # EVERY non-primary tracker role, keyed by role, rows identical to `tracker` above:
    # 'right' (2pt+), 'head' (3pt+), 'left_foot'/'right_foot' (5pt). A DICT rather than one
    # field per role because the 5pt feet are the third and fourth of these and each new
    # field had to be threaded through five places (ingest, save, counts, status, mock) —
    # a role missed in one of them records an EMPTY file with no error anywhere. Keys are
    # pre-created for every role fmt knows, so "declared but never streamed" is an empty
    # list (no file written) rather than a KeyError.
    tracker_by_role: dict = field(default_factory=dict)
    markers: list = field(default_factory=list)    # episode-split wall-clock times [t]
    # Per-HAND modalities, keyed by role ('left' = primary/historical, 'right'). The
    # chest tracker is deliberately absent from all three: it has no camera, F/T or
    # gripper (recording_format's PER-HAND banner). A role whose device never streamed
    # keeps an EMPTY channel, and _flush_take writes no file for it — absence stays a
    # fact rather than being zero-filled into a fake reading.
    gripper_by_role: dict = field(default_factory=dict)   # role -> rows [t, enc, cmd, open, closed]
    ft_by_role: dict = field(default_factory=dict)        # role -> FtChannel
    cam_by_role: dict = field(default_factory=dict)       # role -> CamChannel

    def __post_init__(self) -> None:
        for role in fmt.TRACKER_FILE_BY_ROLE:
            if role != "left":
                self.tracker_by_role.setdefault(role, [])
        for role in fmt.HAND_ROLES:
            self.gripper_by_role.setdefault(role, [])
            self.ft_by_role.setdefault(role, FtChannel())
            self.cam_by_role.setdefault(role, CamChannel())

    @property
    def tracker_right(self) -> list:
        """The RIGHT hand's rows. Kept as a name so existing readers are unchanged."""
        return self.tracker_by_role["right"]

    @property
    def tracker_head(self) -> list:
        """The CHEST's rows (see :attr:`tracker_right`)."""
        return self.tracker_by_role["head"]

    # -- convenience accessors for the PRIMARY (LEFT) hand ----------------------
    # The left hand is referenced constantly (it is the master camera grid, the
    # unsuffixed on-disk files and the take-completeness gate), so name it once here
    # instead of spelling ['left'] at ~20 call sites.
    @property
    def gripper(self) -> list:
        return self.gripper_by_role["left"]

    @property
    def ft(self) -> FtChannel:
        return self.ft_by_role["left"]

    @property
    def cam(self) -> CamChannel:
        return self.cam_by_role["left"]


class Recorder:
    """Holds recording state + buffers. ALL state lives on the single asyncio loop
    thread: every frame handler, the control handler, and the status broadcaster
    run as coroutines on that one loop, and the blocking flush is dispatched to an
    executor only AFTER the buffer is detached into a private local. So there is no
    cross-thread shared mutable state and no lock is needed (and none is used)."""

    def __init__(self, out_root: str, session_name: str | None = None,
                 mode: str = "1pt", serial_role_map: "dict | None" = None,
                 camera_fps_nominal: float = 30.0,
                 world_align: "dict | None" = None):
        self._out_root = out_root
        # Nominal CAPTURE fps of the camera bridge (NOT the publish rate). Used
        # only to bucket the recorded camera dt into integer capture periods for
        # the uniformity check in _flush_take.
        self._camera_fps_nominal = float(camera_fps_nominal)
        # Tracking mode (1pt/2pt/3pt) + the canonical serial->role map for this
        # session, recorded in session_meta + each take_meta so offline tooling
        # reads the tracker count / LEFT-canonical roles explicitly instead of
        # guessing from which tracker_*.npy files exist.
        self._mode = mode
        # WORLD ALIGNMENT provenance. The tracker bridge multiplies EVERY recorded pose
        # by R_world (--yaw-deg / --world-rpy-deg / --world-quat), and until now that
        # rotation was recorded NOWHERE — printed once to stdout and lost. Two sessions
        # taken under different alignments were byte-indistinguishable, so they could
        # not be pooled, compared, or checked against a deploy-time frame. Captured here
        # so "aligned world" is a reconstructible claim rather than a label.
        # None/{} = never supplied (legacy launch) — deliberately NOT defaulted to
        # identity, since "unknown" and "identity" are different facts.
        self._world_align = dict(world_align or {})
        self._serial_role_map = dict(serial_role_map or {})
        # BASE-STATION provenance, latched from the tracker bridge's optional
        # "lighthouses" wire field (see vive_tracker_bridge.snapshot). How many VIVE
        # stations were triangulating is a first-class property of a dataset's
        # tracking quality — a 2-of-3 session has real occlusion dropouts a 3-station
        # one does not — and it was recorded nowhere, so a glitchy session could not
        # be explained after the fact. {} = the bridge never sent it (a pre-2026-08-27
        # bridge), which is NOT the same claim as "no stations".
        self._lighthouses: dict = {}
        self.recording = False
        # 'c' key -> return-to-home relay. The keyboard bridge sends {"cmd":"home"};
        # we bump this counter and surface it in status so the viz (the IK owner)
        # observes the increment and resets on its render thread. The recorder
        # itself never touches the robot — it only relays the intent.
        self.home_seq = 0
        self.take_idx = 0
        self.buf: TakeBuffers | None = None
        self.last_saved = ""
        # Camera dedup + timeline health now live PER HAND on TakeBuffers.cam_by_role
        # (CamChannel), because each D405 is its own bridge process with its own pacing
        # and a non-uniform grid must be attributable to one camera. The counters exist
        # at all because a ~12 Hz 2-vs-3-capture-period alternation shipped 97 takes
        # without a single log line: `approx_hz.camera = 11.8` was written into all 97
        # take_meta.json files and nothing ever read it.
        self._last_saved_take_dir = ""
        # Session dir is created lazily on the first start.
        self._session_dir: str | None = None
        self._session_name = session_name

    # -- session/take lifecycle (loop-thread only) --------------------------
    def _ensure_session(self) -> str:
        if self._session_dir is None:
            name = self._session_name or f"session_{_ts_str(time.time())}"
            sd = os.path.join(self._out_root, name)
            os.makedirs(os.path.join(sd, fmt.CALIB_DIR), exist_ok=True)
            self._session_dir = sd
            self._write_session_meta()
            print(f"[recorder] session -> {sd}", flush=True)
        return self._session_dir

    def _write_session_meta(self) -> None:
        meta = {
            "created_wall": time.time(),
            "created_str": _ts_str(time.time()),
            "clock": "time.time() wall-clock seconds (single rig clock)",
            "quaternion_convention": fmt.QUATERNION_CONVENTION,  # 'wxyz' (raw); converter maps to xyzw
            "record_model": "continuous_take_with_episode_split_markers",
            # Tracking mode + LEFT-canonical role provenance (see recording_format
            # LEFT-canonical banner). primary_role is always 'left'; tracker.npy is
            # that role, tracker_right.npy/tracker_head.npy are the others.
            "mode": self._mode,
            "primary_role": "left",
            "serial_role_map": self._serial_role_map,
            # The libsurvive-world -> robot-world rotation baked into every pose in
            # this session (see self._world_align). null = not supplied at launch,
            # which is NOT the same claim as identity.
            "world_align": self._world_align or None,
            # How many VIVE base stations were in the solve when this session opened
            # (see self._lighthouses). null = the bridge did not report it. Snapshot at
            # session creation, NOT per take: a station joining mid-session is rare and
            # rewriting session_meta under a running recording is not worth the risk.
            # Deliberately NOT in merge_sessions.CONTRACT_FIELDS — two sittings of the
            # same task may legitimately have had different stations up, and that must
            # not refuse a merge.
            "lighthouses": self._lighthouses or None,
            "schema_version": 4,
        }
        with open(os.path.join(self._session_dir, fmt.SESSION_META_FILE), "w") as f:
            json.dump(meta, f, indent=2)

    def start(self, task: str = "") -> None:
        if self.recording:
            return
        self._ensure_session()
        # A fresh TakeBuffers resets every per-hand camera dedup / gap counter with it
        # (they live on CamChannel), so there is nothing left to clear by hand here.
        self.buf = TakeBuffers(task=task, start_wall=time.time())
        self.recording = True
        print(f"[recorder] START {fmt.take_dir_name(self.take_idx)} task={task!r}", flush=True)

    def home(self) -> int:
        """Relay a 'return to initial pose' intent from the 'c' key (R4). We don't
        own the robot/IK — just bump a counter the viz observes in status and turns
        into its render-thread reset. Works whether or not a take is recording."""
        self.home_seq += 1
        print(f"[recorder] HOME request #{self.home_seq} (relayed to viz)", flush=True)
        return self.home_seq

    def mark(self) -> int:
        """Record an episode-split marker at the current wall-clock (DECISION-E).
        No-op when not recording. Returns the running marker count."""
        if not self.recording or self.buf is None:
            return 0
        self.buf.markers.append([time.time()])
        n = len(self.buf.markers)
        print(f"[recorder] MARKER #{n} at +{time.time() - self.buf.start_wall:.1f}s "
              f"-> splits the take into {n + 1} episodes", flush=True)
        return n

    def stop(self) -> str | None:
        """Stop + flush synchronously (used by the control handler). Disk I/O runs
        inline here; for the live path prefer :meth:`stop_async` which offloads
        the flush so the asyncio loop keeps draining the bridges."""
        buf, take_dir = self._detach_for_flush()
        if buf is None:
            return None
        self._flush_take(take_dir, buf)
        print(f"[recorder] STOP -> saved {take_dir}", flush=True)
        return take_dir

    async def stop_async(self) -> str | None:
        """Stop + flush OFF the asyncio loop (M1): np.save + JPEG writes are
        blocking and would otherwise stall every WS reader (and overflow the F/T
        ring) for the flush duration. buf is detached first so it's private."""
        import asyncio
        buf, take_dir = self._detach_for_flush()
        if buf is None:
            return None
        await asyncio.get_running_loop().run_in_executor(None, self._flush_take, take_dir, buf)
        print(f"[recorder] STOP -> saved {take_dir}", flush=True)
        return take_dir

    async def flush_on_shutdown(self) -> str | None:
        return await self.stop_async()

    def _detach_for_flush(self):
        """Atomically take the current buffer AND reserve its take dir / index on
        the LOOP THREAD (review #1/#3). Returns (buf, take_dir) or (None, None).

        The index reservation + ``_last_saved`` bookkeeping happen HERE, before any
        ``await`` — so two stops can never race to the same ``take_NNN`` even if a
        future second stop-capable client overlaps with an in-flight flush, and the
        commit state is correct the instant the buffer is detached (so a shutdown
        that cancels the post-await step still leaves a coherent counter)."""
        if not self.recording or self.buf is None:
            return None, None
        self.recording = False
        buf = self.buf
        self.buf = None
        # M4: require BOTH camera frames AND tracker to save — the converter needs
        # both to build a UMI episode; a camera-only/tracker-only take is unusable.
        # Gated on the LEFT (primary) hand only: its camera is the MASTER grid every
        # other stream resamples onto. A missing RIGHT camera is a degraded take, not
        # an unusable one, so it must NOT throw away a good left-hand demonstration.
        if not buf.cam.index or not buf.tracker:
            print("[recorder] STOP: incomplete take (need LEFT camera + tracker); nothing saved", flush=True)
            # Review #2: a rejected take must NOT leave _last_saved_take_dir pointing
            # at the PREVIOUS good take, or a follow-up Discard would delete that good
            # take. Clear the pointer so Discard can only target a take this stop saved
            # (here: none).
            self._last_saved_take_dir = ""
            self.last_saved = ""
            return None, None
        take_dir = os.path.join(self._session_dir, fmt.take_dir_name(self.take_idx))
        self.take_idx += 1
        self._last_saved_take_dir = take_dir
        self.last_saved = take_dir
        return buf, take_dir

    def discard_last(self) -> bool:
        """Delete the most recently saved take dir (post-stop) or the in-progress
        buffer if currently recording."""
        if self.recording:
            self.buf = None
            self.recording = False
            print("[recorder] DISCARD in-progress take", flush=True)
            return True
        if self._last_saved_take_dir and os.path.isdir(self._last_saved_take_dir):
            shutil.rmtree(self._last_saved_take_dir, ignore_errors=True)
            print(f"[recorder] DISCARD saved {self._last_saved_take_dir}", flush=True)
            self._last_saved_take_dir = ""
            self.take_idx = max(0, self.take_idx - 1)
            self.last_saved = ""
            return True
        return False

    def _flush_take(self, take_dir: str, buf: TakeBuffers) -> None:
        # M2: ATOMIC flush. Write everything into a sibling .tmp dir, write
        # take_meta.json LAST, then os.rename to the final name. The converter
        # treats take_meta.json as the commit marker and skips any dir lacking it
        # — so a crash/SIGTERM mid-flush leaves a .tmp (ignored), never a
        # half-written take that aborts conversion. This runs in an executor (off
        # the asyncio loop), so it must touch only its private `buf`/paths.
        tmp_dir = take_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        tracker = np.asarray(buf.tracker, dtype=np.float64).reshape(-1, len(fmt.TRACKER_COLS))
        markers = np.asarray(buf.markers, dtype=np.float64).reshape(-1, len(fmt.MARKER_COLS))

        np.save(os.path.join(tmp_dir, fmt.TRACKER_FILE), tracker)   # PRIMARY = LEFT
        # 2pt/3pt/5pt: save every secondary tracker channel that actually streamed
        # (additive; old tooling ignores unknown .npy files). Same column layout as
        # TRACKER_COLS, and the same ABSENCE RULE as the right hand's modalities — a role
        # with no samples writes NO file, so "this puck was not in the session" never
        # becomes an empty array that reads as "the puck was there and silent".
        for _role, _rows in buf.tracker_by_role.items():
            if _rows:
                np.save(os.path.join(tmp_dir, fmt.TRACKER_FILE_BY_ROLE[_role]),
                        np.asarray(_rows, dtype=np.float64).reshape(-1, len(fmt.TRACKER_COLS)))
        np.save(os.path.join(tmp_dir, fmt.MARKER_FILE), markers)

        # ===== per-HAND modalities (LEFT keeps the historical unsuffixed names) =====
        # One loop, two hands, names from fmt.HAND_FILES_BY_ROLE — so a right-hand file
        # cannot be forgotten and the two sides cannot drift in column layout.
        #
        # ABSENCE RULE: the LEFT hand always writes gripper.npy + ft_wrench.npy (even
        # empty) because those files have always existed and downstream treats a 0-row
        # array as "no samples"; the RIGHT hand writes NOTHING unless its device
        # actually streamed. That asymmetry is deliberate: an empty ft_wrench_right.npy
        # would claim a right-hand sensor was present and silent, which is a different
        # (and much more alarming) fact than "no right sensor is wired up yet".
        arrs: dict[str, dict[str, np.ndarray]] = {}
        for role in fmt.HAND_ROLES:
            names = fmt.HAND_FILES_BY_ROLE[role]
            is_primary = role == "left"
            ftc = buf.ft_by_role[role]
            camc = buf.cam_by_role[role]
            a = {
                "gripper": np.asarray(buf.gripper_by_role[role], dtype=np.float64
                                      ).reshape(-1, len(fmt.GRIPPER_COLS)),
                "ft": np.asarray(ftc.raw, dtype=np.float64).reshape(-1, len(fmt.FT_COLS)),
                "ft_comp": np.asarray(ftc.comp, dtype=np.float64).reshape(-1, len(fmt.FT_COLS)),
                "cam_index": np.asarray(camc.index, dtype=np.float64
                                        ).reshape(-1, len(fmt.CAMERA_INDEX_COLS)),
            }
            arrs[role] = a
            if is_primary or a["gripper"].shape[0]:
                np.save(os.path.join(tmp_dir, names["gripper"]), a["gripper"])
            if is_primary or a["ft"].shape[0]:
                np.save(os.path.join(tmp_dir, names["ft"]), a["ft"])
            # Compensated sidecar: write ONLY when we actually captured compensated
            # samples (raw-only sessions never produce them). Its absence is how the
            # converter/replay know the take is raw-only — so it is never pre-created.
            if a["ft_comp"].shape[0]:
                np.save(os.path.join(tmp_dir, names["ft_comp"]), a["ft_comp"])
            # Tare events [K,7] = wall, Fx..Tz — one row per tare during the take (R5).
            # Written only when at least one tare landed (raw/mock takes may have none).
            if ftc.tare_events:
                np.save(os.path.join(tmp_dir, names["tare_events"]),
                        np.asarray(ftc.tare_events, dtype=np.float64).reshape(-1, 7))
            # Camera dir: always for the primary (its grid is the take's master
            # timeline), only when frames arrived for the right hand.
            if is_primary or camc.index:
                cam_dir = os.path.join(tmp_dir, names["camera_dir"])
                os.makedirs(cam_dir, exist_ok=True)
                np.save(os.path.join(cam_dir, fmt.CAMERA_INDEX_FILE), a["cam_index"])
                for i, jpeg in enumerate(camc.jpegs):
                    with open(os.path.join(cam_dir, fmt.camera_frame_name(i)), "wb") as f:
                        f.write(jpeg)

        # Primary-hand shorthands for the meta block below (unchanged semantics).
        gripper = arrs["left"]["gripper"]
        ft = arrs["left"]["ft"]
        ft_comp = arrs["left"]["ft_comp"]
        cam_index = arrs["left"]["cam_index"]

        # Provenance + advisory continuity in take_meta (the commit marker).
        stop_wall = time.time()
        stale_frac = 0.0
        if tracker.shape[0]:
            sc = fmt.TRACKER_COLS.index("stale")
            stale_frac = float((tracker[:, sc] >= 0.5).mean())
        # F/T provenance, resolved PER HAND (the two Bota sensors have independent
        # payload configs, tares and sign verification — see FtChannel). The primary
        # hand's values keep the historical flat take_meta keys; the right hand's live
        # under `hands.right` so no existing reader changes meaning.
        def _ft_provenance(role: str) -> dict:
            ftc = buf.ft_by_role[role]
            comp_rows = int(arrs[role]["ft_comp"].shape[0])
            # 'mixed' if the batch flag disagreed mid-take, else the latched bool — but
            # ONLY meaningful when FT was actually recorded AND a compensated sidecar
            # was written (a no-FT take is never 'compensated').
            have_comp = comp_rows > 0
            if ftc.mixed:
                compensated: "bool | str" = "mixed"
            else:
                compensated = bool(ftc.compensated) and have_comp
            # WHICH compensation engine (bota|rnea|'mixed'), only meaningful when a
            # compensated sidecar was written (a raw take has no algo). 'mixed' if the
            # batch flag flipped mid-take. None otherwise — downstream defaults to bota.
            if not have_comp:
                algo: "str | None" = None
                signv: "bool | None" = None
            elif ftc.algo_mixed:
                algo = "mixed"
                signv = bool(ftc.sign_verified) if ftc.sign_verified is not None else None
            else:
                algo = ftc.comp_algo
                signv = ftc.sign_verified
            return {
                "ft_compensated": compensated,
                "ft_comp_algo": algo,
                "ft_sign_verified": signv,
                "ft_tare_wrench": ftc.tare_wrench,
                "ft_tare_wall": ftc.tare_wall,
                "tare_event_wall": [float(r[0]) for r in ftc.tare_events],
            }

        prov = {role: _ft_provenance(role) for role in fmt.HAND_ROLES}
        ft_compensated = prov["left"]["ft_compensated"]
        ft_comp_algo = prov["left"]["ft_comp_algo"]
        ft_sign_verified = prov["left"]["ft_sign_verified"]
        sec_counts = {role: len(rows) for role, rows in buf.tracker_by_role.items()}
        tracker_right = np.asarray(buf.tracker_right, dtype=np.float64).reshape(-1, len(fmt.TRACKER_COLS))
        tracker_rows = {"left": tracker, "right": tracker_right}
        # WHICH modalities each hand actually contributed. Written so downstream never
        # has to infer capability from which files happen to exist — "the right Bota is
        # not wired up yet" and "the right Bota was wired up and read nothing" are
        # different facts, and only this block can tell them apart.
        hands_meta = {}
        for role in fmt.HAND_ROLES:
            a = arrs[role]
            hands_meta[role] = {
                "tracker": int(tracker_rows[role].shape[0]),
                "camera": int(a["cam_index"].shape[0]),
                "ft": int(a["ft"].shape[0]),
                "ft_comp": int(a["ft_comp"].shape[0]),
                "gripper": int(a["gripper"].shape[0]),
                "files": {k: v for k, v in fmt.HAND_FILES_BY_ROLE[role].items()},
                **prov[role],
            }
        meta = {
            "task": buf.task,
            "take_dir": os.path.basename(take_dir),
            "start_wall": buf.start_wall,
            "stop_wall": stop_wall,
            "quaternion_convention": fmt.QUATERNION_CONVENTION,  # wxyz (raw)
            # LEFT-canonical: tracker.npy = primary (left) hand. See session_meta.
            "mode": self._mode,
            "primary_role": "left",
            "counts": {
                "tracker": int(tracker.shape[0]),        # PRIMARY = LEFT hand
                # One key per SECONDARY role, always present and 0 when that puck was not
                # in this session (the same "absence is a fact" rule as the hand
                # modalities). Named for the on-disk file's stem so `counts` and the take
                # directory read the same: tracker_right / tracker_head /
                # tracker_left_foot / tracker_right_foot.
                **{f"tracker_{role}": int(n) for role, n in sec_counts.items()},
                "gripper": int(gripper.shape[0]),
                "ft": int(ft.shape[0]),
                "ft_comp": int(ft_comp.shape[0]),
                "camera": int(cam_index.shape[0]),
                "tare_events": int(len(buf.ft.tare_events)),
                # RIGHT-hand per-hand modalities. ALWAYS present, 0 when that device
                # never streamed — a stable key set means the status/audit tooling does
                # not have to branch on hardware. See take_meta["hands"] for provenance.
                "gripper_right": int(arrs["right"]["gripper"].shape[0]),
                "ft_right": int(arrs["right"]["ft"].shape[0]),
                "ft_comp_right": int(arrs["right"]["ft_comp"].shape[0]),
                "camera_right": int(arrs["right"]["cam_index"].shape[0]),
                "tare_events_right": int(len(buf.ft_by_role["right"].tare_events)),
            },
            # Per-hand modality provenance (LEFT = primary). The flat ft_* keys below
            # stay the LEFT hand's, byte-identical to before.
            "hands": hands_meta,
            "ft_valid": bool(ft.shape[0] > 0),
            # ft_wrench.npy is ALWAYS raw; ft_wrench_comp.npy holds the compensated
            # wrench when present. ft_compensated tells downstream which to prefer.
            "ft_compensated": ft_compensated,
            # Which engine produced ft_wrench_comp.npy (bota|rnea|'mixed'|null) + whether
            # its global sign was verified on this hand. Lets the offline pipeline
            # segregate takes that silently switched compensation model / used an
            # unverified (possibly sign-inverted) RNEA config. null = raw take (no comp).
            "ft_comp_algo": ft_comp_algo,
            "ft_sign_verified": ft_sign_verified,
            "ft_tare_wrench": buf.ft.tare_wrench,   # LAST compensator DC offset (provenance) or null
            "ft_tare_wall": buf.ft.tare_wall,       # wall-clock of that last tare or null
            # Every tare during the take (R5) — start-of-episode auto-tare + any
            # manual/end tare captured before stop; empty if none landed. Full 6-vec
            # rows live in tare_events.npy; here we surface just the wall-clock times.
            "tare_event_wall": [float(r[0]) for r in buf.ft.tare_events],
            "tracker_stale_fraction": round(stale_frac, 4),
            # Episode-split markers (absolute wall-clock); n_episodes == n_markers+1.
            "n_markers": int(markers.shape[0]),
            "n_episodes": int(markers.shape[0] + 1),
            "marker_wall": [float(x) for x in markers[:, 0]] if markers.shape[0] else [],
            "stream_t_span": {
                "tracker": [float(tracker[0, 0]), float(tracker[-1, 0])] if tracker.shape[0] else None,
                "ft": [float(ft[0, 0]), float(ft[-1, 0])] if ft.shape[0] else None,
                "camera": [float(cam_index[0, 0]), float(cam_index[-1, 0])] if cam_index.shape[0] else None,
            },
        }
        dur = max(stop_wall - buf.start_wall, 1e-6)
        meta["approx_hz"] = {k: round(v / dur, 1) for k, v in meta["counts"].items()}

        # ===== camera timeline uniformity (loud, and on disk) =====
        # `camera_dt_k` is the histogram of round(dt * camera_fps_nominal); a
        # single key == a uniform grid. `camera_capture_gaps` is the same thing
        # measured from the SDK frame number (authoritative when the bridge sends
        # `capture_seq`). Both land in take_meta so the converter and the offline
        # audit can gate on them instead of re-deriving.
        #
        # Run PER HAND: each D405 is its own bridge process with its own pacing, so a
        # non-uniform grid must be attributable to one camera. The LEFT hand's numbers
        # keep the historical flat keys (`camera_dt_k`, `camera_grid_uniform`, …) because
        # its grid is the MASTER timeline the converter resamples everything onto; the
        # RIGHT hand's land under `hands.right.camera_*` and warn without gating.
        def _cam_health(cam_idx: np.ndarray, camc: CamChannel) -> dict:
            cam_t = cam_idx[:, 0] if cam_idx.shape[0] else np.zeros(0)
            cam_dt = np.diff(cam_t) if cam_t.size > 1 else np.zeros(0)
            # From the camera timestamps themselves, not from counts/duration: the
            # latter is meaningless if the take is shorter than one frame period.
            cam_hz = 1.0 / float(np.mean(cam_dt)) if cam_dt.size and np.mean(cam_dt) > 0 else 0.0
            k_hist: dict[str, int] = {}
            uniform = True
            if cam_dt.size:
                fps_nom = float(self._camera_fps_nominal or 0.0)
                if fps_nom <= 0:
                    # No nominal fps configured: fall back to the observed median step.
                    fps_nom = 1.0 / max(float(np.median(cam_dt)), 1e-9)
                ks, counts = np.unique(np.round(cam_dt * fps_nom).astype(int), return_counts=True)
                k_hist = {str(int(k)): int(c) for k, c in zip(ks, counts)}
                uniform = bool(counts.max() >= 0.99 * counts.sum())
            # `capture_seq` (the SDK frame number) is the AUTHORITATIVE signal and it
            # needs NO nominal fps: a correctly paced bridge produces exactly one
            # positive gap value. The dt-based `k` histogram above needs
            # camera_fps_nominal to be right, and a wrong one silently collapses the
            # histogram into a single bucket (the round(dt/median) degeneracy) -- so it
            # must not be the only test.
            cap_gaps = camc.capture_gaps
            cap_uniform = (len(cap_gaps) == 1 and min(cap_gaps) > 0) if cap_gaps else None
            if cap_uniform is not None:
                uniform = uniform and cap_uniform
            if cam_dt.size < 10:
                # Too short to judge; None rather than a false "uniform".
                grid_uniform = None
            else:
                grid_uniform = bool(uniform and camc.seq_gaps == 0)
            return {
                "camera_hz": cam_hz,
                "camera_dt_median": float(np.median(cam_dt)) if cam_dt.size else None,
                "camera_dt_k": k_hist,
                "camera_capture_gaps": {str(k): v for k, v in sorted(cap_gaps.items())},
                "camera_seq_gaps": int(camc.seq_gaps),
                "camera_grid_uniform": grid_uniform,
            }

        for role in fmt.HAND_ROLES:
            h = _cam_health(arrs[role]["cam_index"], buf.cam_by_role[role])
            hands_meta[role].update({k: v for k, v in h.items() if k != "camera_hz"})
            if role == "left":
                for k in ("camera_dt_median", "camera_dt_k", "camera_capture_gaps",
                          "camera_seq_gaps", "camera_grid_uniform"):
                    meta[k] = h[k]
            if h["camera_grid_uniform"] is False:
                _who = "LEFT (MASTER GRID)" if role == "left" else "RIGHT"
                print(
                    f"[recorder] ⚠ WARNING {_who} camera grid NON-UNIFORM: {h['camera_hz']:.2f} Hz, "
                    f"round(dt*fps) histogram {h['camera_dt_k']}, "
                    f"capture gaps {h['camera_capture_gaps']}, "
                    f"lost frames (seq gaps) {h['camera_seq_gaps']}. "
                    f"The LEFT camera timeline is the MASTER GRID of the training dataset and "
                    f"the loader indexes frames with no dt input -- a non-uniform grid "
                    f"silently corrupts every learned velocity. Fix the camera bridge "
                    f"pacing before collecting more.",
                    flush=True,
                )
        with open(os.path.join(tmp_dir, fmt.TAKE_META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        # Atomic commit: rename .tmp -> final. If a stale final dir exists (re-run),
        # remove it first so rename succeeds.
        if os.path.isdir(take_dir):
            shutil.rmtree(take_dir, ignore_errors=True)
        os.rename(tmp_dir, take_dir)

    # -- frame ingestion (loop-thread only) ---------------------------------
    def on_tracker(self, msg: dict) -> None:
        # Latch the base-station count BEFORE the recording guard: session_meta is
        # written on the FIRST start(), so a value only captured while recording
        # would always be one take too late. Cheap (a dict copy at the tracker rate
        # only while the field is present) and never touches the pose path.
        lh = msg.get("lighthouses")
        if isinstance(lh, dict):
            self._lighthouses = dict(lh)
        if not self.recording or self.buf is None:
            return
        # Role-keyed format (all modes now): {"mode":"1pt"|"2pt"|"3pt"|"5pt","t":float,
        # "trackers":{"left":{...},"right":{...},"head":{...},"left_foot":{...},...}}.
        # The PRIMARY hand is LEFT -> tracker.npy; every other role goes to its own file
        # via fmt.TRACKER_FILE_BY_ROLE (right/head/left_foot/right_foot).
        trackers = msg.get("trackers")
        if trackers is not None:
            t_outer = float(msg.get("t", time.time()))
            for role, data in trackers.items():
                pos = data.get("pos")
                quat = data.get("quat")
                if pos is None or quat is None:
                    continue
                t = float(data.get("t", t_outer))
                seq = float(data.get("seq", 0))
                stale = 1.0 if data.get("stale", False) else 0.0
                row = [t, seq, pos[0], pos[1], pos[2],
                       quat[0], quat[1], quat[2], quat[3], stale]
                if role == "left":
                    self.buf.tracker.append(row)          # PRIMARY hand
                elif role in self.buf.tracker_by_role:
                    self.buf.tracker_by_role[role].append(row)
                # An unknown role is DROPPED rather than filed anywhere: the bridge and
                # this file share fmt.TRACKER_FILE_BY_ROLE, so a role that is not in it
                # has no on-disk name to be written under.
            return
        # Legacy flat format (old right-primary logs / UMI-on-Air). Stored in the
        # primary tracker.npy verbatim; session_meta carries no `mode` so downstream
        # treats it as the historical right-primary layout.
        pos = msg.get("pos")
        quat = msg.get("quat")
        if pos is None or quat is None:
            return
        t = float(msg.get("t", time.time()))
        seq = float(msg.get("seq", len(self.buf.tracker)))
        stale = 1.0 if msg.get("stale", False) else 0.0
        # quat is WXYZ from the bridge — stored verbatim (see recording_format).
        self.buf.tracker.append([t, seq, pos[0], pos[1], pos[2],
                                 quat[0], quat[1], quat[2], quat[3], stale])

    # The per-hand handlers below are ONE implementation each, parameterized by role,
    # with a thin `*_right` entry point — so the two hands can never drift in dedup,
    # start-guard or provenance discipline. `on_gripper`/`on_camera`/`on_ft_batch`
    # keep their names and their LEFT-hand meaning, which is what every existing caller
    # (and recording/test_ft_compensated.py, recording/test_camera_timeline.py) uses.
    def on_gripper(self, msg: dict, role: str = "left") -> None:
        if not self.recording or self.buf is None:
            return
        t = float(msg.get("t", time.time()))
        self.buf.gripper_by_role[role].append([
            t,
            float(msg.get("actual", msg.get("cmd", 0.0))),
            float(msg.get("cmd", 0.0)),
            float(msg.get("open", 0.0)),
            float(msg.get("closed", 0.0)),
        ])

    def on_gripper_right(self, msg: dict) -> None:
        self.on_gripper(msg, role="right")

    def on_camera(self, msg: dict, role: str = "left") -> None:
        if not self.recording or self.buf is None:
            return
        cam = self.buf.cam_by_role[role]
        seq = int(msg.get("seq", cam.last_seq + 1))
        if seq == cam.last_seq:
            return  # dedup: bridge re-broadcasts; only keep genuinely new frames
        prev_seq = cam.last_seq
        t = float(msg.get("t", time.time()))
        # H2: the RealSense bridge re-sends its LAST frame (new seq, OLD t) on a
        # client (re)connect. Accepting it at take start plants a pre-take
        # timestamp as the master grid's first frame. Drop any frame stamped
        # before this take began.
        if t < self.buf.start_wall:
            return
        cam.last_seq = seq
        b64 = msg.get("jpeg_b64")
        if not b64:
            return
        import base64
        try:
            jpeg = base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            return
        # ===== timeline health (see CamChannel) =====
        # A `seq` gap > 1 means the bridge produced an image that never reached
        # us: the old polled broadcaster overwrote its latest-value register
        # between two polls. Silent before; counted now.
        if prev_seq >= 0 and seq - prev_seq > 1:
            cam.seq_gaps += seq - prev_seq - 1
        # `capture_seq` is the SDK frame number, so its delta is the true capture
        # decimation. A single value in this histogram == a uniform grid.
        cap = msg.get("capture_seq")
        if cap is not None:
            cap = int(cap)
            if cam.last_capture_seq >= 0:
                d = cap - cam.last_capture_seq
                cam.capture_gaps[d] = cam.capture_gaps.get(d, 0) + 1
            cam.last_capture_seq = cap
        cam.jpegs.append(jpeg)
        cam.index.append([t, float(seq)])

    def on_camera_right(self, msg: dict) -> None:
        self.on_camera(msg, role="right")

    def on_ft_batch_right(self, msg: dict) -> None:
        self.on_ft_batch(msg, role="right")

    def on_ft_batch(self, msg: dict, role: str = "left") -> None:
        if not self.recording or self.buf is None:
            return
        ch = self.buf.ft_by_role[role]
        # Latch the compensated-vs-raw provenance from the FIRST batch and assert
        # every later batch agrees (the publisher fixes it per session, so a
        # disagreement means an invariant broke — flag 'mixed' loudly rather than
        # silently mislabel safety-relevant force data). Legacy/mock frames omit
        # the key -> False (raw). See recording_format's F/T invariant.
        comp_flag = bool(msg.get("compensated", False))
        if ch.compensated is None:
            ch.compensated = comp_flag
        elif comp_flag != ch.compensated and not ch.mixed:
            ch.mixed = True
            print(f"[recorder] WARNING: {role} ft_batch 'compensated' flipped "
                  f"{ch.compensated}->{comp_flag} mid-take; marking take ft_compensated='mixed'",
                  flush=True)
        # Algorithm provenance: latch WHICH engine compensated (bota|rnea) from the
        # first batch that carries the key; a mid-take flip -> 'mixed' (loud), same
        # discipline as the compensated flag above. Only compensated batches carry it
        # (the publisher omits it on the raw-only session), so a raw take stays None.
        algo = msg.get("ft_comp_algo")
        if algo is not None:
            if ch.comp_algo is None:
                ch.comp_algo = str(algo)
            elif str(algo) != ch.comp_algo and not ch.algo_mixed:
                ch.algo_mixed = True
                print(f"[recorder] WARNING: {role} ft_batch 'ft_comp_algo' flipped "
                      f"{ch.comp_algo}->{algo} mid-take; marking take ft_comp_algo='mixed' "
                      f"(compensated wrench blends two compensation models)", flush=True)
        # Sign-verified: once any batch reports unverified, the take is unverified
        # (an inverted RNEA sign contaminates the whole compensated trace).
        if "ft_sign_verified" in msg:
            sv = bool(msg["ft_sign_verified"])
            if ch.sign_verified is None:
                ch.sign_verified = sv
            elif not sv:
                ch.sign_verified = False
        # Tare provenance (advisory): last non-null wins for the single latch, AND
        # append to the tare-event list (deduped by wall-clock) so every tare during
        # the take — the auto-tare on episode START, plus any manual/end tare whose
        # batch lands before stop detaches the buffer — is recoverable for R5 (a
        # vertical marker on the replay F/T curve).
        if msg.get("tare_wrench") is not None:
            tw = [float(x) for x in msg["tare_wrench"]]
            twall = msg.get("tare_wall")
            ch.tare_wrench = tw
            ch.tare_wall = twall
            if twall is not None and (not ch.tare_events
                                      or ch.tare_events[-1][0] != float(twall)):
                ch.tare_events.append([float(twall), *tw])
        # The F/T publisher drains a ring, so the first batch after START can carry
        # samples captured BEFORE the take began (stale ring contents). Drop those,
        # mirroring on_camera's H2 guard, so ft_wrench.npy starts at the take.
        start = self.buf.start_wall
        for s in msg.get("samples", []):
            if len(s) >= 7 and float(s[0]) >= start:
                ch.raw.append([float(s[0]), float(s[1]), float(s[2]), float(s[3]),
                               float(s[4]), float(s[5]), float(s[6])])
        # Compensated sidecar (only present when the session is compensated); same
        # start-guard so the two files share their leading timestamp.
        for s in msg.get("samples_comp", []):
            if len(s) >= 7 and float(s[0]) >= start:
                ch.comp.append([float(s[0]), float(s[1]), float(s[2]), float(s[3]),
                                float(s[4]), float(s[5]), float(s[6])])

    def status(self) -> dict:
        b = self.buf
        return {
            "type": "status",
            "recording": self.recording,
            "take_idx": self.take_idx,
            "task": b.task if b else "",
            "frames": {
                "tracker": len(b.tracker) if b else 0,          # PRIMARY = LEFT
                **{f"tracker_{role}": (len(b.tracker_by_role[role]) if b else 0)
                   for role in fmt.TRACKER_FILE_BY_ROLE if role != "left"},
                "gripper": len(b.gripper) if b else 0,
                "ft": len(b.ft.raw) if b else 0,
                "camera": len(b.cam.index) if b else 0,
                # RIGHT-hand per-hand modalities. ALWAYS present, 0 when that device is
                # absent, so the viz status panel never has to branch on hardware.
                "gripper_right": len(b.gripper_by_role["right"]) if b else 0,
                "ft_right": len(b.ft_by_role["right"].raw) if b else 0,
                "camera_right": len(b.cam_by_role["right"].index) if b else 0,
            },
            "markers": len(b.markers) if b else 0,
            "elapsed": (time.time() - b.start_wall) if (b and self.recording) else 0.0,
            "last_saved": self.last_saved,
            # The session root, so a GUI can write calibration artifacts (e.g. the
            # tracker mount rotation) WITH the take instead of holding them in RAM.
            # None until the first start() creates the session.
            "session_dir": self._session_dir,
            "home_seq": self.home_seq,   # 'c'-key return-to-home relay (viz observes)
        }


def _ts_str(t: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(t).strftime("%Y.%m.%d_%H.%M.%S")


# ---------------------------------------------------------------------------
# WS plumbing (subscribe to bridges + serve the control socket)
# ---------------------------------------------------------------------------
async def _subscribe(url: str, on_message, name: str, stop: asyncio.Event) -> None:
    import websockets
    while not stop.is_set():
        try:
            async with websockets.connect(url, max_size=None) as ws:
                print(f"[recorder] connected {name} {url}", flush=True)
                async for raw in ws:
                    try:
                        on_message(json.loads(raw))
                    except (ValueError, TypeError):
                        continue
        except Exception:  # noqa: BLE001 - bridge down / not up yet
            pass
        if not stop.is_set():
            await asyncio.sleep(1.0)


async def _serve_control(rec: Recorder, host: str, port: int, stop: asyncio.Event) -> None:
    import websockets
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                cmd = msg.get("cmd")
                if cmd == "start":
                    rec.start(task=str(msg.get("task", "")))
                elif cmd == "stop":
                    await rec.stop_async()
                elif cmd == "marker":
                    rec.mark()
                elif cmd == "home":
                    rec.home()
                elif cmd == "discard":
                    rec.discard_last()
        except Exception:  # noqa: BLE001
            pass
        finally:
            clients.discard(ws)

    async with websockets.serve(handler, host, port):
        print(f"[recorder] control WS at ws://{host}:{port}", flush=True)
        while not stop.is_set():
            if clients:
                websockets.broadcast(clients, json.dumps(rec.status()))
            await asyncio.sleep(0.2)


def _serial_role_map_from_args(args) -> dict:
    """Build the canonical serial->role map for this session from the mode +
    per-role serial args (LEFT-canonical: 'left' is the primary hand, present in
    every mode). Empty entries are skipped so the map only records assigned roles."""
    roles = fmt.TRACKER_ROLES_BY_MODE.get(args.mode, ("left",))
    serial_for = {"left": args.left_serial, "right": args.right_serial,
                  "head": args.head_serial, "left_foot": args.left_foot_serial,
                  "right_foot": args.right_foot_serial}
    return {serial_for[r]: r for r in roles if serial_for.get(r)}


def _world_align_from_args(args) -> dict:
    """Build the world-alignment provenance block from the launch args.

    Quaternion form wins (it is what the bridge applies verbatim); the rpy form is
    recorded as given so the operator's intent survives alongside it. Returns ``{}``
    when neither was supplied — "unknown", which must NOT be written as identity."""
    quat = getattr(args, "world_quat", None)
    rpy = getattr(args, "world_rpy_deg", None)
    if quat:
        return {"quat_wxyz": [float(v) for v in quat], "source": "world_quat"}
    if rpy:
        return {"rpy_deg": [float(v) for v in rpy], "source": "world_rpy_deg"}
    return {}


async def _run(args) -> int:
    import signal
    rec = Recorder(args.out, session_name=args.session_name,
                   mode=args.mode, serial_role_map=_serial_role_map_from_args(args),
                   world_align=_world_align_from_args(args),
                   camera_fps_nominal=args.camera_fps)
    stop = asyncio.Event()
    # H1: flush any in-progress take on shutdown. The launcher tears the rig down
    # with SIGTERM (kill); asyncio.run only maps SIGINT->KeyboardInterrupt, so
    # without an explicit SIGTERM handler a live take is lost. Both signals just
    # set the stop Event; the finally below flushes.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, ValueError):  # non-main thread / unsupported
            pass
    tasks = [
        asyncio.create_task(_subscribe(args.tracker_url, rec.on_tracker, "tracker", stop)),
        asyncio.create_task(_subscribe(args.camera_url, rec.on_camera, "camera", stop)),
        asyncio.create_task(_subscribe(args.gripper_url, rec.on_gripper, "gripper", stop)),
        asyncio.create_task(_subscribe(args.ft_url, rec.on_ft_batch, "ft", stop)),
        asyncio.create_task(_serve_control(rec, args.host, args.ctrl_port, stop)),
    ]
    # RIGHT-hand per-hand modalities. Subscribed only when a URL was given, so a 1pt
    # launch (or a 3pt launch before the right-hand devices are wired up) opens no extra
    # sockets at all. `_subscribe` already retries a down bridge forever without
    # logging, so an unconnected right hand is silent rather than noisy — and the take
    # simply carries 0 right-hand frames (see recording_format's PER-HAND banner).
    right_streams = [
        (getattr(args, "camera_right_url", None), rec.on_camera_right, "camera_right"),
        (getattr(args, "gripper_right_url", None), rec.on_gripper_right, "gripper_right"),
        (getattr(args, "ft_right_url", None), rec.on_ft_batch_right, "ft_right"),
    ]
    for url, handler, name in right_streams:
        if url:
            print(f"[recorder] RIGHT hand: subscribing {name} {url}", flush=True)
            tasks.append(asyncio.create_task(_subscribe(url, handler, name, stop)))
    stop_task = asyncio.create_task(stop.wait())
    try:
        # Run until a signal sets stop; then cancel the subscribers/server.
        await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        # Flush a live take so it isn't lost on teardown (loop-thread call).
        if rec.recording:
            print("[recorder] shutdown: flushing in-progress take", flush=True)
            await rec.flush_on_shutdown()
    return 0


# ---------------------------------------------------------------------------
# Self-contained mock: synthesize all 4 streams + a scripted CONTINUOUS take
# with 2 episode-split markers (-> 3 episodes), no WS. Produces a real on-disk
# session so record->convert (split-at-markers) can be smoke-tested.
# ---------------------------------------------------------------------------
def _run_mock(args) -> int:
    # In 1pt the mock synthesizes a single flat (legacy) tracker stream; the on_tracker
    # legacy path files it into tracker.npy. In 2pt/3pt it emits the ROLE-KEYED frame
    # the real bridge sends (left + right [+ head]) AND — for whichever right-hand
    # modality URLs were requested — a synthetic right camera / F-T / gripper. That is
    # what lets `--mock --mode 3pt` exercise the dual-hand write path with no hardware:
    # the right-hand devices are not connected yet, so the mock IS the only way to
    # regression-test the take layout they will produce.
    mode = getattr(args, "mode", "1pt")
    rec = Recorder(args.out, session_name=args.session_name or "session_mock",
                   mode=mode,
                   serial_role_map=_serial_role_map_from_args(args),
                   world_align=_world_align_from_args(args),
                   camera_fps_nominal=getattr(args, "camera_fps", 30.0))
    tracker_roles = fmt.TRACKER_ROLES_BY_MODE.get(mode, ("left",))
    role_keyed = len(tracker_roles) > 1
    # Right-hand modalities are synthesized only when the caller asked for that URL —
    # the same "a URL asserts the device exists" rule the live path uses.
    mock_right = {
        "camera": bool(getattr(args, "camera_right_url", "")),
        "ft": bool(getattr(args, "ft_right_url", "")),
        "gripper": bool(getattr(args, "gripper_right_url", "")),
    }
    if role_keyed:
        print(f"[recorder] mock mode={mode} roles={list(tracker_roles)} "
              f"right-hand={[k for k, v in mock_right.items() if v] or 'none'}", flush=True)
    rec.start(task=args.mock_task)
    t0 = time.time()
    dur = args.mock_seconds
    # Drop 2 markers (at 1/3 and 2/3) so the converter splits into 3 episodes.
    marker_at = [dur / 3.0, 2.0 * dur / 3.0]
    next_marker = 0
    # Native-ish rates: tracker 90, gripper 30, ft 250, camera 15.
    # MEASURED rates from the real rig (97 takes, session_2026.07.25), not the
    # nominal datasheet numbers: tracker is 77.7 Hz not 90, gripper 28.9 not 30,
    # F/T 178.9 not 250. The mock existing with wrong rates is how a 12 Hz camera
    # went unnoticed in every rehearsal.
    rates = {"tracker": 77.7, "gripper": 28.9, "ft": 178.9, "camera": 15.0}
    next_t = {k: 0.0 for k in rates}
    cam_seq = 0
    n = 0
    while True:
        now = time.time() - t0
        if now >= dur:
            break
        wall = t0 + now
        if next_marker < len(marker_at) and now >= marker_at[next_marker]:
            rec.mark()
            next_marker += 1
        if now >= next_t["tracker"]:
            next_t["tracker"] += 1.0 / rates["tracker"]
            # small circle + slow yaw
            x, y, z = 0.4 + 0.08 * math.sin(0.5 * now), -0.2 + 0.08 * math.cos(0.5 * now), 0.8 + 0.05 * math.sin(now)
            ang = 0.3 * math.sin(0.7 * now)
            quat = [math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
            if role_keyed:
                # Role-keyed frame, each role on its OWN offset so a left/right mix-up
                # downstream is visible instead of two identical traces. The chest
                # ('head') is a slow body reference near the torso, not a hand orbit.
                # One DISTINCT offset per role so a role that lands in the wrong file is
                # visible as a wrong trace rather than as two identical ones. The feet sit
                # BELOW and in front (the operator's own stance), which also makes a
                # left/right foot swap read as a mirrored pair.
                offs = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.40, 0.0),
                        "head": (-0.30, 0.20, 0.35),
                        "left_foot": (-0.20, 0.10, -0.75), "right_foot": (-0.20, 0.30, -0.75)}
                rec.on_tracker({
                    "mode": mode, "t": wall,
                    "trackers": {
                        role: {"t": wall, "seq": n, "stale": False,
                               "pos": [x + offs[role][0], y + offs[role][1], z + offs[role][2]],
                               "quat": quat}
                        for role in tracker_roles
                    },
                })
            else:
                rec.on_tracker({"t": wall, "seq": n, "pos": [x, y, z],
                                "quat": quat, "stale": False})
        if now >= next_t["gripper"]:
            next_t["gripper"] += 1.0 / rates["gripper"]
            frac = 0.5 * (1 + math.sin(0.8 * now))
            rec.on_gripper({"t": wall, "actual": frac * 4.90, "cmd": frac * 4.90,
                            "open": 4.90, "closed": 0.0})
            if mock_right["gripper"]:
                # Antiphase, so a left/right swap in the take is obvious on a plot.
                rfrac = 0.5 * (1 - math.sin(0.8 * now))
                rec.on_gripper_right({"t": wall, "actual": rfrac * 4.90, "cmd": rfrac * 4.90,
                                      "open": 4.90, "closed": 0.0})
        if now >= next_t["ft"]:
            next_t["ft"] += 1.0 / rates["ft"]
            rec.on_ft_batch({"samples": [[wall,
                                          math.sin(2 * now), math.cos(2 * now), 1.0 + 0.1 * now % 1,
                                          0.01 * math.sin(now), 0.02 * math.cos(now), 0.0]]})
            if mock_right["ft"]:
                rec.on_ft_batch_right({"samples": [[wall,
                                                    -math.sin(2 * now), -math.cos(2 * now), 2.0,
                                                    0.0, 0.0, 0.03 * math.sin(now)]]})
        if now >= next_t["camera"]:
            next_t["camera"] += 1.0 / rates["camera"]
            jpeg = _synth_jpeg(now)
            import base64
            rec.on_camera({"t": wall, "seq": cam_seq,
                           # capture_seq = SDK frame number; at 30 fps capture and
                           # a 15 Hz publish rate it advances by exactly 2, which
                           # is what the uniformity check buckets on.
                           "capture_seq": cam_seq * 2,
                           "w": 64, "h": 48,
                           "jpeg_b64": base64.b64encode(jpeg).decode("ascii")})
            if mock_right["camera"]:
                # A DIFFERENT image (phase-shifted) so the two camera dirs can never be
                # confused for one stream written twice.
                rjpeg = _synth_jpeg(now + 3.7)
                rec.on_camera_right({"t": wall, "seq": cam_seq, "capture_seq": cam_seq * 2,
                                     "w": 64, "h": 48,
                                     "jpeg_b64": base64.b64encode(rjpeg).decode("ascii")})
            cam_seq += 1
        n += 1
        time.sleep(0.0005)
    take = rec.stop()
    # Seed calibration files so the converter has a complete (if placeholder) session.
    _seed_mock_calibration(rec._session_dir)
    print(f"[recorder] mock session at {rec._session_dir} (take: {take}, "
          f"{len(marker_at)} markers -> {len(marker_at) + 1} episodes)", flush=True)
    return 0


def _synth_jpeg(t: float) -> bytes:
    from PIL import Image
    import io
    h, w = 48, 64
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (127 * (1 + np.sin(2 * math.pi * (xx + 0.1 * t)))).astype(np.uint8)
    g = (127 * (1 + np.sin(2 * math.pi * (yy + 0.13 * t)))).astype(np.uint8)
    b = np.full((h, w), int(80 + 80 * math.sin(t)), dtype=np.uint8)
    bio = io.BytesIO()
    Image.fromarray(np.stack([r, g, b], 2), "RGB").save(bio, "JPEG", quality=70)
    return bio.getvalue()


def _seed_mock_calibration(session_dir: str) -> None:
    calib = os.path.join(session_dir, fmt.CALIB_DIR)
    os.makedirs(calib, exist_ok=True)
    gr = os.path.join(calib, fmt.GRIPPER_RANGE_FILE)
    if not os.path.exists(gr):
        with open(gr, "w") as f:
            json.dump({"enc_closed": 0.0, "enc_open": 4.90,
                       "width_closed_m": 0.0, "width_open_m": 0.085,
                       "_note": "PLACEHOLDER mock calibration; measure for a real rig"}, f, indent=2)
    tx = os.path.join(calib, fmt.TCP_EXTRINSIC_FILE)
    if not os.path.exists(tx):
        with open(tx, "w") as f:
            json.dump({"T_tracker_tcp": np.eye(4).tolist(),
                       "calibrated": False,
                       "_note": "IDENTITY PLACEHOLDER -> UNCALIBRATED_DO_NOT_TRAIN"}, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/recordings", help="session root dir")
    p.add_argument("--session-name", default=None, help="fixed session dir name (else timestamped)")
    # Tracking mode + LEFT-canonical serial->role provenance, recorded in
    # session_meta/take_meta. The launcher forwards these; the primary hand is left.
    p.add_argument("--mode", choices=tuple(fmt.TRACKER_ROLES_BY_MODE), default="1pt",
                   help="tracking mode recorded in meta: 1pt=left only, 2pt=+right, "
                        "3pt=+head, 5pt=+both feet")
    p.add_argument("--left-serial", default=None, help="LEFT (primary) hand tracker serial (provenance)")
    p.add_argument("--right-serial", default=None, help="RIGHT hand tracker serial (2pt/3pt; provenance)")
    p.add_argument("--head-serial", default=None, help="HEAD/torso tracker serial (3pt/5pt; provenance)")
    p.add_argument("--left-foot-serial", default=None, help="LEFT-foot tracker serial (5pt; provenance)")
    p.add_argument("--right-foot-serial", default=None, help="RIGHT-foot tracker serial (5pt; provenance)")
    p.add_argument("--world-rpy-deg", type=float, nargs=3, default=None,
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="PROVENANCE ONLY (this process never transforms a pose): the "
                        "libsurvive-world -> robot-world alignment the TRACKER BRIDGE "
                        "was launched with. Recorded into session_meta.world_align so "
                        "the frame every pose sits in is reconstructible. Pass the same "
                        "values the bridge got.")
    p.add_argument("--world-quat", type=float, nargs=4, default=None,
                   metavar=("W", "X", "Y", "Z"),
                   help="PROVENANCE ONLY: same as --world-rpy-deg, as a wxyz quaternion "
                        "(what the bridge's --world-quat received). Wins over the rpy form.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--ctrl-port", type=int, default=fmt.RECORDER_CTRL_PORT_DEFAULT)
    p.add_argument("--tracker-url", default="ws://127.0.0.1:8772")
    p.add_argument("--camera-url", default=f"ws://127.0.0.1:{fmt.CAMERA_WS_PORT_DEFAULT}")
    p.add_argument("--camera-fps", type=float, default=30.0,
                   help="nominal camera CAPTURE fps (not the publish rate); only used to "
                        "bucket the recorded camera dt for the uniformity check")
    p.add_argument("--gripper-url", default="ws://127.0.0.1:8774")
    p.add_argument("--ft-url", default=f"ws://127.0.0.1:{fmt.FT_WS_PORT_DEFAULT}")
    # RIGHT-hand per-hand modalities. Default EMPTY (not a port) on purpose: passing a
    # URL is the launcher's way of asserting "this device exists", so a 1pt session and
    # a 3pt session whose right-hand hardware is not wired up yet both open zero extra
    # sockets and record zero right-hand frames — no phantom stream, no zero-fill.
    # The right TRACKER needs none of these (it rides the single role-keyed :8772 frame).
    p.add_argument("--camera-right-url", default="",
                   help=f"RIGHT-hand D405 bridge (e.g. ws://127.0.0.1:{fmt.CAMERA_RIGHT_WS_PORT_DEFAULT}); "
                        "empty = no right-hand camera")
    p.add_argument("--gripper-right-url", default="",
                   help=f"RIGHT-hand gripper bridge (e.g. ws://127.0.0.1:{fmt.GRIPPER_RIGHT_WS_PORT_DEFAULT}); "
                        "empty = no right-hand gripper")
    p.add_argument("--ft-right-url", default="",
                   help=f"RIGHT-hand F/T publisher (e.g. ws://127.0.0.1:{fmt.FT_RIGHT_WS_PORT_DEFAULT}); "
                        "empty = no right-hand F/T")
    p.add_argument("--mock", action="store_true", help="self-contained synthetic session (no WS), then exit")
    p.add_argument("--mock-seconds", type=float, default=4.0)
    p.add_argument("--mock-task", default="mock_pick")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # DECISION-B reminder, printed once at startup so the wxyz convention is never
    # silently assumed downstream.
    print(f"[recorder] raw tracker quaternions stored as {fmt.QUATERNION_CONVENTION.upper()} "
          f"(converted to xyzw ONLY in processing/convert_to_umi.py)", flush=True)

    if args.mock:
        return _run_mock(args)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[recorder] bye", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
