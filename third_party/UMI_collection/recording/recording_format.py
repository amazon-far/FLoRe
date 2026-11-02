#!/usr/bin/env python3
"""Shared definitions of the raw on-disk recording layout + wire frames.

Imported by BOTH the live recorder (``recording/recorder.py``) and the offline
converter (``processing/convert_to_umi.py``). As of DECISION-I both run in the
SAME ``.venv`` (collection + conversion are one maintained environment), so keep
this module dependency-light (stdlib + numpy) — it must import in the viz/record
context too.

CONTINUOUS-TAKE model (DECISION-E)
----------------------------------
Recording is CONTINUOUS: one Start..Stop span is a single "take", recorded
unbroken at each stream's native rate. The operator does NOT gesture to begin or
end each demonstration. Instead they drop an episode-split MARKER at the boundary
between demonstrations (long-press ``o+p`` 1.5 s on the macropad, or the GUI
"Mark boundary" button). Markers are recorded as bare wall-clock timestamps; the
OFFLINE converter splits each take into N+1 UMI episodes at the N marker times.
This keeps capture dumb + lossless and pushes all segmentation to post-processing
(the project's "record raw, post-process" principle).

Raw layout (one directory per session; one ``take_NNN`` per Start..Stop span; each
modality its OWN file at its OWN native rate, with a per-sample timestamp column —
NO alignment and NO episode splitting at record time, those are the converter's
job):

    <session_dir>/                       e.g. data/recordings/session_2026.06.20_23.10.00
      session_meta.json                  rig + clock + calibration provenance
      calibration/
        gripper_range.json               encoder->width 2-point calib (may be absent -> soft warn)
        tcp_extrinsic.json               tracker->TCP 4x4 (absent/identity -> UNCALIBRATED)
      take_000/
        tracker.npy      float64 [N,10]  t, seq, x,y,z, qw,qx,qy,qz, stale  (RAW world;
                                         the PRIMARY hand = LEFT — see LEFT-canonical note)
        tracker_right.npy float64 [N,10] the RIGHT-hand tracker (2pt/3pt; absent in 1pt)
        tracker_head.npy float64 [N,10]  the HEAD/CHEST tracker (3pt/5pt; absent otherwise)
        tracker_left_foot.npy  float64 [N,10] the LEFT-foot tracker  (5pt only)
        tracker_right_foot.npy float64 [N,10] the RIGHT-foot tracker (5pt only)
        gripper.npy      float64 [N,5]   t, encoder_rad, cmd_rad, open_rad, closed_rad  (LEFT)
        gripper_right.npy float64 [N,5]  the RIGHT hand's gripper (see PER-HAND note)
        ft_wrench.npy    float64 [N,7]   t, Fx,Fy,Fz, Tx,Ty,Tz   (ALWAYS the RAW sensor wrench; LEFT)
        ft_wrench_comp.npy float64 [Nc,7] t, Fx,Fy,Fz, Tx,Ty,Tz (gravity/inertia-COMPENSATED;
                                         present ONLY when a Bota payload config was active —
                                         absent for raw-only sessions; see F/T note below)
        ft_wrench_right.npy      float64 [N,7]  the RIGHT hand's RAW wrench
        ft_wrench_comp_right.npy float64 [Nc,7] the RIGHT hand's COMPENSATED wrench
        markers.npy      float64 [M,1]   t   (episode-split wall-clock times; M may be 0)
        camera/                          the LEFT hand's D405
          frame_000000.jpg ...           D405 JPEG bytes verbatim from the bridge (no re-encode)
          frame_index.npy  float64 [N,2] t, seq   (row i <-> frame_%06d.jpg)
        camera_right/                    the RIGHT hand's D405, SAME layout
          frame_000000.jpg ...
          frame_index.npy  float64 [N,2]
        take_meta.json                   task, start/stop wall-clock, per-stream counts, markers
      take_001/ ...

!!! PER-HAND MODALITIES — EVERY RIGHT-HAND FILE IS OPTIONAL !!!
    The rig's 3pt configuration is: LEFT hand = tracker + camera + F/T + gripper,
    RIGHT hand = tracker + camera + F/T + gripper, CHEST = tracker ONLY (no camera,
    no F/T, no gripper — it is a body reference). 5pt adds TWO MORE body references,
    the two FEET — same statement: pucks, and nothing else (:data:`BODY_ROLES`).

    The LEFT hand is the PRIMARY and its files are the historical, unsuffixed ones.
    Every RIGHT-hand file carries a ``_right`` suffix and is written ONLY when that
    device actually streamed during the take. So:
      * a 1pt session is byte-identical to before (no ``*_right`` file exists);
      * a 3pt session recorded before the right-hand camera/Bota/gripper are wired up
        still produces a valid take — it simply has ``tracker_right.npy`` and no
        ``camera_right/``, and the converter emits no ``camera1_rgb``.
    ABSENCE IS A FACT, NOT AN ERROR: ``take_meta.json["counts"]`` carries a 0 for each
    right-hand stream and ``take_meta.json["hands"]`` records which modalities each hand
    actually contributed, so downstream never has to infer capability from which files
    happen to exist. Do NOT zero-fill a missing right-hand stream at record time — that
    would make "the sensor was unplugged" indistinguishable from "the sensor read zero".

    NOTE the two different "right"s: ``tracker_right.npy`` is the right-hand TRACKER
    (present whenever MODE is 2pt/3pt/5pt, since it only needs a puck), while
    ``camera_right/`` / ``ft_wrench_right.npy`` / ``gripper_right.npy`` need real
    right-hand HARDWARE. A 3pt take with a right tracker but no right camera is normal
    and expected during bring-up.

!!! LEFT-CANONICAL PRIMARY HAND — DON'T ASSUME 'RIGHT' DOWNSTREAM !!!
    The PRIMARY hand is the LEFT hand in EVERY mode (1pt/2pt/3pt/5pt). The primary
    tracker's samples go to ``tracker.npy`` (so the converter's ``robot0_*`` and
    the raw-replay main arm are the LEFT hand); the right hand (2pt/3pt/5pt) goes to
    ``tracker_right.npy``, the head/torso (3pt/5pt) to ``tracker_head.npy`` and the
    two feet (5pt) to ``tracker_{left,right}_foot.npy``. The
    canonical serial->role map + the mode are recorded in
    ``session_meta.json`` (``mode`` + ``serial_role_map`` + ``primary_role``) and
    in each ``take_meta.json`` (``mode``), so downstream never has to infer the
    tracker count from which files happen to exist. (Historical takes recorded
    before this change stored the right hand in ``tracker.npy``; they carry no
    ``mode`` key and are treated as the legacy right-primary layout.)

!!! QUATERNION CONVENTION (DECISION-B) — DO NOT FORGET DOWNSTREAM !!!
    The tracker columns ``qw,qx,qy,qz`` are stored **WXYZ** (libsurvive's order,
    as emitted by ``bridges/tracker/vive_tracker_bridge.py``). This is the RAW,
    most-faithful capture. scipy / UMI use **XYZW**. The conversion to XYZW
    happens at EXACTLY ONE place: ``umi_convert_core.quat_wxyz_to_xyzw`` (called
    by the converter). Anything new that reads ``tracker.npy`` directly MUST
    treat columns [5:9] as WXYZ. ``session_meta.json["quaternion_convention"]``
    records this on every session, and both the recorder and converter PRINT a
    one-line wxyz reminder at runtime.

!!! F/T WRENCH PROVENANCE (record-both) — DON'T ASSUME RAW DOWNSTREAM !!!
    ``ft_wrench.npy`` is ALWAYS the RAW Bota wrench (tool gravity + inertia still
    in it). When a Bota payload config is active at record time the recorder ALSO
    writes ``ft_wrench_comp.npy`` — the gravity/inertia-COMPENSATED wrench at the
    same wall-clock timestamps, captured live (the offline pipeline can NOT
    reconstruct it: compensation needs the sensor's IMU orientation per sample,
    which is not stored). Both files are in the SAME sensor frame + sign
    convention — only the magnitudes differ — so any resample/rotate of one is
    valid on the other. Downstream PREFERS the compensated sidecar when present
    (replay display, ``robot0_ft_wrench`` in the zarr) and falls back to raw; raw
    is kept as the irreversible audit/safety net (a contaminated startup tare, a
    mis-estimated payload, or a compensator bug can all be caught/diagnosed
    against it). ``take_meta.json["ft_compensated"]`` records whether the sidecar
    was written for that take (older takes lack the key -> treated as raw).

    INVARIANT: within ONE take every ``ft_batch`` carries the SAME ``compensated``
    flag, because the viz selects raw-vs-compensated ONCE per session (a fixed
    payload config) and recompensation only runs inside the already-compensated
    loop. The recorder LATCHES the flag at the first F/T sample and flags the take
    ``"mixed"`` (loud warning) if a later batch disagrees, so a provenance
    conflict can never be silently mislabeled.

All timestamps are wall-clock ``time.time()`` seconds (ONE clock for the whole
rig). The recorder stamps each sample at the bridge-provided capture time ``t``
when present (closest to the physical event; best for sync), falling back to
receive time; F/T is stamped at the in-process sensor read. The converter makes
each UMI episode's ``episode_timestamps`` RELATIVE (first frame == 0, DECISION-A);
the absolute wall-clock is preserved here in the raw files for cross-checking.
"""
from __future__ import annotations

# Column layouts (indices into the per-stream .npy arrays). Single source of
# truth shared by recorder + converter so the two never drift.
TRACKER_COLS = ("t", "seq", "x", "y", "z", "qw", "qx", "qy", "qz", "stale")
GRIPPER_COLS = ("t", "encoder_rad", "cmd_rad", "open_rad", "closed_rad")
FT_COLS = ("t", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
CAMERA_INDEX_COLS = ("t", "seq")
MARKER_COLS = ("t",)  # episode-split wall-clock times (DECISION-E)

TRACKER_FILE = "tracker.npy"            # PRIMARY hand = LEFT (see LEFT-canonical banner)
TRACKER_RIGHT_FILE = "tracker_right.npy"  # RIGHT hand (2pt/3pt); absent in 1pt
TRACKER_HEAD_FILE = "tracker_head.npy"    # HEAD/torso (3pt only); absent otherwise
# ---- 5pt FEET (see the BODY-REFERENCE banner below) --------------------------
# The 4th and 5th keypoints, added 2026-09-09. Same ten columns, same wire, same
# absence rule as the chest: a tracker-only body reference with no camera, F/T or
# gripper. Named for the ROLE the low level consumes ('left_foot'/'right_foot' are
# FLoRe's own `layout.KEYPOINT_ORDER` / `display.DISPLAY_HOME_POS` keys), NOT for
# the puck, so a dataset that later carries REAL foot poses lands on the same names
# the 5pt stack already reads.
TRACKER_LEFT_FOOT_FILE = "tracker_left_foot.npy"    # LEFT foot/ankle (5pt only)
TRACKER_RIGHT_FOOT_FILE = "tracker_right_foot.npy"  # RIGHT foot/ankle (5pt only)
#: role -> on-disk tracker file. The primary role ('left') is the main tracker.npy.
TRACKER_FILE_BY_ROLE = {
    "left": TRACKER_FILE,
    "right": TRACKER_RIGHT_FILE,
    "head": TRACKER_HEAD_FILE,
    "left_foot": TRACKER_LEFT_FOOT_FILE,
    "right_foot": TRACKER_RIGHT_FOOT_FILE,
}
GRIPPER_FILE = "gripper.npy"
FT_FILE = "ft_wrench.npy"           # ALWAYS raw (see F/T provenance banner above)
FT_COMP_FILE = "ft_wrench_comp.npy"  # gravity/inertia-compensated; written only when available
MARKER_FILE = "markers.npy"
TARE_EVENTS_FILE = "tare_events.npy"  # [K,7] wall, Fx..Tz — one row per F/T tare (R5)
TARE_EVENT_COLS = ("t", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
CAMERA_DIR = "camera"
CAMERA_INDEX_FILE = "frame_index.npy"

# ---- RIGHT-hand per-hand modalities (see the PER-HAND banner above) ----------
# Written ONLY when the right-hand device streamed during the take. Same column
# layouts / same directory structure as the LEFT (primary) files, so every reader is
# the same code with a different path.
GRIPPER_RIGHT_FILE = "gripper_right.npy"
FT_RIGHT_FILE = "ft_wrench_right.npy"
FT_COMP_RIGHT_FILE = "ft_wrench_comp_right.npy"
TARE_EVENTS_RIGHT_FILE = "tare_events_right.npy"
CAMERA_RIGHT_DIR = "camera_right"

#: hand role -> its per-modality on-disk names. The ONE place the left/right file
#: naming is decided; recorder, converter and both replay viewers all index this, so a
#: new per-hand modality cannot be added to one side only.
#: 'head' (the CHEST tracker) is deliberately ABSENT: it has no camera / F-T / gripper.
HAND_FILES_BY_ROLE = {
    "left": {
        "tracker": TRACKER_FILE,
        "gripper": GRIPPER_FILE,
        "ft": FT_FILE,
        "ft_comp": FT_COMP_FILE,
        "tare_events": TARE_EVENTS_FILE,
        "camera_dir": CAMERA_DIR,
    },
    "right": {
        "tracker": TRACKER_RIGHT_FILE,
        "gripper": GRIPPER_RIGHT_FILE,
        "ft": FT_RIGHT_FILE,
        "ft_comp": FT_COMP_RIGHT_FILE,
        "tare_events": TARE_EVENTS_RIGHT_FILE,
        "camera_dir": CAMERA_RIGHT_DIR,
    },
}

#: Which tracker roles each MODE brings up (LEFT-canonical: 'left' is always present).
#: The single source of truth for "how many pucks", shared by the launcher's role list,
#: the recorder's expectations and the replay viewers' file discovery.
TRACKER_ROLES_BY_MODE = {
    "1pt": ("left",),
    "2pt": ("left", "right"),
    "3pt": ("left", "right", "head"),
    "5pt": ("left", "right", "head", "left_foot", "right_foot"),
}

#: Roles that can carry a camera / F-T / gripper. The CHEST and FOOT trackers cannot —
#: they are body references, so 3pt/5pt add PUCKS but not further sets of sensors.
HAND_ROLES = ("left", "right")

#: BODY-REFERENCE roles: a puck and nothing else. The chest (3pt) plus the two feet
#: (5pt). They share EVERY property that distinguishes the chest from a hand — no
#: camera / F-T / gripper, no TCP lever, no gripper meshes, a triad AT the frame origin,
#: and a rotation-only puck->body relabel — which is why the code branches on this
#: tuple rather than on the literal string ``"head"``: a fifth keypoint added as
#: "another head" is one forgotten ``elif`` away from being silently dropped.
BODY_ROLES = ("head", "left_foot", "right_foot")

#: The two FEET on their own, for the places that treat them as a pair.
FOOT_ROLES = ("left_foot", "right_foot")


def hand_roles_for_mode(mode: str) -> tuple[str, ...]:
    """The hand roles (camera/F-T/gripper capable) active in ``mode``.

    1pt -> ('left',); 2pt/3pt/5pt -> ('left','right'). No body reference is ever
    returned: the chest and foot pucks have no per-hand modality (see the PER-HAND
    banner)."""
    roles = TRACKER_ROLES_BY_MODE.get(mode, ("left",))
    return tuple(r for r in roles if r in HAND_ROLES)


def body_roles_for_mode(mode: str) -> tuple[str, ...]:
    """The BODY-REFERENCE roles active in ``mode`` (see :data:`BODY_ROLES`).

    1pt/2pt -> (); 3pt -> ('head',); 5pt -> ('head','left_foot','right_foot')."""
    roles = TRACKER_ROLES_BY_MODE.get(mode, ("left",))
    return tuple(r for r in roles if r in BODY_ROLES)
TAKE_META_FILE = "take_meta.json"
SESSION_META_FILE = "session_meta.json"
CALIB_DIR = "calibration"
GRIPPER_RANGE_FILE = "gripper_range.json"
TCP_EXTRINSIC_FILE = "tcp_extrinsic.json"
#: CHEST (3pt 'head' keypoint) puck->chest-BODY transform, the head's analogue of
#: TCP_EXTRINSIC_FILE. Rotation-only (the chest has no TCP lever), so it moves
#: ``head_rot_axis_angle`` and leaves ``head_pos`` bit-identical. Absent = the raw
#: chest-puck axes are stored, which is what every pre-2026-08-26 3pt dataset carries.
HEAD_EXTRINSIC_FILE = "head_extrinsic.json"
#: The two FEET's puck->body relabels, same shape and same rotation-only rule as the
#: chest's. ONE file for both feet (``{"T_tracker_left_foot": 4x4, ...}``) because the
#: two pucks are mounted by the same jig on the same pass — and it is DELIBERATELY a
#: separate file from the chest's rather than a fourth key in it, so an existing 3pt
#: session's calibration bytes never change. Absent = each foot puck's RAW axes are
#: stored, which is a fact about the session and not a default worth hiding.
FOOT_EXTRINSIC_FILE = "foot_extrinsic.json"
#: Per-role puck->wrist MOUNT rotation captured by the viz's "Reset tracker axes to
#: wrist" button, as {role: [w,x,y,z]}. Written into <session>/calibration/ so it lives
#: WITH the take: a mount rotation held only in RAM dies on restart, and a session
#: recorded under a different mount would be indistinguishable from one recorded under
#: this one. Absent = no capture was made (treated as identity everywhere).
TRACKER_MOUNT_FILE = "tracker_mount.json"

# The recorder writes raw tracker quaternions in this order; the converter (and
# anything else reading tracker.npy) must honor it. See the module banner.
QUATERNION_CONVENTION = "wxyz"


def take_dir_name(i: int) -> str:
    return f"take_{i:03d}"


def camera_frame_name(i: int) -> str:
    return f"frame_{i:06d}.jpg"


# ---- F/T full-rate publisher wire frame (localhost WS the viz exposes) ----
# The viz's in-process Bota SensorReader appends every read to a bounded ring;
# a tiny WS publisher drains it and broadcasts BATCHES of samples so the 250 Hz
# stream reaches the out-of-process recorder LOSSLESSLY (a 1-sample-per-message
# scheme can't keep up; batching does). Frame:
#   {"type": "ft_batch",
#    "compensated": bool,                            # is samples_comp present + meaningful?
#    "samples":      [[t, Fx,Fy,Fz, Tx,Ty,Tz], ...], # RAW wrench (always)
#    "samples_comp": [[t, Fx,Fy,Fz, Tx,Ty,Tz], ...], # COMPENSATED wrench (only if compensated)
#    "ft_comp_algo":    "bota"|"rnea",               # which engine compensated (comp sessions)
#    "ft_sign_verified": bool,                       # was the engine's global sign verified?
#    "tare_wrench":     [6] | null,                  # compensator DC offset (provenance)
#    "tare_wall":       float | null}                # wall-clock of that tare
# "compensated" is FIXED for a session (see the F/T provenance banner's invariant);
# the recorder latches it per take. Legacy/mock frames omit these keys -> treated
# as raw / unverified. ``tare_wrench``/``tare_wall`` update on each tare (startup,
# recompensation, and the auto-tares on episode start/end); the recorder latches
# the last non-null per take.
FT_WS_PORT_DEFAULT = 8775
# The RIGHT hand gets its OWN publisher on its own port rather than a role-keyed payload
# on :8775. Two reasons: (1) the frame above is consumed by the already-shipped recorder
# and by force_torque/ tooling, and widening it would need every reader updated in
# lockstep; (2) one publisher per sensor keeps the "one singleton per owner" model — each
# publisher drains exactly one SensorReader's ring, so a right-hand sensor that is absent
# or dies simply never opens its port instead of silently starving a shared batch.
FT_RIGHT_WS_PORT_DEFAULT = 8779

# ---- Per-hand bridge ports (LEFT = the historical unsuffixed ports) ----------
# The tracker bridge stays SINGLE (one pysurvive USB singleton serves all 3 roles in one
# role-keyed frame); the camera and gripper are per-hand devices, so they get one bridge
# process each. Declared here so the launcher, the recorder defaults and the health
# probes cannot drift apart.
TRACKER_WS_PORT_DEFAULT = 8772          # all roles, role-keyed frame
CAMERA_WS_PORT_DEFAULT = 8873           # LEFT-hand D405  (8773/8777 belong to the DEPLOY stack's
CAMERA_RIGHT_WS_PORT_DEFAULT = 8877     # RIGHT-hand D405  robot-mounted bridges; both stacks share a box)
GRIPPER_WS_PORT_DEFAULT = 8774          # LEFT-hand DM4310 (bidirectional)
GRIPPER_RIGHT_WS_PORT_DEFAULT = 8778    # RIGHT-hand DM4310 (bidirectional)

# ---- Recorder control WS (the recorder process exposes this; the viz GUI +
# the keyboard gesture bridge send intents to it) ----
#   {"cmd": "start", "task": "<name>"}   begin a continuous take (s key)
#   {"cmd": "stop"}                      end the take + flush atomically (e key)
#   {"cmd": "home"}                      relay 'c' return-to-initial-pose to the viz
#   {"cmd": "marker"}                    record an episode-split marker (DECISION-E)
#   {"cmd": "discard"}                   drop the in-progress take / last saved take
# and the recorder broadcasts status:
#   {"type":"status","recording":bool,"take_idx":int,"task":str,
#    "frames":{"tracker":int,"tracker_right":int,"tracker_head":int,
#              "gripper":int,"ft":int,"camera":int,
#              "gripper_right":int,"ft_right":int,"camera_right":int},
#    "markers":int,"elapsed":float,"last_saved":str,"home_seq":int}
# The three *_right counters are ALWAYS present and read 0 when that device is absent —
# a stable key set means the viz's status panel does not have to branch on hardware.
# home_seq increments on each 'c' press; the viz observes the change and resets
# the robot to home on its render thread (the recorder never touches the robot).
RECORDER_CTRL_PORT_DEFAULT = 8776
