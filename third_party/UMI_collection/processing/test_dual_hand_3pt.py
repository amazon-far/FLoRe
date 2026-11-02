#!/usr/bin/env python3
"""3pt DUAL-HAND pipeline: record -> take layout -> convert -> zarr keys -> viewer flags.

The rig's 3pt configuration is LEFT hand = tracker+camera+F/T+gripper, RIGHT hand =
tracker+camera+F/T+gripper, CHEST = tracker ONLY. Every RIGHT-hand modality is
INDEPENDENTLY optional, so the properties worth pinning are not "3pt works" but:

  1. a 1pt session is BYTE-IDENTICAL to before (no ``*_right`` file, no camera1_rgb);
  2. a 3pt session with the right hardware present writes every right-hand file and
     converts to camera1_rgb / robot1_ft_wrench / a REAL robot1_gripper_width;
  3. a 3pt session recorded BEFORE the right camera/Bota/gripper are wired up still
     produces a valid take + zarr — it just has no right-hand modality, and nothing is
     zero-filled into looking like a measurement;
  4. the two hands' streams never cross (distinct values end up in distinct keys);
  5. camera1_rgb is paired onto the master grid CAUSALLY (lag, never a future frame);
  6. the chest keypoint carries pose only.

Run:  PYTHONPATH=recording:processing .venv/bin/python processing/test_dual_hand_3pt.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "recording"))
sys.path.insert(0, os.path.join(_ROOT, "replay"))

import recording_format as fmt  # noqa: E402
import umi_convert_core as core  # noqa: E402

_PY = os.path.join(_ROOT, ".venv", "bin", "python")
_CHECKS = 0


def check(cond, msg: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


# --------------------------------------------------------------------------- #
# fixtures: real recorder + real converter runs (no hardware)
# --------------------------------------------------------------------------- #
def _record(out: str, mode: str, right: bool, seconds: float = 3.0,
            serials: bool = False) -> str:
    """Run the REAL recorder in --mock and return the take dir.

    ``serials`` passes a DISTINCT fake serial per role of the mode, so session_meta's
    serial->role map is exercised (it is provenance only — the recorder never resolves a
    device by it). Off by default so the historical fixtures are byte-identical."""
    cmd = [_PY, os.path.join(_ROOT, "recording", "recorder.py"), "--mock",
           "--out", out, "--mode", mode, "--mock-seconds", str(seconds)]
    if serials:
        flag = {"left": "--left-serial", "right": "--right-serial", "head": "--head-serial",
                "left_foot": "--left-foot-serial", "right_foot": "--right-foot-serial"}
        for i, role in enumerate(fmt.TRACKER_ROLES_BY_MODE[mode]):
            cmd += [flag[role], f"LHR-FAKE{i:04d}"]
    if right:
        # A URL is the recorder's "this device exists" assertion — the same rule the
        # live launcher uses, so the mock exercises the real gate.
        cmd += ["--camera-right-url", "ws://mock", "--ft-right-url", "ws://mock",
                "--gripper-right-url", "ws://mock"]
    subprocess.run(cmd, check=True, capture_output=True)
    return os.path.join(out, "session_mock", "take_000")


def _convert(session_dir: str, out_zarr: str) -> None:
    subprocess.run([_PY, os.path.join(_ROOT, "processing", "convert_to_umi.py"),
                    session_dir, "-o", out_zarr, "--out-res", "64", "64"],
                   check=True, capture_output=True)


def _open_zarr(path: str):
    import zarr
    from imagecodecs_numcodecs import register_codecs
    register_codecs()
    return zarr.open(zarr.ZipStore(path, mode="r"), "r")


# --------------------------------------------------------------------------- #
def test_1pt_take_layout_is_unchanged(tmp: str) -> None:
    take = _record(os.path.join(tmp, "r1"), "1pt", right=False, seconds=2.0)
    files = set(os.listdir(take))
    for name in files:
        check(not name.endswith("_right.npy") and name != fmt.CAMERA_RIGHT_DIR,
              f"1pt take carries no right-hand artifact ({name})")
    check(fmt.TRACKER_RIGHT_FILE not in files, "1pt: no tracker_right.npy")
    check(fmt.CAMERA_DIR in files and fmt.FT_FILE in files and fmt.GRIPPER_FILE in files,
          "1pt: the historical left-hand files are all present")


def test_3pt_without_right_hardware_records_tracker_only(tmp: str) -> None:
    """The state the rig is in TODAY: three pucks, one camera, one Bota, one gripper."""
    take = _record(os.path.join(tmp, "r3a"), "3pt", right=False, seconds=2.0)
    files = set(os.listdir(take))
    check(fmt.TRACKER_RIGHT_FILE in files and fmt.TRACKER_HEAD_FILE in files,
          "3pt: right + chest TRACKERS recorded (a puck needs no extra hardware)")
    check(fmt.CAMERA_RIGHT_DIR not in files, "3pt w/o right camera: no camera_right/ dir")
    check(fmt.FT_RIGHT_FILE not in files, "3pt w/o right Bota: no ft_wrench_right.npy")
    check(fmt.GRIPPER_RIGHT_FILE not in files, "3pt w/o right gripper: no gripper_right.npy")
    meta = json.load(open(os.path.join(take, fmt.TAKE_META_FILE)))
    # ABSENCE IS A FACT: the counters exist and read 0, and hands.right says which
    # modalities were contributed — so downstream never infers capability from
    # which files happen to exist.
    for key in ("camera_right", "ft_right", "gripper_right"):
        check(meta["counts"][key] == 0, f"counts.{key} == 0 (present key, honest zero)")
    check(meta["hands"]["right"]["tracker"] > 0, "hands.right.tracker > 0 (the puck)")
    check(meta["hands"]["right"]["camera"] == 0, "hands.right.camera == 0")
    check(meta["hands"]["right"]["camera_grid_uniform"] is None,
          "an absent right camera reports grid_uniform=None, not a false True")


def test_3pt_with_right_hardware_writes_every_file(tmp: str) -> None:
    take = _record(os.path.join(tmp, "r3b"), "3pt", right=True, seconds=3.0)
    files = set(os.listdir(take))
    for name in (fmt.TRACKER_RIGHT_FILE, fmt.TRACKER_HEAD_FILE, fmt.CAMERA_RIGHT_DIR,
                 fmt.FT_RIGHT_FILE, fmt.GRIPPER_RIGHT_FILE):
        check(name in files, f"3pt dual-hand take writes {name}")
    # The two hands must not be the same bytes written twice.
    ftl = np.load(os.path.join(take, fmt.FT_FILE))
    ftr = np.load(os.path.join(take, fmt.FT_RIGHT_FILE))
    check(ftl.shape[1] == ftr.shape[1] == len(fmt.FT_COLS),
          "both hands' wrench files share the FT_COLS layout")
    check(not np.allclose(ftl[:, 1:4], ftr[:, 1:4]),
          "left and right wrench are DISTINCT (streams did not cross)")
    gl = np.load(os.path.join(take, fmt.GRIPPER_FILE))
    gr = np.load(os.path.join(take, fmt.GRIPPER_RIGHT_FILE))
    check(not np.allclose(gl[:, 1], gr[:, 1]), "left and right gripper encoders are DISTINCT")
    cam_r = os.path.join(take, fmt.CAMERA_RIGHT_DIR)
    idx_r = np.load(os.path.join(cam_r, fmt.CAMERA_INDEX_FILE))
    njpg = len([f for f in os.listdir(cam_r) if f.endswith(".jpg")])
    check(njpg == idx_r.shape[0] > 0,
          f"camera_right/ has one JPEG per index row ({njpg} == {idx_r.shape[0]})")
    left0 = open(os.path.join(take, fmt.CAMERA_DIR, fmt.camera_frame_name(0)), "rb").read()
    right0 = open(os.path.join(cam_r, fmt.camera_frame_name(0)), "rb").read()
    check(left0 != right0, "the two camera dirs hold DIFFERENT images (not one stream twice)")


def test_5pt_records_two_more_body_pucks(tmp: str) -> None:
    """5pt = a 3pt take PLUS two tracker files, and nothing else.

    The whole claim of the mode: the two feet are BODY REFERENCES like the chest, so they
    add PUCKS and no hardware. What could go wrong silently is a foot whose rows land in
    another role's file (then two keypoints carry one puck's motion and the audit still
    passes), so this asserts the five traces are DISTINCT as well as present."""
    take = _record(os.path.join(tmp, "r5"), "5pt", right=True, seconds=3.0, serials=True)
    files = set(os.listdir(take))
    for name in (fmt.TRACKER_FILE, fmt.TRACKER_RIGHT_FILE, fmt.TRACKER_HEAD_FILE,
                 fmt.TRACKER_LEFT_FOOT_FILE, fmt.TRACKER_RIGHT_FOOT_FILE):
        check(name in files, f"5pt take writes {name}")
    # No new HARDWARE files: the feet have no camera / F-T / gripper of their own.
    for stem in ("foot", "left_foot", "right_foot"):
        check(not any(f.startswith(f"camera_{stem}") or f.startswith(f"ft_wrench_{stem}")
                      or f.startswith(f"gripper_{stem}") for f in files),
              f"5pt invented no per-hand file for {stem}")
    rows = {role: np.load(os.path.join(take, fmt.TRACKER_FILE_BY_ROLE[role]))
            for role in fmt.TRACKER_ROLES_BY_MODE["5pt"]}
    for role, arr in rows.items():
        check(arr.ndim == 2 and arr.shape[1] == len(fmt.TRACKER_COLS) and arr.shape[0] > 0,
              f"{role}: [N,{len(fmt.TRACKER_COLS)}] with N>0 ({arr.shape})")
    names = list(rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = rows[names[i]][:, 2:5], rows[names[j]][:, 2:5]
            check(not np.allclose(a, b),
                  f"{names[i]} and {names[j]} carry DISTINCT positions (no crossed streams)")
    meta = json.load(open(os.path.join(take, fmt.TAKE_META_FILE)))
    check(meta["mode"] == "5pt" and meta["primary_role"] == "left",
          "the take stamps mode=5pt, primary=left")
    for role in ("right", "head", "left_foot", "right_foot"):
        check(meta["counts"][f"tracker_{role}"] > 0, f"counts.tracker_{role} > 0")
    sess = json.load(open(os.path.join(os.path.dirname(take), fmt.SESSION_META_FILE)))
    check(set(sess["serial_role_map"].values()) == set(fmt.TRACKER_ROLES_BY_MODE["5pt"]),
          "session_meta maps a serial to each of the five roles")
    # A 1pt take must not grow a foot file, and its counters stay honest zeros.
    take1 = _record(os.path.join(tmp, "r5b"), "1pt", right=False, seconds=1.5)
    files1 = set(os.listdir(take1))
    check(fmt.TRACKER_LEFT_FOOT_FILE not in files1 and fmt.TRACKER_RIGHT_FOOT_FILE not in files1,
          "1pt writes no foot tracker file")
    m1 = json.load(open(os.path.join(take1, fmt.TAKE_META_FILE)))
    check(m1["counts"]["tracker_left_foot"] == 0 and m1["counts"]["tracker_right_foot"] == 0,
          "1pt reports the foot counters as present-and-zero (absence is a fact)")


def test_audit_requires_ALL_FIVE_in_5pt(tmp: str) -> None:
    """A glitch on ONE 5pt FOOT must fail the take — the same conjunctive gate the chest
    gets. Recording in 5pt is the statement that all five keypoints are part of the
    dataset, and a foot has no ``*_pose_glitch`` mask downstream any more than the chest
    does, so an unbridged excursion would otherwise ship unflagged."""
    src = _record(os.path.join(tmp, "r5c"), "5pt", right=True, seconds=3.0)
    sess = os.path.join(tmp, "r5c_sess", "session_mock")
    os.makedirs(os.path.dirname(sess), exist_ok=True)
    shutil.copytree(os.path.dirname(src), sess)
    take = os.path.join(sess, os.path.basename(src))
    # An unbridged position teleport on the RIGHT foot only.
    path = os.path.join(take, fmt.TRACKER_RIGHT_FOOT_FILE)
    arr = np.load(path)
    arr[len(arr) // 2:, 2] += 0.75
    np.save(path, arr)
    import audit_session as aud

    r = aud.audit_take(take, 16)
    check(not r["all_roles_clean"], "a RIGHT-foot teleport fails the 5pt take")
    check("right_foot" in r["dirty_roles"], f"right_foot is named dirty ({r['dirty_roles']})")
    check("left_foot" in r["clean_roles"] and "head" in r["clean_roles"],
          f"the other body references stay clean ({r['clean_roles']})")
    check(any("right_foot" in e and "right_foot_pose_glitch" in e for e in r["warn"]),
          f"the finding names the role and the per-role mask the converter writes ({r['warn']})")


def test_dual_hand_converts_to_the_additive_zarr_keys(tmp: str) -> None:
    session = os.path.join(tmp, "r3b", "session_mock")
    out = os.path.join(tmp, "dual.zarr.zip")
    _convert(session, out)
    z = _open_zarr(out)
    keys = set(z["data"].keys())
    for k in ("camera1_rgb", "robot1_ft_wrench", "robot1_gripper_width",
              "robot1_eef_pos", "robot1_eef_rot_axis_angle", "head_pos", "head_rot_axis_angle"):
        check(k in keys, f"dual-hand zarr carries {k}")
    T = z["data"]["camera0_rgb"].shape[0]
    for k in ("camera1_rgb", "robot1_ft_wrench", "robot1_gripper_width", "head_pos"):
        check(z["data"][k].shape[0] == T, f"{k} is length-T ({T}) like camera0_rgb")
    check(z.attrs["n_robots"] == 2 and z.attrs["n_cameras"] == 2,
          "n_robots/n_cameras are DERIVED from the contents (2/2)")
    hands = z.attrs["hands"]
    check(all(hands["right"][m] for m in ("pose", "camera", "ft", "gripper")),
          "attrs.hands.right reports every modality present")
    check(hands["chest"]["pose"] and not any(hands["chest"][m] for m in ("camera", "ft", "gripper")),
          "attrs.hands.chest is POSE ONLY (a body reference has no per-hand sensor)")
    check(not np.allclose(z["data"]["robot0_ft_wrench"][:], z["data"]["robot1_ft_wrench"][:]),
          "robot0/robot1 wrench stay distinct through the conversion")
    check(not np.allclose(z["data"]["robot0_gripper_width"][:],
                          z["data"]["robot1_gripper_width"][:]),
          "robot1_gripper_width is a REAL second stream, not a copy or a zero-fill")
    check(not np.array_equal(z["data"]["camera0_rgb"][0], z["data"]["camera1_rgb"][0]),
          "camera0/camera1 stay distinct through the JpegXl round-trip")
    check("camera1_pair_rule" in z.attrs and z.attrs["camera1_pair_rule"] == "causal_hold_on_camera0_grid",
          "the camera1 pairing rule is recorded in attrs")


def test_absent_right_hardware_converts_without_the_keys(tmp: str) -> None:
    session = os.path.join(tmp, "r3a", "session_mock")
    out = os.path.join(tmp, "tracker_only.zarr.zip")
    _convert(session, out)
    z = _open_zarr(out)
    keys = set(z["data"].keys())
    check("robot1_eef_pos" in keys and "head_pos" in keys,
          "the right + chest POSES still convert (they only need pucks)")
    for k in ("camera1_rgb", "robot1_ft_wrench", "robot1_gripper_width"):
        check(k not in keys,
              f"{k} is ABSENT rather than zero-filled — 'no sensor' stays distinguishable "
              f"from 'the sensor read zero'")
    check(z.attrs["n_cameras"] == 1, "n_cameras stays 1 with no right camera")
    check(z.attrs["hands"]["right"]["pose"] and not z.attrs["hands"]["right"]["camera"],
          "attrs.hands.right: pose yes, camera no")


def test_1pt_zarr_data_keys_are_unchanged(tmp: str) -> None:
    session = os.path.join(tmp, "r1", "session_mock")
    out = os.path.join(tmp, "one.zarr.zip")
    _convert(session, out)
    z = _open_zarr(out)
    keys = set(z["data"].keys())
    for k in ("robot1_eef_pos", "robot1_ft_wrench", "camera1_rgb", "head_pos",
              "robot1_gripper_width"):
        check(k not in keys, f"1pt zarr has no {k}")
    check(z.attrs["n_robots"] == 1 and z.attrs["n_cameras"] == 1, "1pt attrs: 1 robot, 1 camera")


# --------------------------------------------------------------------------- #
# the camera1 pairing rule, in isolation
# --------------------------------------------------------------------------- #
def test_frame_pairing_is_causal() -> None:
    """camera1 is paired by SELECTION (images cannot be interpolated). Causal = the
    chosen frame never postdates the grid time — lag is a safer error than a
    future-frame leak on the action-label grid."""
    src = np.array([0.0, 0.1, 0.2, 0.3])
    grid = np.array([0.05, 0.1, 0.19, 0.31])
    idx, st = core.resample_frame_index(src, grid, causal=True)
    check(list(idx) == [0, 1, 1, 3], f"causal hold picks the latest frame at/<= t (got {list(idx)})")
    check(np.all(src[idx] <= grid + 1e-12), "no chosen frame is in the FUTURE of its grid time")
    check(st["skew_max_s"] <= 0.0, "causal skew is non-positive (the view lags, never leads)")
    # A grid time before the secondary camera's first frame has no causal answer.
    idx2, st2 = core.resample_frame_index(np.array([1.0, 1.1]), np.array([0.5, 1.05]), causal=True)
    check(st2["n_pre"] == 1 and idx2[0] == 0,
          "a pre-start grid time is COUNTED (n_pre) and clamped, not silently paired")
    # A slower secondary camera reuses frames — recorded, so it is auditable.
    idx3, st3 = core.resample_frame_index(np.array([0.0, 0.2]), np.array([0.0, 0.05, 0.1, 0.15]),
                                          causal=True)
    check(st3["reused_max"] == 4, f"frame reuse is measured (got {st3['reused_max']})")
    # nearest mode exists but is NOT the default; assert it differs so the choice is real.
    idx4, _ = core.resample_frame_index(src, np.array([0.09]), causal=False)
    idx5, _ = core.resample_frame_index(src, np.array([0.09]), causal=True)
    check(idx4[0] == 1 and idx5[0] == 0, "nearest and causal genuinely differ")
    check(core.resample_frame_index(np.zeros(0), grid)[0].shape == (0,),
          "an empty secondary stream returns an empty selection (no crash)")


def test_format_declares_one_naming_source() -> None:
    """recorder / converter / both viewers index HAND_FILES_BY_ROLE, so a per-hand
    modality cannot be added to one side only."""
    check(set(fmt.HAND_FILES_BY_ROLE) == {"left", "right"},
          "only the HANDS have per-hand files ('head' is a body reference)")
    lk, rk = fmt.HAND_FILES_BY_ROLE["left"], fmt.HAND_FILES_BY_ROLE["right"]
    check(set(lk) == set(rk), "both hands declare the SAME modality set")
    for k in lk:
        check(lk[k] != rk[k], f"left/right {k} are different paths ({lk[k]} vs {rk[k]})")
    check(lk["tracker"] == fmt.TRACKER_FILE and lk["camera_dir"] == fmt.CAMERA_DIR,
          "the LEFT hand keeps the historical unsuffixed names")
    check(fmt.hand_roles_for_mode("1pt") == ("left",), "1pt has one hand")
    check(fmt.hand_roles_for_mode("2pt") == fmt.hand_roles_for_mode("3pt")
          == fmt.hand_roles_for_mode("5pt") == ("left", "right"),
          "3pt and 5pt add PUCKS, not further sets of sensors")
    check("head" not in fmt.hand_roles_for_mode("3pt"),
          "the chest is never a hand role (no camera / F-T / gripper)")
    # 5pt: the two FEET are BODY REFERENCES, exactly the statement the chest is.
    check(fmt.body_roles_for_mode("5pt") == ("head", "left_foot", "right_foot"),
          "5pt declares three body references (chest + both feet)")
    check(fmt.body_roles_for_mode("3pt") == ("head",) and fmt.body_roles_for_mode("1pt") == (),
          "body_roles_for_mode is unchanged for 1pt/3pt")
    check(not (set(fmt.BODY_ROLES) & set(fmt.HAND_ROLES)),
          "a role is a hand or a body reference, never both")
    check(set(fmt.TRACKER_FILE_BY_ROLE) == set(fmt.HAND_ROLES) | set(fmt.BODY_ROLES),
          "every role has exactly one on-disk tracker file")
    check(len(set(fmt.TRACKER_FILE_BY_ROLE.values())) == len(fmt.TRACKER_FILE_BY_ROLE),
          "no two roles share a tracker filename (a shared one silently merges two pucks)")
    check(fmt.TRACKER_ROLES_BY_MODE["5pt"][:3] == fmt.TRACKER_ROLES_BY_MODE["3pt"],
          "5pt is 3pt plus the two feet, in wire order")
    ports = {fmt.TRACKER_WS_PORT_DEFAULT, fmt.CAMERA_WS_PORT_DEFAULT,
             fmt.CAMERA_RIGHT_WS_PORT_DEFAULT, fmt.GRIPPER_WS_PORT_DEFAULT,
             fmt.GRIPPER_RIGHT_WS_PORT_DEFAULT, fmt.FT_WS_PORT_DEFAULT,
             fmt.FT_RIGHT_WS_PORT_DEFAULT, fmt.RECORDER_CTRL_PORT_DEFAULT}
    check(len(ports) == 8, f"all eight bridge ports are distinct (got {sorted(ports)})")


# --------------------------------------------------------------------------- #
# the zarr viewer reads exactly what is in the file
# --------------------------------------------------------------------------- #
def test_zarr_viewer_flags_track_the_file(tmp: str) -> None:
    import replay_zarr_viz as rzv
    dual = rzv.ZarrEpisodes(os.path.join(tmp, "dual.zarr.zip"))
    check(all([dual.has_robot1, dual.has_cam1, dual.has_ft1, dual.has_grip1, dual.has_head]),
          "dual-hand file -> every secondary flag True")
    tracker_only = rzv.ZarrEpisodes(os.path.join(tmp, "tracker_only.zarr.zip"))
    check(tracker_only.has_robot1 and tracker_only.has_head,
          "tracker-only 3pt -> poses present")
    check(not any([tracker_only.has_cam1, tracker_only.has_ft1, tracker_only.has_grip1]),
          "tracker-only 3pt -> right-hand modality flags False (the viewer draws no frustum)")
    one = rzv.ZarrEpisodes(os.path.join(tmp, "one.zarr.zip"))
    check(not any([one.has_robot1, one.has_cam1, one.has_ft1, one.has_grip1, one.has_head]),
          "1pt file -> every secondary flag False (viewer identical to before)")


def test_home_anchored_display_starts_at_the_urdf_zero_pose(tmp: str) -> None:
    """Home-anchored display: PER-ROLE home + PER-ROLE reference.

    The operator starts a session with both handheld units held like the G1's own zero
    configuration (forearms level, jaws parallel), so with the anchor ON frame 0 must put
    each keypoint EXACTLY at that keypoint's zero-pose home — hands parallel and 0.297 m
    apart, chest at the torso point. This is the same per-hand anchor the teleop bridge
    applies, so what the operator sees while collecting matches what deploy does.

    The cost, asserted here so it can't be forgotten: each hand is re-zeroed
    independently, so the operator's MEASURED hand-to-hand offset does not survive — only
    the CHANGE in their relative pose does. The measured geometry is what the anchor-OFF
    default draws, and OFF is the fidelity check."""
    import gripper_only_view as gov
    rng = np.random.default_rng(0)
    T = 6
    posL = rng.normal(size=(T, 3))
    posR = posL + np.array([0.0, -0.4, 0.0])          # a fixed 40 cm lateral offset
    posC = posL + np.array([-0.3, 0.2, 0.35])
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))

    def rebase(pos, role):
        hp, hq = gov.HOME_WRIST_POSE[role]
        out = np.empty_like(pos)
        outq = np.empty((pos.shape[0], 4))
        for k in range(pos.shape[0]):
            out[k], outq[k] = gov.home_anchor(hp, hq, pos[0], q[0], pos[k], q[k])
        return out, outq

    dL, qL = rebase(posL, "left")
    dR, qR = rebase(posR, "right")
    dC, _ = rebase(posC, "head")
    for role, got in (("left", dL[0]), ("right", dR[0]), ("head", dC[0])):
        check(np.allclose(got, gov.HOME_WRIST_POSE[role][0]),
              f"frame 0 of the {role} keypoint sits EXACTLY at its zero-pose home")
    sep = float(gov.HOME_WRIST_POSE["left"][0][1] - gov.HOME_WRIST_POSE["right"][0][1])
    check(abs((dL[0][1] - dR[0][1]) - sep) < 1e-12,
          f"the two hands start {sep:.6f} m apart (the home dict's own separation)")
    check(np.allclose(qL[0], qR[0]),
          "the two hands start with IDENTICAL rotation — the jaws are parallel")
    check(np.allclose(dL[0][[0, 2]], dR[0][[0, 2]]),
          "at frame 0 the hands differ ONLY in y (level forearms, same reach + height)")
    # Relative MOTION is preserved even though the absolute offset is re-zeroed.
    check(np.allclose((dR - dL) - (dR[0] - dL[0]), (posR - posL) - (posR[0] - posL[0])),
          "the CHANGE in the hand-to-hand offset survives the per-hand rebasing exactly")
    # And the thing that is deliberately NOT preserved, pinned so it stays a choice.
    check(not np.allclose(dR[0] - dL[0], posR[0] - posL[0]),
          "the MEASURED 40 cm offset is deliberately replaced by the robot's own "
          "separation (anchor OFF is what shows the measured geometry)")


def test_raw_viewer_gives_the_right_hand_the_SAME_frame_chain(tmp: str) -> None:
    """The raw viewer must apply the puck->TCP extrinsic to the RIGHT hand and the CHEST's
    own rotation-only body relabel to the head — the converter's own per-role rule
    ("robot1_* = RIGHT hand, same TCP extrinsic as robot0 (FK-symmetric wrists)"; the chest
    is a body reference with no TCP, but its puck IS bolted on rotated).

    This is a regression test for a real bug: the right cluster was drawn on the RAW puck
    frame while the left was drawn on the TCP, so the two hands rendered **0.2305 m and
    180 deg** apart and the right gripper meshes hung on a puck frame while the left's hung
    on a wrist frame. Uses the REAL rig calibration, not the session's — a mock session has
    an IDENTITY extrinsic, under which the bug is invisible."""
    import json

    import replay_raw_viz as rrv
    import umi_convert_core as ucore
    from scipy.spatial.transform import Rotation

    cal = os.path.join(_ROOT, "robot_assets", "calibration", "tcp_extrinsic.json")
    T = np.asarray(json.load(open(cal))["T_tracker_tcp"], dtype=float)
    check(not np.allclose(T[:3, :3], np.eye(3)) and np.linalg.norm(T[:3, 3]) > 0.2,
          "the rig extrinsic is a real transform (rotation + ~0.23 m), so this test can bite")
    hcal = os.path.join(_ROOT, "robot_assets", "calibration", "head_extrinsic.json")
    TH = np.asarray(json.load(open(hcal))["T_tracker_head"], dtype=float)
    check(not np.allclose(TH[:3, :3], np.eye(3)) and np.allclose(TH[:3, 3], 0.0),
          "the chest relabel is a real ROTATION with EXACTLY zero translation")
    check(not np.allclose(TH[:3, :3], T[:3, :3]),
          "the chest relabel is NOT the hands' M — a different puck on a different bracket")

    session = os.path.join(tmp, "r3b", "session_mock")
    take = rrv.RawTake(os.path.join(session, "take_000"), session)
    check(sorted(take.sec_trackers) == ["head", "right"],
          "the 3pt take exposes both secondary keypoints to the viewer")
    for role in ("right", "head"):
        raw_p, raw_q = take.sec_trackers[role]
        out_p, out_q = take.sec_pose_disp(role, T, TH)
        moved = float(np.linalg.norm(out_p - raw_p, axis=1).mean())
        spun = float(np.abs(np.abs(out_q) - np.abs(raw_q)).max())
        if role == "head":
            check(moved == 0.0,
                  "the CHEST keeps its RAW puck POSITION (its relabel is rotation-only, so "
                  "head_pos is bit-identical), matching convert_to_umi")
            check(spun > 1e-3,
                  "...but its AXES are relabelled into the chest body frame, so the triad "
                  "draws the same frame head_rot_axis_angle stores")
        else:
            check(moved > 0.2,
                  f"the RIGHT hand DOES get the puck->TCP extrinsic ({moved:.4f} m), so its "
                  f"cluster rides the same frame as the left's and as robot1_eef_pos")

    # And it must be the converter's own numbers, not a re-derivation that could drift.
    import convert_to_umi as conv
    for role, fname, extr in (("right", fmt.TRACKER_RIGHT_FILE, T),
                              ("head", fmt.TRACKER_HEAD_FILE, TH)):
        arr = np.load(os.path.join(session, "take_000", fname))
        g = take._resample_tracker(arr, deglitch=False, query_t_abs=take.cam_t_abs)
        vpos, vrot = ucore.apply_tcp_extrinsic(
            g[0], Rotation.from_quat(g[1][:, [1, 2, 3, 0]]), extr)
        c6 = conv._resample_tracker_to_6d(arr, take.cam_t_abs, extr)
        err = float(np.abs(vpos - c6[:, :3]).max())
        rerr = float(np.abs(vrot.as_rotvec() - c6[:, 3:]).max())
        check(err < 1e-9 and rerr < 1e-9,
              f"the viewer's {role} pose is bit-equal to the converter's own function "
              f"({err:.1e} m, {rerr:.1e} rad)")


def test_audit_sees_glitches_on_the_SECONDARY_trackers(tmp: str) -> None:
    """``audit_session`` must inspect the right-hand and chest pucks, not just the primary.

    Regression test for a real hole: ``audit_take`` LOADED ``tracker_right.npy`` /
    ``tracker_head.npy`` into locals and then never looked at them, so every lighthouse
    glitch on those pucks passed the session audit silently. That is worse than it sounds
    — the converter writes a ``robot0_pose_glitch`` mask (which the HL loader's
    ``drop_glitch_windows`` reads) ONLY for the primary, so a secondary glitch reaches
    ``robot1_*``/``head_*`` with nothing downstream able to mask it, and the audit is the
    only gate that could have caught it.

    Injects the two failure modes that survive resampling into a training label: an
    UNBRIDGED position teleport (a fabricated interpolation ramp) on the right hand, and a
    there-and-back ROTATION excursion (invisible to the position-only deglitch) on the
    chest. The no-jump requirement is CONJUNCTIVE over the roles the MODE declares, so
    either one alone must fail the whole take."""
    import audit_session as aud

    src = os.path.join(tmp, "r3b", "session_mock")           # the 3pt fixture take
    sess = os.path.join(tmp, "audit_glitch", "session_mock")
    shutil.rmtree(os.path.dirname(sess), ignore_errors=True)
    shutil.copytree(src, sess)
    td = os.path.join(sess, "take_000")

    clean = aud.audit_take(td, 16)
    check(sorted(clean["secondary_trackers"]) == ["head", "right"],
          "the audit reports both secondary keypoints as present on a 3pt take")
    check(clean["mode_roles"] == ["left", "right", "head"],
          "a 3pt take declares all THREE keypoints as required")
    check(not any("right tracker" in w or "head " in w for w in clean["warn"]),
          "a CLEAN 3pt take raises no secondary-tracker warning (no false positives)")
    check(clean["all_roles_clean"] and clean["dirty_roles"] == [],
          f"and passes the no-jump gate on every declared keypoint {clean['clean_roles']}")

    # RIGHT hand: a 0.45 m step that never returns = an unbridged excursion.
    rp = os.path.join(td, fmt.TRACKER_RIGHT_FILE)
    arr = np.load(rp)
    arr[arr.shape[0] // 2:, 2] += 0.45
    np.save(rp, arr)
    # CHEST: a 60 deg there-and-back rotation over 2 samples, position untouched.
    hp = os.path.join(td, fmt.TRACKER_HEAD_FILE)
    h = np.load(hp)
    i0 = h.shape[0] // 2
    half = np.radians(60.0) / 2.0
    cw, sw = np.cos(half), np.sin(half)
    for i in (i0, i0 + 1):
        w, x, y, z = h[i, 5:9]
        h[i, 5:9] = [w * cw - x * sw, w * sw + x * cw, y * cw + z * sw, z * cw - y * sw]
    np.save(hp, h)

    bad = aud.audit_take(td, 16)
    check(bad.get("right_unbridged", 0) >= 1,
          "the audit counts the RIGHT hand's unbridged excursion "
          f"({bad.get('right_unbridged')} found, residual "
          f"{bad.get('right_residual_jump_mm', 0):.0f} mm)")
    check(any("right tracker" in e and "unbridged" in e for e in bad["warn"]),
          "a DECLARED role's glitch is reported (advisory since 2026-09-11: the converter "
          "masks it per role via right_pose_glitch, so it costs windows, not the take)")
    check(any("robot1_eef_pos" in e for e in bad["warn"]),
          "the message names the ZARR KEY the glitch corrupts")
    check(any("right_pose_glitch" in e for e in bad["warn"]),
          "and names the per-role mask the converter writes for it")
    check(bad.get("head_rot_excursions", 0) >= 1,
          f"the audit catches the CHEST's rotation excursion "
          f"({bad.get('head_rot_excursions')} found, peak "
          f"{bad.get('head_rot_excursion_amp_deg', 0):.0f} deg) — the position-only "
          f"deglitch cannot remove it")
    check(any("head" in e and "rotation excursion" in e for e in bad["warn"]),
          "and reports it too, naming the chest")
    check(not bad["all_roles_clean"] and sorted(bad["dirty_roles"]) == ["head", "right"],
          f"the no-jump verdict is CONJUNCTIVE: both bad keypoints are named dirty "
          f"{bad['dirty_roles']} even though the primary is clean {bad['clean_roles']}")
    check(len(bad["warn"]) > len(clean["warn"]) and len(bad["err"]) == len(clean["err"]),
          "and the take is FLAGGED but not blocked: the converter keeps it and masks the windows")


def test_audit_requires_BOTH_hands_in_2pt(tmp: str) -> None:
    """In 2pt the requirement covers both hands: a RIGHT-hand jump must fail the take even
    though the left (primary) is spotless. The chest is not declared in 2pt, so it must not
    be demanded — the gate follows the MODE, not whatever files happen to exist."""
    import audit_session as aud

    take = _record(os.path.join(tmp, "audit2pt"), "2pt", right=False, seconds=4.0)
    sess = os.path.dirname(take)
    shutil.copy(os.path.join(_ROOT, "robot_assets", "calibration", "tcp_extrinsic.json"),
                os.path.join(sess, fmt.CALIB_DIR, fmt.TCP_EXTRINSIC_FILE))

    r0 = aud.audit_take(take, 16)
    check(r0["mode_roles"] == ["left", "right"],
          "a 2pt take declares exactly the two hands (no chest)")
    check(r0["all_roles_clean"],
          f"a clean 2pt take passes on both hands {r0['clean_roles']}")

    rp = os.path.join(take, fmt.TRACKER_RIGHT_FILE)
    arr = np.load(rp)
    arr[arr.shape[0] // 2:, 3] += 0.5          # unbridged 0.5 m step in y
    np.save(rp, arr)
    r1 = aud.audit_take(take, 16)
    check(r1["dirty_roles"] == ["right"] and not r1["all_roles_clean"],
          "a RIGHT-hand jump fails the 2pt take even with a spotless primary")
    check(any("right tracker" in e for e in r1["warn"]),
          "and it is reported (advisory: masked per role by the converter, take kept)")


def test_audit_leaves_an_UNDECLARED_role_advisory(tmp: str) -> None:
    """Severity follows the declared MODE, not file presence. A secondary puck that the
    mode does not declare is still audited (so nothing goes unlooked-at) but stays a WARN:
    no declared keypoint depends on it, and `--hands left` conversions are unaffected."""
    import audit_session as aud

    src = os.path.join(tmp, "r3b", "session_mock")
    sess = os.path.join(tmp, "audit_undecl", "session_mock")
    shutil.rmtree(os.path.dirname(sess), ignore_errors=True)
    shutil.copytree(src, sess)
    td = os.path.join(sess, "take_000")
    # Re-stamp the take as 1pt while the right/head files remain on disk.
    mp = os.path.join(td, fmt.TAKE_META_FILE)
    meta = json.load(open(mp))
    meta["mode"] = "1pt"
    json.dump(meta, open(mp, "w"))
    arr = np.load(os.path.join(td, fmt.TRACKER_RIGHT_FILE))
    arr[arr.shape[0] // 2:, 2] += 0.45
    np.save(os.path.join(td, fmt.TRACKER_RIGHT_FILE), arr)

    r = aud.audit_take(td, 16)
    check(r["mode_roles"] == ["left"], "a 1pt stamp declares only the primary")
    check(r["secondary_trackers"] == ["right", "head"],
          "the undeclared pucks are still INSPECTED (never silently skipped)")
    check(any("right tracker" in w for w in r["warn"]),
          "and their findings are reported")
    check(not any("right tracker" in e for e in r["err"]),
          "but stay ADVISORY — the mode does not declare them, so they block nothing")
    check("advisory" in " ".join(r["warn"]),
          "and the message says so explicitly")


def test_audit_fails_a_declared_role_that_recorded_NOTHING(tmp: str) -> None:
    """A take stamped ``mode=3pt`` whose head puck recorded NOTHING cannot satisfy a
    conjunctive gate: the converter just omits ``head_*``, so the zarr reads as clean
    lower-mode data and the shortfall surfaces only at training time. A keypoint that does
    not exist is not 'clean' — it is missing."""
    import audit_session as aud

    src = os.path.join(tmp, "r3b", "session_mock")
    sess = os.path.join(tmp, "audit_nohead", "session_mock")
    shutil.rmtree(os.path.dirname(sess), ignore_errors=True)
    shutil.copytree(src, sess)
    td = os.path.join(sess, "take_000")
    os.remove(os.path.join(td, fmt.TRACKER_HEAD_FILE))

    r = aud.audit_take(td, 16)
    check(r["secondary_trackers"] == ["right"],
          "the audit reports only the keypoints that actually recorded")
    check(any("mode=3pt" in e and "head" in e for e in r["err"]),
          "a DECLARED keypoint that recorded nothing is an ERR")
    check(r["dirty_roles"] == ["head"] and not r["all_roles_clean"],
          "and it counts as failing the no-jump gate, not as absent-and-fine")


def test_audit_missing_PRIMARY_is_not_scored_clean(tmp: str) -> None:
    """A take with no usable ``tracker.npy`` must not read as jump-free.

    Regression test for a false pass this gate created: the primary's metrics are absent
    when its block never ran, and reading them through ``.get(key, 0)`` scored a take with
    NO primary tracker as clean on every keypoint. Absent metrics are not evidence of
    cleanliness. There was also no error at all for a missing primary — ``no ft_wrench.npy``
    was an err, but the more load-bearing ``tracker.npy`` was silent."""
    import audit_session as aud

    src = os.path.join(tmp, "r3b", "session_mock")
    sess = os.path.join(tmp, "audit_noprim", "session_mock")
    shutil.rmtree(os.path.dirname(sess), ignore_errors=True)
    shutil.copytree(src, sess)
    td = os.path.join(sess, "take_000")
    os.remove(os.path.join(td, fmt.TRACKER_FILE))

    r = aud.audit_take(td, 16)
    check(any("tracker.npy" in e for e in r["err"]),
          "a missing PRIMARY tracker is an ERR")
    check("left" in r["dirty_roles"] and not r["all_roles_clean"],
          f"and the primary is NOT scored clean {r['dirty_roles']}")


def test_audit_is_unchanged_for_a_1pt_take(tmp: str) -> None:
    """The secondary checks must be purely ADDITIVE: a 1pt take has no secondary files, so
    no new key, warning or error may appear. This is what lets the historical per-session
    QC verdicts (and the recorded clean-list exclusions) still reproduce — separately
    checked byte-for-byte against real 1pt sessions."""
    import audit_session as aud

    take = _record(os.path.join(tmp, "audit1pt"), "1pt", right=False)
    r = aud.audit_take(take, 16)
    check(r["secondary_trackers"] == [],
          "a 1pt take reports no secondary keypoints")
    check(not any(w.startswith(("right ", "head ")) for w in r["warn"]),
          "and raises no secondary-tracker warning")
    check(not any(e.startswith(("right ", "head ")) for e in r["err"]),
          "and no secondary-tracker error")
    check(not any(k.startswith(("right_", "head_")) for k in r),
          "and records no secondary metric keys at all, so the audit table is unchanged")
    check(r["mode_roles"] == ["left"] and r["all_roles_clean"],
          "the conjunctive gate degenerates to the primary alone in 1pt")


def main() -> int:
    if not os.path.exists(_PY):
        print(f"SKIP: no venv python at {_PY}")
        return 0
    tmp = tempfile.mkdtemp(prefix="dual3pt_")
    try:
        print("== take layout ==")
        test_1pt_take_layout_is_unchanged(tmp)
        test_3pt_without_right_hardware_records_tracker_only(tmp)
        test_3pt_with_right_hardware_writes_every_file(tmp)
        test_5pt_records_two_more_body_pucks(tmp)
        print("== conversion ==")
        test_dual_hand_converts_to_the_additive_zarr_keys(tmp)
        test_absent_right_hardware_converts_without_the_keys(tmp)
        test_1pt_zarr_data_keys_are_unchanged(tmp)
        print("== pairing + format contract ==")
        test_frame_pairing_is_causal()
        test_format_declares_one_naming_source()
        print("== viewers ==")
        test_zarr_viewer_flags_track_the_file(tmp)
        test_home_anchored_display_starts_at_the_urdf_zero_pose(tmp)
        test_raw_viewer_gives_the_right_hand_the_SAME_frame_chain(tmp)
        print("== session audit: every declared keypoint must be jump-free ==")
        test_audit_sees_glitches_on_the_SECONDARY_trackers(tmp)
        test_audit_requires_BOTH_hands_in_2pt(tmp)
        test_audit_leaves_an_UNDECLARED_role_advisory(tmp)
        test_audit_fails_a_declared_role_that_recorded_NOTHING(tmp)
        test_audit_missing_PRIMARY_is_not_scored_clean(tmp)
        test_audit_is_unchanged_for_a_1pt_take(tmp)
        test_audit_requires_ALL_FIVE_in_5pt(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nLOGIC OK ({_CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
