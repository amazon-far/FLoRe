# Replay & conversion fidelity

Two offline viewers let you confirm a recording is good **before** training on the
converted dataset, and pinpoint anything the raw→Zarr conversion lost. They are
read-only and independent of the live `RECORD=1 ./run_collection_viz.sh` pipeline.

| Tool | Shows | Launch | Port |
|---|---|---|---|
| `replay/replay_raw_viz.py` | the RAW take (yam gripper cluster + TCP path, jaws, recorded D405, F/T) | `./run_replay_viz.sh <session\|take>` | `8083` |
| `replay/replay_zarr_viz.py` | the CONVERTED `.zarr.zip`, one episode at a time, *as stored* | `./run_replay_zarr_viz.sh <file.zarr.zip>` | `8084` |

Both put a **draggable timeline** (real seconds) in the right panel — drag to stop
at a timestamp, `⏵` to play in true time, `⏮/⏭` to step. The zarr tool adds a
**clickable episode dropdown** (the converter splits one take into N+1 episodes at
the markers).

## The authoritative pose chain: raw puck → the zarr TCP action

**Every transform between the physical VIVE puck and the trained `robot0_eef_pos`, in
order. There are exactly TWO coordinate transforms — and one of them happens at RECORD
time, not in the converter.** Anything not listed here is not applied.

```
        physical VIVE puck
               │
   ① libsurvive solves the raw pose            (libsurvive world frame)
               │
   ② tracker bridge LEFT-multiplies R_world     ← WORLD_RPY, launcher default "0 0 180"
      p ← R_world·p ,  q ← R_world⊗q              bridges/tracker/vive_tracker_bridge.py
               │                                  ::_aligned_pose_payload
               ▼
        tracker.npy  ← WHAT IS RECORDED already contains ②
               │       (provenance: session_meta.world_align)
               │
   ③ hygiene + resample (NOT a frame change)    convert_to_umi / replay_raw_viz
      drop stale rows → make_strictly_increasing
      → deglitch_tracker_positions (PRIMARY only)
      → resample onto the ~15 Hz camera grid
        (Slerp rotation, clamped interp1d position)
               │
   ④ the TCP extrinsic — the ONE converter-side transform
      world_T_tcp = world_T_puck @ T_tracker_tcp
      T_tracker_tcp = [ M | (0, −0.225, 0.05) ]   robot_assets/calibration/tcp_extrinsic.json
      M = [[0,−1,0],[−1,0,0],[0,0,−1]]            (the −y,−x,−z puck→wrist relabel, det=+1)
               │
   ⑤ pose_to_6d → rotvec, cast f32
               ▼
   robot0_eef_pos [T,3] + robot0_eef_rot_axis_angle [T,3]
```

So, starting from `tracker.npy`: **the zarr TCP is the recorded tracker pose put through
exactly one extrinsic (④).** No home-anchor, no re-levelling, no second world rotation, no
per-episode re-basing. (③ is data hygiene and rate-matching, not a change of frame.)

### The other two keypoints take the same chain, with a different step ④

`tracker_right.npy` → `robot1_eef_*` is the chain above **verbatim** — same `T_tracker_tcp`,
because the two hands are the same assembly on FK-symmetric wrists.

`tracker_head.npy` → `head_*` (the 3pt CHEST/torso body reference) also takes ①②③ verbatim,
then its **own** step ④ — one that is **rotation-only**:

```
   ④' the CHEST body relabel
      world_T_chest = world_T_puck @ T_tracker_head
      T_tracker_head = [ H | (0, 0, 0) ]        robot_assets/calibration/head_extrinsic.json
      H = Ry(90°)·Rx(90°) = [[0,1,0],[0,0,−1],[−1,0,0]]      (det=+1, ORDER 3: H³=I)
        ⇒ x_new = −z_old,  y_new = +x_old,  z_new = −y_old
               ▼
   head_pos [T,3]  (BIT-IDENTICAL to the raw puck position — the translation is zero)
   head_rot_axis_angle [T,3]  (the chest BODY frame, not the raw chest-puck axes)
```

Three things not to get wrong here:

* **`H` is not `M`.** Different puck, different bracket. And `M` is symmetric (`M²=I`, so
  `M⁻¹=M` and a forward/inverse slip silently cancels) while `H` has order 3 — reusing one
  for the other, or inverting it, mis-orients the keypoint with no other symptom.
* **Zero translation is enforced, not assumed.** `load_head_extrinsic` raises on a nonzero
  translation: the chest has no lever (there is no wrist or TCP to reach), and a chest
  keypoint quietly displaced by a few cm is invisible in every plot and path overlay.
* **Only the ROTATION moves, so nothing else can witness a drift.** The live `RAW_FRAME=1`
  chest triad (`collection_viz.raw_chest_pose_from_puck`), the raw replay viewer
  (`sec_pose_disp`) and the zarr viewer (which draws `head_*` as stored) must therefore all
  agree by construction; `viz/test_logic_collection.py` pins the live constant to this file
  and `processing/test_dual_hand_3pt.py` pins the viewer to the converter's own function.

The relabel is recorded per dataset in `attrs.T_tracker_head` (identity == the raw
chest-puck axes), which is the only way to tell the two conventions apart after the fact.

⚠️ **3pt datasets converted before 2026-08-26 carry the RAW chest-puck axes in
`head_rot_axis_angle`** (`head_pos` is unaffected). Anything supervised on the head
*rotation* — the 32-D 3pt layout's `[23:29]` head rot6d, and `hl_head_action_pose` — is in
the old convention, so re-convert before mixing with, or deploying against, a rig running
this file. The relative-pose transform does not save you: a constant right-multiplication
**conjugates** the anchor-relative rotation (`(R_aH)ᵀ(R_tH) = Hᵀ(R_aᵀR_t)H`), it does not
cancel the way a global left-multiplication like step ② does.

### Why step ② does not affect the action

`R_world` is a **rigid GLOBAL** rotation of the whole world frame, so it cancels out of every
relative quantity — which is what a delta/relative-pose policy consumes. Measured on
`session_2026.08.06_16.35.57/take_000` (718 samples), with vs without `Rz(180°)`:

| quantity | difference |
|---|---|
| per-step \|Δpos\| | **1.2e-17 m** |
| per-step body Δrot | **1.1e-14 deg** |

It only rotates the **absolute** trajectory in world (here, 180° about z). The live
`RAW_FRAME=1` collection view, both replay viewers, and the zarr all show the SAME aligned
frame, so what you see == what is stored. To record in the libsurvive-native frame instead,
collect with `WORLD_RPY="0 0 0"` — but do not change it casually: it makes new sessions'
absolute frame inconsistent with old ones, for zero gain on the deltas.

### Step ④ was wrong before 2026-08-06 — re-convert older datasets

The extrinsic used to be stored as a **pure translation** `(0.225, 0, −0.05)` with identity
rotation. That number is the lever in **WRIST** coordinates, but `apply_tcp_extrinsic`
applies the extrinsic in the **PUCK BODY** frame, and those two frames differ by exactly `M`.
The correct puck-frame form is `M·(0.225,0,−0.05) = (0,−0.225,0.05)` **plus the rotation `M`**
— the same single 4×4 verified interactively on `replay/inspect_tracker_frame.py` (`:8088`).
Applying a wrist-frame vector in the puck frame put every stored eef **33 cm off in position
and 180° off in rotation**. The identical conflation also lived in the replay viewers'
`wrist_offset_from_stored` (it added a puck-frame translation to a wrist-frame vector); both
now use `wrist_pose_from_stored`, a full 4×4 inverse that returns a rotation too, plus
`puck_triad_local_quat` so the "tracker" triad renders at the raw puck axes instead of
inheriting the wrist axes.

⚠️ **Any dataset converted before this fix carries a 33 cm / 180° wrong `robot0_eef_pos`
and must be RE-CONVERTED and RE-TRAINED.** `hl_serving._TCP_OFFSET` (the wrist→TCP lever,
`0.242`) is a *different leg* and is unaffected — do not set the two equal.

### Verification (re-runnable)

On `session_2026.08.06_16.35.57`, checked against independently hand-composed transforms
(not the repo helpers), the chain above holds to storage precision:

| check | result |
|---|---|
| zarr `robot0_eef_pos` == raw puck ∘ extrinsic (hand-composed) | 1.8e-5 mm / 8e-7 deg |
| zarr eef == the raw viewer's camera-grid TCP | 1.8e-5 mm / 8e-7 deg |
| stored eef == `wrist + 0.242·wrist_x` (the measured datum) | 1.5e-5 mm |
| all three drawn triads: zarr viewer vs raw viewer | 1.5e-5 mm / 8e-7 deg |
| zarr "tracker" triad == the raw puck in `tracker.npy` | 1.5e-5 mm / 8e-7 deg |

The residual is `float32` storage in the zarr (the viewers compute in `float64`), not a
transform error. `replay/test_replay_logic.py` pins the cluster math for the current
remapped extrinsic, a legacy pure-translation one, and identity.

## Why two tools, and how they line up

Both tools draw the SAME gripper-only cluster the live collection viz shows for
`MODE=1pt/2pt` (`DISPLAY_MODE=gripper`): the three yam-gripper meshes, exactly the
three axis triads (TCP, tracker, wrist_yaw_link), the wrist-mounted camera, F/T, and
the TCP path as a green polyline. Neither runs the mink IK — the cluster is placed
with pure numpy (`gripper_only_view`). The raw tool derives the TCP (`robot0_eef_pos`)
pose from the recorded VIVE puck through the **same** `tracker->TCP` extrinsic the
converter uses (`umi_convert_core.apply_tcp_extrinsic` + the session's
`tcp_extrinsic.json`), so the raw-take cluster and the `.zarr.zip` cluster are
numerically identical — a raw take and its conversion can be scrubbed side by side.

The zarr tool renders **only what the `.zarr.zip` contains** (the stored TCP pose,
the stored 224×224 RGB, the resampled F/T and gripper width), reading the extrinsic
back from the dataset's own `T_tracker_tcp` attr; the raw tool computes it live from
the raw puck.

The two run at different **playback rates on purpose.** The zarr tool plays the stored
data, which lives on the ~15 Hz camera grid. The raw tool plays the moving cluster (and
its green path, F/T arrows, gripper) at the tracker's **native ~80 Hz**, holding each
~15 Hz RGB frame across the ~5 display frames it spans — so the recorded hand motion
reads smoothly instead of stepping between camera frames. This is display-only: the
raw tool's camera-grid resample still equals `robot0_eef_pos` to ~1e-8 m (the fidelity
contract), and a display sample landing on a camera timestamp equals the stored value —
the denser grid restores motion, it does not invent a different trajectory.

Both viewers now default to the **RAW stored frame** — the same convention as
`RAW_FRAME=1 collect_data.sh` — so the two are directly comparable frame-for-frame and
match what the operator saw while collecting. Each still offers a **home-anchored** toggle
(upright, z-up, re-based onto a fixed home); that applies a world rotation the DATA does not
contain, so it is for comfortable eyeballing only and must never be used to judge fidelity.
Comparing the two viewers in the raw frame is what surfaces a conversion artifact (a
frozen-tracker run, a misaligned camera/pose, a resampling kink) before a training run
instead of after one.

## What conversion can silently lose — and how to spot it

`processing/convert_to_umi.py` resamples four asynchronous streams onto each
episode's camera-frame grid (Slerp for rotation, clamped `interp1d` for
position/width, anti-aliased interval-mean for the 250 Hz F/T). The resampling
core is unit-tested (`processing/test_convert_core.py`), but several failure modes
are *data-dependent* and only visible on the actual recording:

| Symptom in the zarr replay | Likely cause | Where to confirm |
|---|---|---|
| `gripper_width` flat, never crosses 0.08 m | stale gripper endpoints zeroing the signal (see below) | `inspect_umi.py` prints the width range; compare to the raw replay |
| TCP path has a kink / teleports | tracker dropout interpolated as motion, or a quaternion-order slip | scrub both tools at the same timestamp |
| TCP path has a frozen run | a stale-tracker stretch (`stale==1` rows are dropped, but a long gap is held) | raw replay shows the same hold → it's real, not a bug |
| camera and pose drift apart over an episode | timestamp misalignment between streams | step frame-by-frame near the end of a long episode |
| an episode is implausibly long / a sliver | a missed / doubled episode-split marker | `recording/review_take.py` (fix markers before converting) |
| `UNCALIBRATED_DO_NOT_TRAIN` banner | identity TCP extrinsic (see §user-intervention) | `inspect_umi.py` attrs |

The timestamps in the zarr are **relative per episode** (each starts at 0,
DECISION-A); the raw tool shows take-relative seconds. Same motion, different zero.

## The gripper-width gotcha (fixed in the converter; know it anyway)

**The DM4310 is an absolute encoder that re-references on every motor power-cycle**,
so the open/closed encoder *readings* shift between sessions. The repo default
`robot_assets/calibration/gripper_range.json` ships the i2rt **nominal** endpoints
(`enc_open=4.9`, `enc_closed=0.0`) and is explicitly "re-map per power-cycle".

If a session's live encoder sits outside `[0, 4.9]` (e.g. all-negative after a
re-reference), mapping with the nominal endpoints **clamps every width to one rail**
→ `gripper_width` is a flat line, every frame reads as "grasping", and the whole
grasp signal is destroyed — silently.

**Fix (in `umi_convert_core.effective_gripper_calib`, used by the converter and the
review/replay tools):** the recorder already stores the **homed endpoints
per-sample** in `gripper.npy` columns `open_rad`/`closed_rad` (the startup
self-test homes the real stops each launch and the gripper bridge reports them on
every frame). The converter now **prefers those recorded endpoints**, keeping the
physical `width_*_m` from the JSON. It logs which it used:

```
take_000: gripper endpoints from RECORDED homing (enc -6.524..-1.254 rad)
```

If you see `from gripper_range.json (no/degenerate recorded endpoints…)` on a real
session, the gripper bridge wasn't reporting endpoints — the width may be wrong;
check the raw replay's gripper chart.

## Items that still need a human (flagged for the operator)

These are **not** auto-resolvable and affect training quality. Confirm them per rig:

1. **`width_open_m` is a URDF estimate, not a measurement.**
   `gripper_range.json` has `width_open_m = 0.095 m` with the note "CONFIRM with
   calipers before training (crosses UMI's 0.08 m grasp threshold)." Because UMI
   thresholds grasp at a hard-coded 0.08 m, a wrong `width_open_m` shifts which
   frames count as grasping. **Action:** measure the real two-finger opening with
   calipers and set `width_open_m` (and `width_closed_m` if non-zero). The encoder
   endpoints are taken from the recording automatically; only the metres need you.

2. **TCP extrinsic — now a ROTATING transform; re-measure after any mount change.**
   `tcp_extrinsic.json` is `[ M | (0, −0.225, 0.05) ]` in the **puck body** frame (see
   the pose-chain section above): the rotation `M` is the measured −y,−x,−z puck→wrist
   axis relabel, and the translation is the puck→TCP lever expressed in that same frame.
   It composes to the TCP sitting at `wrist_yaw_link + 0.242 m` along wrist +x — the
   measured fingertip-midpoint datum — with the TCP axes equal to the wrist axes.
   `calibrated: true`; an identity extrinsic would make the converter stamp
   `UNCALIBRATED_DO_NOT_TRAIN`. Full history is in the file's `_note`. **Re-measure
   puck→TCP (both the rotation and the lever) if you change the gripper or the tracker
   mount** — the puck→wrist axis relabel is a property of how the puck is bolted on.

3. **Camera→TCP extrinsic is NOT stored in the zarr.** The zarr replay mounts the
   frustum on the TCP with a fixed *approximate* outward rotation and a
   `--cam-offset` you can pass (the zarr tool has no `--cam-rpy` — the rotation is
   hardcoded); it is a viewing aid, not the calibrated camera pose. UMI training
   uses the RGB pixels, not a 3D camera pose, so this doesn't affect the dataset —
   but don't read the frustum's 3D placement as ground truth. (The live viz and the
   raw replay tool expose `--cam-offset/--cam-rpy`, likewise marked UNCALIBRATED.)

4. **`out_res` / image transform.** The converter stores 224×224 with
   aspect-preserving resize + center crop (UMI default). If your policy expects a
   different resolution or a full-frame (no-crop) image, set `--out-res` and verify
   the crop in the zarr replay (the stored RGB is exactly what trains).
