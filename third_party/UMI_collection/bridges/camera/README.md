# camera bridge — RealSense D405 RGB over WebSocket

`realsense_bridge.py` owns the gripper-mounted Intel RealSense **D405** (a USB
singleton) and publishes its live RGB feed as base64-encoded JPEG over a
localhost WebSocket. The viz (`viz/collection_viz.py`) is a thin WS client: it
decodes each frame and paints it on a camera frustum that follows the gripper.
Splitting the camera into its own process mirrors the tracker / gripper bridges
in the hub-and-spoke layout.

- **Publishes:** D405 **COLOR** stream, `rgb8` **640x480@30**, JPEG-encoded,
  base64 in a JSON text frame, on **`ws://127.0.0.1:8773`** by default — the collection
  launcher passes `--port 8873` (RIGHT `8877`), because 8773/8777 are what the deploy
  stack's robot-mounted bridges bind and both stacks share a box.
- **Env:** runs under **`.venv-vive`** (py3.9) — the only venv with
  `pyrealsense2`. (`opencv` is used for fast JPEG encode, with a PIL fallback.)
- **D405 color quirk:** on the D405 the COLOR stream is exposed on the
  **Stereo Module** sensor, not a separate RGB sensor. This is handled inside
  the bridge; nothing downstream needs to know.

## Run

```bash
# under .venv-vive
python bridges/camera/realsense_bridge.py                 # first device, color 640x480
python bridges/camera/realsense_bridge.py --serial 323622272456
python bridges/camera/realsense_bridge.py --stream left-ir
python bridges/camera/realsense_bridge.py --mock          # no hardware
```

`run_collection_viz.sh` launches this bridge as part of the full stack.

## Wire format

One WS text frame per image:

```json
{"seq": 0, "t": 0.0, "w": 640, "h": 480, "stream": "color", "jpeg_b64": "<base64 JPEG>"}
```

base64 keeps each frame a single JSON text frame, so the client stays simple; a
640x480 JPEG at this rate is well within a localhost socket's budget. A newly
(re)connected client is sent the last-known frame immediately (level-triggered),
so it paints without waiting for the next capture.

## Hardware-swap robustness

- **No `--serial` ⇒ first enumerated device.** A same-model D405 swap needs no
  config change — just replug.
- **`--serial <sn>`** pins a specific device when more than one is attached.
- **`--stream left-ir`** falls back to the left infrared imager (grayscale) when
  color is unavailable or undesired; `--stream color` (default) uses the RGB
  stream off the Stereo Module.
- **`--mock`** streams a synthetic moving gradient with no hardware, so the viz
  and end-to-end smoke tests run camera-less.

## Reliability

A dedicated reader thread owns the `pyrealsense2` pipeline and writes the latest
JPEG under a lock; the asyncio broadcaster only ships the newest frame (so a slow
client never stalls capture). The reader uses `try_wait_for_frames` (which
returns `(ok, frames)` and does **not** raise on timeout) inside an outer
**reconnect loop**: a transient USB-contention stall tears down and rebuilds the
pipeline after a short backoff rather than crashing the bridge.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--host` | `127.0.0.1` | WS bind (localhost-only) |
| `--port` | `8773` | WS port (collection passes 8873 / 8877; deploy keeps 8773 / 8777) |
| `--serial` | `None` | D405 serial; `None` ⇒ first device |
| `--stream` | `color` | `color` \| `left-ir` |
| `--width` | `640` | capture width |
| `--height` | `480` | capture height |
| `--fps` | `30` | camera capture fps (D405 color: `5/15/30/60/90` at 640x480; `5/15/30` at 1280x720) |
| `--publish-hz` | `15.0` | publish rate. **MUST be an integer divisor of `--fps`** — the bridge refuses to start otherwise. `0` = every frame |
| `--jpeg-quality` | `80` | JPEG quality |
| `--report-every` | `10.0` | seconds between achieved-rate + frame-gap-histogram log lines (`0` = off) |
| `--mock` | off | synthetic frames, no camera (uses the SAME pacer as the hardware path) |

(`--rate-hz` was removed: the WS broadcaster is now event-driven. A polled
latest-value register drops frames once the publish rate approaches the poll rate
— measured 8.1% loss at publish 30 / poll 30, and 19.1% unthrottled.)

## ⚠️ TIMELINE INVARIANT — the published grid must be UNIFORM

This bridge's frames become the **master timeline** of the training dataset:
`processing/convert_to_umi.py` takes `cam_t = cam_index[:, 0]` and resamples pose,
gripper and wrench onto it, and both training loaders index windows in **frame
units with no dt input**. A non-uniform camera dt is therefore multiplicative
noise on every learned velocity and force time-derivative.

Publishing is **count-based decimation on the SDK frame number** (`fn % decim`),
not a wall-clock throttle. The original code used
`if (now - last_pub) < period: continue` with `last_pub = now`; with `--fps 30`
(33.3439 ms measured, std 0.3 µs) and `--publish-hz 15` (66.6667 ms) the decision
margin is **+21 µs** while host arrival jitter is **300–470 µs**, so the
accept/reject decision was made by scheduler noise, and the re-anchored reference
made the phase random-walk. **Result: 97 recorded takes at 11.90 Hz with dt
alternating between 2 and 3 capture periods (47.9% / 52.1%), ±20% rms per-step
jitter.** After the fix: 15.000 Hz, dt 66.67 ms ±0.5 ms, one single frame-gap
bucket.

Guards: the bridge refuses a non-integer decimation at startup, logs the achieved
rate + frame-gap histogram every `--report-every` seconds, and prints a
non-uniform-gap / sensor-drop count at exit. `recording/recorder.py` writes
`camera_grid_uniform` / `camera_dt_k` / `camera_seq_gaps` into every `take_meta`
and warns loudly at stop; `processing/convert_to_umi.py` stamps
`attrs["camera_dt_uniform"]` on the zarr; `processing/audit_session.py` warns per
take. Tests: `bridges/camera/test_pacing.py`, `recording/test_camera_timeline.py`.

## Gotcha

Only **one** process can own the D405 at a time. If the bridge can't open the
camera (or keeps hitting the reconnect loop), close any other app holding the
device first (`realsense-viewer`, another bridge instance, etc.), then restart.
