#!/usr/bin/env python3
"""RealSense D405 RGB -> WebSocket bridge for the data-collection viz.

The D405 is rigidly mounted on the gripper, so its live RGB feed is the
egocentric view the viz parents under the moving tracker frame. This bridge owns
the camera (a USB singleton) and streams JPEG frames over a localhost WebSocket;
the viz decodes them and paints them on a camera frustum that follows the
tracker. Splitting it into its own process mirrors the tracker / gripper bridges:
the camera lives in the ``pika`` env (the only one with pyrealsense2), while the
viz runs under ``bota``.

D405 specifics (probed on serial 323622272456): the COLOR stream is exposed on
the **Stereo Module** sensor as ``rgb8`` at 424x240 / 480x270 / 640x360 /
640x480 / 848x480 @ 5/15/30/60/90 and 1280x720 @ 5/15/30. We default to color
640x480@30 published at 15 Hz because:
  * ``rgb8`` matches viser's channel order directly (no BGR swap);
  * 640x480 JPEG-encodes in ~4.5 ms (q80), so 30 Hz costs ~40% of one core --
    encode is NOT the bottleneck, and 15 Hz is comfortable;
  * 1280x720 encodes at p99 ~33 ms, which does not fit a 33 ms budget;
  * "left RGB camera" = the D405 color stream (its color image is derived from
    the left imager). ``--stream left-ir`` falls back to the left infrared
    imager (grayscale) if that's what you meant.

⚠️ TIMELINE INVARIANT (the recorded camera grid must be UNIFORM)
----------------------------------------------------------------
This bridge's published frames become the MASTER TIMELINE of the training
dataset: ``processing/convert_to_umi.py`` takes ``cam_t = cam_index[:, 0]`` and
resamples every other modality (pose / gripper / wrench) onto it, and the
training dataset indexes windows in **frame units** with no dt input. A
non-uniform camera dt therefore silently corrupts every learned velocity and
every force time-derivative.

Publishing is decimated **by frame count** (``fn % decim``), not by wall clock.
The previous implementation used ``if (now - last_pub) < period: continue`` with
``last_pub = now``, i.e. a period test whose reference is reset to the *actual*
publish instant. With ``--fps 30`` (33.3439 ms measured, std 0.3 us) and
``--publish-hz 15`` (66.667 ms) the margin is **+21 us** while host arrival
jitter is 300-470 us -- 14-22x larger. So the accept/reject decision was made by
OS scheduling noise, and because the reference re-anchored on each publish the
phase random-walked: the recorded data came out at **11.90 Hz** with dt
alternating between 2 and 3 capture periods (47.9% / 52.1%), i.e. +-20% rms
per-step jitter. Count-based decimation is immune to arrival jitter by
construction. Keying it on the SDK ``frame_number`` (not a local counter) means a
genuine sensor drop leaves a hole of exactly ``2*decim`` -- detectable
downstream -- instead of silently sliding the phase.

Wire format (one WS text frame per image)::

    {"seq": int, "t": float, "t_host": float, "capture_seq": int,
     "w": int, "h": int, "stream": "color"|"left-ir", "jpeg_b64": "<base64 JPEG>"}

base64 keeps it a single JSON text frame (simpler client; the localhost socket
carries ~9.8 Mbit/s even at 30 Hz).

GIL/perf: ``wait_for_frames`` blocks, so — exactly like ``bota_ft_viz`` — a
dedicated reader thread owns the pipeline and writes the latest JPEG under a
lock; the asyncio broadcaster is woken per frame (event-driven, not polled — a
polled latest-value register drops 8-19% of frames once the publish rate
approaches the poll rate). ``--mock`` synthesizes a moving gradient so the
viz/eval run with no camera attached, using the SAME decimation path.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import sys
import threading
import time

import numpy as np
import websockets  # noqa: E402

WS_HOST = "127.0.0.1"
WS_PORT = 8773


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """RGB uint8 HxWx3 -> JPEG bytes. Prefers cv2, falls back to PIL/imageio."""
    try:
        import cv2  # noqa: PLC0415

        # One thread: cv2 otherwise spawns one per core (24 here), which both
        # doubles the p99 encode time (13.3 -> 7.3 ms measured) and steals cores
        # from the IK render loop and the Bota EKF in the sibling processes.
        try:
            cv2.setNumThreads(1)
        except Exception:  # noqa: BLE001
            pass
        bgr = rgb[:, :, ::-1]  # cv2 expects BGR
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return buf.tobytes()
    except Exception:  # noqa: BLE001
        pass
    from PIL import Image  # noqa: PLC0415

    bio = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


def resolve_decimation(fps: int, publish_hz: float, tol: float = 0.01) -> int:
    """Integer capture-frame decimation for a requested publish rate.

    Returns ``N`` such that publishing every ``N``-th captured frame yields
    ``fps / N`` Hz with a UNIFORM period. Raises ``ValueError`` when the request
    is not an integer divisor of ``fps``, because "approximately N" is exactly
    the failure mode this replaces: a wall-clock period test that lands within a
    few microseconds of a capture-period boundary is decided by scheduler noise,
    and the recorded timeline comes out non-uniform.

    ``publish_hz <= 0`` means "publish every frame" -> 1.
    """
    if publish_hz is None or publish_hz <= 0:
        return 1
    if fps <= 0:
        raise ValueError(f"--fps must be > 0, got {fps}")
    exact = fps / publish_hz
    decim = max(1, int(round(exact)))
    if abs(exact - decim) > tol:
        raise ValueError(
            f"--publish-hz {publish_hz} does not divide --fps {fps} "
            f"(fps/publish_hz = {exact:.4f}, nearest integer {decim}). "
            f"Non-integer decimation cannot produce a uniform camera timeline, and "
            f"the camera timeline is the master grid of the training dataset. "
            f"Pick one of: "
            + ", ".join(f"{fps / n:g} Hz (1/{n})" for n in range(1, 7))
        )
    return decim


class CameraState:
    """Lock-guarded latest JPEG frame + metadata."""

    def __init__(self, stream: str = "color"):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.w = 0
        self.h = 0
        self.n = 0
        self.t = 0.0
        self.t_host = 0.0
        self.capture_seq = -1
        self.connected = False
        self.error = ""
        self.stream = stream  # "color" | "left-ir" — reflected in the wire frame
        # Set by the reader after every new frame so the broadcaster is
        # event-driven instead of polling a latest-value register.
        self.on_frame = None  # Optional[Callable[[], None]]
        # Diagnostics for the uniformity invariant.
        self.published = 0
        self.capture_gaps = 0  # publishes whose frame_number delta != decim
        self.dropped_by_sdk = 0  # sensor-side frame_number discontinuities

    def set_frame(
        self,
        jpeg: bytes,
        w: int,
        h: int,
        t: float,
        capture_seq: int,
        t_host: float | None = None,
    ) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.w = w
            self.h = h
            self.n += 1
            # `t` is the frame's own timestamp, captured BEFORE JPEG encode (and
            # from the sensor clock when available). Stamping after encode --
            # which is what this used to do -- folded a 4.5 ms mean / 13 ms p99
            # encode time into the master timeline of the dataset.
            self.t = t
            self.t_host = t_host if t_host is not None else t
            self.capture_seq = capture_seq
            self.published += 1
        cb = self.on_frame
        if cb is not None:
            cb()


class CameraReader(threading.Thread):
    def __init__(self, state: CameraState, args):
        super().__init__(daemon=True)
        self.state = state
        self.args = args
        self.decim = resolve_decimation(args.fps, args.publish_hz)
        # Phase/gap state must SPAN the reconnect loop: these used to be
        # `_stream_once` locals, so a mid-take USB stall (pipeline.stop/start,
        # measured 1.6 s = 24 publish periods) punched a hole that NONE of the
        # four guards saw. `frame_number` restarts at 0 on every pipeline.start,
        # so the publish phase is preserved, but the hole must still be counted.
        self._base_fn = None
        self._last_pub_fn = None
        self._last_fn = None
        # NOT `self._stop`: that shadows threading.Thread._stop, which Thread.join()
        # calls internally -> TypeError: 'Event' object is not callable.
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            if self.args.mock:
                self._run_mock()
            else:
                self._run_hardware()
        except Exception as exc:  # noqa: BLE001
            with self.state.lock:
                self.state.connected = False
                self.state.error = str(exc)
            print(f"[camera] reader error: {exc!r}", flush=True)

    def _run_hardware(self):
        import pyrealsense2 as rs  # noqa: PLC0415

        # Outer reconnect loop: a transient USB-contention stall raises
        # RuntimeError from the frame read; instead of letting the reader thread
        # die for the rest of the session, stop the pipeline, back off, and
        # re-start it. self._stop breaks out cleanly.
        backoff = 0
        while not self._stop_evt.is_set():
            try:
                self._stream_once(rs)
                backoff = 0  # clean exit (stop requested)
            except Exception as exc:  # noqa: BLE001
                with self.state.lock:
                    self.state.connected = False
                    self.state.error = str(exc)
                if self._stop_evt.is_set():
                    break
                wait = min(0.5 * (2 ** backoff), 5.0)
                backoff += 1
                print(f"[camera] pipeline error ({exc!r}); reconnecting in {wait:.1f}s", flush=True)
                # Sleep in small slices so --stop is responsive during backoff.
                end = time.perf_counter() + wait
                while time.perf_counter() < end and not self._stop_evt.is_set():
                    time.sleep(0.05)

    def _stream_once(self, rs) -> None:
        """One pipeline.start -> read-loop -> stop cycle. Raises on a stall."""
        pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        w, h, fps = self.args.width, self.args.height, self.args.fps
        if self.args.stream == "left-ir":
            # The left stereo imager is infrared index 1 on the D405.
            config.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
        else:
            config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        profile = pipeline.start(config)
        dev = profile.get_device()
        # Put the frame clock in the host EPOCH domain. Without this,
        # `frame.get_timestamp()` is milliseconds-since-device-boot
        # (`timestamp_domain.hardware_clock`) and useless as a wall clock. With
        # it, measured dt std is 0.048 ms vs 2.96 ms for a post-encode
        # `time.time()`. The absolute host<->device offset still wobbles ~1.5 ms,
        # which is the same order as the other bridges' own `time.time()` stamps,
        # so cross-modality alignment is a wash and the grid is 60x cleaner.
        global_time = False
        try:
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                    global_time = True
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] global_time unavailable ({exc!r}); using host clock", flush=True)
        decim = self.decim
        print(f"[camera] D405 started: {dev.get_info(rs.camera_info.name)} "
              f"serial={dev.get_info(rs.camera_info.serial_number)} "
              f"{self.args.stream} {w}x{h}@{fps} -> publish every {decim} frame(s) "
              f"= {fps / decim:g} Hz uniform, global_time={global_time}", flush=True)
        with self.state.lock:
            self.state.connected = True
            self.state.error = ""

        last_report = time.perf_counter()
        report_gaps: dict[int, int] = {}
        try:
            while not self._stop_evt.is_set():
                # try_wait_for_frames returns (ok, frames) and does NOT raise on
                # timeout, so a momentary stall just skips a frame instead of
                # killing the reader. (A hard device error still raises -> the
                # outer reconnect loop handles it.)
                ok, frames = pipeline.try_wait_for_frames(2000)
                if not ok:
                    continue
                if self.args.stream == "left-ir":
                    frame = frames.get_infrared_frame(1)
                else:
                    frame = frames.get_color_frame()
                if not frame:
                    continue

                # Timestamps FIRST, before any encode work.
                t_host = time.time()
                if global_time:
                    t = float(frame.get_timestamp()) / 1000.0
                    # Setting the option can succeed on a sensor that is not the
                    # one producing frames (multi-sensor devices), in which case
                    # `get_timestamp()` is still ms-since-device-boot (~1e9) and
                    # every frame would trip the recorder's `t < start_wall`
                    # guard -> a committed take with ZERO camera frames. Verify
                    # once, then trust.
                    if abs(t - t_host) > 0.5:
                        print(f"[camera] frame clock is NOT host-epoch "
                              f"(t={t:.3f} vs host={t_host:.3f}); falling back to "
                              f"the host clock", flush=True)
                        global_time = False
                        t = t_host

                fn = int(frame.get_frame_number())
                if self._base_fn is None:
                    self._base_fn = fn
                if self._last_fn is not None and fn - self._last_fn != 1:
                    with self.state.lock:
                        self.state.dropped_by_sdk += max(0, fn - self._last_fn - 1)
                self._last_fn = fn

                # ===== count-based decimation (the uniformity invariant) =====
                # Keyed on the SDK frame number, so this is immune to host
                # arrival jitter, and a sensor-side drop leaves a hole of exactly
                # 2*decim rather than sliding the phase.
                if (fn - self._base_fn) % decim:
                    continue

                if self._last_pub_fn is not None:
                    gap = fn - self._last_pub_fn
                    report_gaps[gap] = report_gaps.get(gap, 0) + 1
                    if gap != decim:
                        with self.state.lock:
                            self.state.capture_gaps += 1
                self._last_pub_fn = fn

                img = np.asanyarray(frame.get_data())
                if img.ndim == 2:  # grayscale IR -> RGB for a uniform wire format
                    img = np.repeat(img[:, :, None], 3, axis=2)
                jpeg = _encode_jpeg(np.ascontiguousarray(img), self.args.jpeg_quality)
                self.state.set_frame(
                    jpeg, img.shape[1], img.shape[0], t, fn, t_host=t_host
                )

                # ===== periodic self-report =====
                # A one-line "achieved rate + frame-number gap histogram" in the
                # launcher log is what would have made the old 11.9 Hz / 2-vs-3
                # defect unmissable on day one instead of after 97 takes.
                now = time.perf_counter()
                if self.args.report_every > 0 and now - last_report >= self.args.report_every:
                    span = now - last_report
                    npub = sum(report_gaps.values())
                    hist = " ".join(
                        f"{g}:{c / max(npub, 1) * 100:.0f}%"
                        for g, c in sorted(report_gaps.items())
                    )
                    uniform = len(report_gaps) <= 1
                    flag = "" if uniform else "  ** NON-UNIFORM **"
                    print(
                        f"[camera] publish {npub / span:.2f} Hz (target {fps / decim:g}) "
                        f"frame-gaps {{{hist}}}{flag}",
                        flush=True,
                    )
                    report_gaps = {}
                    last_report = now
        finally:
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            with self.state.lock:
                self.state.connected = False

    def _run_mock(self):
        w, h = self.args.width, self.args.height
        with self.state.lock:
            self.state.connected = True
        # Mirror the hardware path: synthesize frames on a FIXED-PHASE capture
        # grid at --fps, then run the SAME count-based decimation. The previous
        # mock used `sleep(publish_period)` after the encode work, which (a) drifts
        # (measured 12.5 Hz for a 15 Hz request) and (b) is a *different* pacer
        # from the hardware path, so --mock structurally could not reproduce the
        # timeline defect that shipped 97 takes.
        decim = self.decim
        cap_period = 1.0 / max(self.args.fps, 1)
        t0 = time.perf_counter()
        fn = 0
        # Precompute coordinate ramps once; only the phase shifts per frame.
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        while not self._stop_evt.is_set():
            deadline = t0 + (fn + 1) * cap_period
            sleep_for = deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            fn += 1
            if fn % decim:
                continue
            t_host = time.time()
            t = time.perf_counter() - t0
            r = (127 * (1 + np.sin(2 * math.pi * (xx + 0.1 * t)))).astype(np.uint8)
            g = (127 * (1 + np.sin(2 * math.pi * (yy + 0.13 * t)))).astype(np.uint8)
            b = np.full((h, w), int(80 + 80 * math.sin(t)), dtype=np.uint8)
            img = np.stack([r, g, b], axis=2)
            jpeg = _encode_jpeg(img, self.args.jpeg_quality)
            self.state.set_frame(jpeg, w, h, t_host, fn, t_host=t_host)


def _frame_msg(state: CameraState) -> "tuple[str, int] | None":
    """Build (json, seq) from the latest image, or None if there is none yet.

    Returns the seq it actually serialised so the caller cannot advance `last_n`
    past the frame it sent: re-reading `state.n` afterwards would let a frame that
    landed in between be discarded by the `n == last_n` test on the next wakeup.
    """
    with state.lock:
        jpeg = state.jpeg
        w, h, n = state.w, state.h, state.n
        t, t_host, cap = state.t, state.t_host, state.capture_seq
        stream = state.stream
    if jpeg is None:
        return None
    return json.dumps({
        "seq": n,
        "t": t,
        "t_host": t_host,
        "capture_seq": cap,
        "w": w,
        "h": h,
        "stream": stream,
        "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
    }), n


async def _serve(state: CameraState, host: str, port: int, stop: threading.Event):
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        peer = getattr(ws, "remote_address", "?")
        print(f"[camera] WS client connected: {peer} (total {len(clients)})", flush=True)
        # Level-triggered initial send: a reconnecting client gets the last-known
        # frame immediately rather than a blank panel until the next NEW frame.
        built = _frame_msg(state)
        if built is not None:
            try:
                await ws.send(built[0])
            except Exception:  # noqa: BLE001
                pass
        try:
            async for _ in ws:
                pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            clients.discard(ws)
            print(f"[camera] WS client gone: {peer} (total {len(clients)})", flush=True)

    loop = asyncio.get_running_loop()
    new_frame = asyncio.Event()

    # EVENT-DRIVEN, not polled. A polled latest-value register aliases: the
    # reader can overwrite `state.jpeg` between two polls, and the frame is gone
    # with no diagnostic. Measured on hardware with the old 30 Hz poll:
    # publish 15 Hz -> 0% loss, but publish 30 Hz -> 8.1% and unthrottled ->
    # 19.1%. Waking on each frame removes the whole failure class (and drops
    # client latency from ~25 ms to ~5 ms).
    def _wake():
        try:
            loop.call_soon_threadsafe(new_frame.set)
        except RuntimeError:  # loop already closed during shutdown
            pass

    state.on_frame = _wake

    async with websockets.serve(handler, host, port, max_size=None):
        print(f"[camera] WebSocket serving at ws://{host}:{port} (event-driven)", flush=True)
        last_n = -1
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(new_frame.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                new_frame.clear()
                if not clients:
                    continue  # skip base64+json entirely with no viewers
                built = _frame_msg(state)
                if built is None:
                    continue
                msg, n = built
                if n == last_n:
                    continue
                last_n = n
                websockets.broadcast(clients, msg)
        finally:
            state.on_frame = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=WS_HOST)
    ap.add_argument("--port", type=int, default=WS_PORT)
    ap.add_argument("--serial", default=None, help="D405 serial (else first device)")
    ap.add_argument("--stream", choices=["color", "left-ir"], default="color")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="camera capture fps (D405 color: 5/15/30/60/90 at 640x480)")
    ap.add_argument(
        "--publish-hz",
        type=float,
        default=15.0,
        help="JPEG encode/publish rate. MUST be an integer divisor of --fps "
        "(publishing is count-based decimation so the recorded camera timeline -- "
        "the master grid of the training dataset -- is uniform). 0 = every frame.",
    )
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument(
        "--report-every",
        type=float,
        default=10.0,
        help="seconds between achieved-rate / frame-gap-histogram log lines (0 = off)",
    )
    ap.add_argument("--mock", action="store_true", help="synthetic frames, no camera")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"[camera] REFUSING non-localhost host {args.host!r}", flush=True)
        sys.exit(2)
    try:
        decim = resolve_decimation(args.fps, args.publish_hz)
    except ValueError as exc:
        print(f"[camera] {exc}", flush=True)
        sys.exit(2)

    print(f"[camera] PID {os.getpid()} mock={args.mock} stream={args.stream} "
          f"{args.width}x{args.height}@{args.fps} -> publish every {decim} frame(s) "
          f"= {args.fps / decim:g} Hz (uniform)", flush=True)
    state = CameraState(stream=args.stream)
    reader = CameraReader(state, args)
    reader.start()
    stop = threading.Event()
    try:
        asyncio.run(_serve(state, args.host, args.port, stop))
    except KeyboardInterrupt:
        print("\n[camera] interrupted.", flush=True)
    finally:
        stop.set()
        reader.stop()
        # BEFORE join: `reader.join` can raise (and used to, via the shadowed
        # `Thread._stop`), which would make this summary unreachable on every
        # clean Ctrl-C / SIGTERM -- i.e. exactly when you want it.
        print(
            f"[camera] published {state.published} frames, "
            f"{state.capture_gaps} non-uniform gap(s), "
            f"{state.dropped_by_sdk} sensor-side drop(s)",
            flush=True,
        )
        reader.join(timeout=2.0)


if __name__ == "__main__":
    main()
