"""Bottom-dock dashboard wrapper for the Bota F/T viser visualizer.

Why this exists: viser 1.0.30's GUI is a fixed right-side panel - it has no
bottom dock and no horizontal grid. kimodo gets its bottom timeline bar from a
*custom viser fork* (nv-tlabs/kimodo-viser), which we don't want to maintain.
Instead we do the other half of kimodo's approach - a custom front-end that
WRAPS viser - without forking it:

    +-------------------------------------------+
    |                                           |
    |     <iframe> viser 3D scene  (top 2/3)    |   arrows + orientation triad
    |                                           |   + the viser control panel
    +---------------------------+---------------+
    | Force   ~~~~~~~~~~~~~~~~~  | Accel ~~~~~~  |   bottom 1/3:
    +---------------------------+---------------+   left 2/3 = Force / Torque
    | Torque  ~~~~~~~~~~~~~~~~~  | Gyro  ~~~~~~  |   right 1/3 = Accel / Gyro
    +---------------------------+---------------+

viser serves plain HTML with no X-Frame-Options/CSP, so it embeds in an iframe.
The charts are drawn on plain <canvas> (no CDN / no external JS - works fully
offline) and fed live over a websocket that streams the rolling history buffer
as JSON. This module owns ONLY the wrapper page + data socket; the 3D scene,
reader thread and SensorState come from bota_ft_viz.

Served by a single ``websockets`` server (already a viser dependency):
  GET  /            -> the wrapper HTML
  WS   /ws          -> periodic JSON frames of the rolling history
"""

from __future__ import annotations

import asyncio
import json
import threading

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from bota_ft_viz import SensorState

# Axis line colors (CSS) - match the in-scene arrow / panel-chart convention.
_AXIS_CSS = ("#dc3232", "#28b446", "#3c6ef0")


def _page_html(viser_url: str) -> str:
    """The wrapper page: iframe (top 2/3) + 4 canvas charts (bottom 1/3)."""
    return _HTML_TEMPLATE.replace("__VISER_URL__", viser_url)


async def _ws_sender(conn, state: SensorState, hz: float, mode: str) -> None:
    """Stream rolling-history + status (incl. recalibration phase) to a client."""
    period = 1.0 / hz
    last_n = -1
    while True:
        with state.lock:
            n = state.n_frames
            exported = state.history.export()
            conn_ok = state.connected
            err = state.last_error
            rate = state.rate_hz
            temp = state.temperature
            recal = {"phase": state.recal_phase, "elapsed": state.recal_elapsed, "message": state.recal_message}
            comp_avail = state.compensation_available
            if comp_avail:
                mode = "COMPENSATED" if state.show_compensated else "RAW"
        status = {"connected": conn_ok, "error": err, "rate": rate, "temp": temp, "mode": mode,
                  "recal": recal, "comp_available": comp_avail}
        # During recalibration there's no new chart data, but we still must push
        # status so the overlay updates - so send on new data OR while recal active.
        if (exported is not None and n != last_n) or recal["phase"] not in ("idle", ""):
            last_n = n
            if exported is not None:
                x, ch = exported
                msg = {
                    "x": x.tolist(),
                    "force": [c.tolist() for c in ch["force"]],
                    "torque": [c.tolist() for c in ch["torque"]],
                    "acc": [c.tolist() for c in ch["acc"]],
                    "gyro": [c.tolist() for c in ch["gyro"]],
                    "status": status,
                }
            else:
                msg = {"status": status}
            await conn.send(json.dumps(msg))
        await asyncio.sleep(period)


async def _ws_receiver(conn, state: SensorState) -> None:
    """Receive commands from a client (e.g. {"cmd":"recalibrate"})."""
    async for raw in conn:
        try:
            cmd = json.loads(raw).get("cmd")
        except Exception:  # noqa: BLE001
            continue
        if cmd == "recalibrate":
            with state.lock:
                if state.compensation_available and state.recal_phase in ("idle", "done", "failed", ""):
                    state.recal_request = True
                    state.recal_phase = "countdown"
                    state.recal_message = "starting..."
        elif cmd == "recal_ack":
            # Browser dismissed the done/failed overlay - clear the phase so it
            # doesn't re-show for this or a new client.
            with state.lock:
                if state.recal_phase in ("done", "failed"):
                    state.recal_phase = "idle"
                    state.recal_message = ""


async def _ws_handler(conn, state: SensorState, hz: float, mode: str) -> None:
    """Run sender + receiver concurrently for one client; exit when either ends."""
    sender = asyncio.ensure_future(_ws_sender(conn, state, hz, mode))
    receiver = asyncio.ensure_future(_ws_receiver(conn, state))
    try:
        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sender.cancel()
        receiver.cancel()


class DashboardServer(threading.Thread):
    """Runs the wrapper HTTP+WS server on its own asyncio loop / thread."""

    def __init__(
        self, state: SensorState, host: str, port: int, viser_url: str, stream_hz: float = 20.0, mode: str = "RAW"
    ):
        super().__init__(daemon=True)
        self.state = state
        self.host = host
        self.port = port
        self.viser_url = viser_url
        self.stream_hz = stream_hz
        self.mode = mode
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._html = _page_html(viser_url).encode("utf-8")

    def _process_request(self, conn, request):
        """Serve the wrapper HTML on a plain GET to '/'; let '/ws' upgrade."""
        if request.path == "/ws":
            return None  # proceed with the websocket handshake
        headers = Headers()
        headers["Content-Type"] = "text/html; charset=utf-8"
        headers["Content-Length"] = str(len(self._html))
        return Response(200, "OK", headers, self._html)

    async def _serve(self) -> None:
        async def handler(conn):
            await _ws_handler(conn, self.state, self.stream_hz, self.mode)

        self._stop_event = asyncio.Event()
        # Hold the server open until stop() sets the event; exiting the
        # ``async with`` then gracefully closes the listening socket (vs.
        # stopping the loop out from under it, which leaks the socket and
        # raises inside run_until_complete).
        async with serve(handler, self.host, self.port, process_request=self._process_request):
            await self._stop_event.wait()

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            self._loop.close()

    def stop(self) -> None:
        loop, ev = self._loop, self._stop_event
        if loop is not None and ev is not None:
            loop.call_soon_threadsafe(ev.set)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Bota F/T Sensor</title>
<style>
  html, body { margin: 0; height: 100%; background: #f4f4f6; font-family: system-ui, sans-serif; }
  #wrap { display: flex; flex-direction: column; height: 100vh; }
  #scene { flex: 2 1 0; border: 0; width: 100%; }
  #dock { flex: 1 1 0; display: flex; border-top: 2px solid #d0d0d8; min-height: 0; }
  #leftcol { flex: 2 1 0; display: flex; flex-direction: column; min-width: 0; }
  #rightcol { flex: 1 1 0; display: flex; flex-direction: column; min-width: 0; border-left: 1px solid #d0d0d8; }
  .cell { flex: 1 1 0; display: flex; flex-direction: column; padding: 4px 8px; min-height: 0;
          border-bottom: 1px solid #e2e2e8; }
  .cell:last-child { border-bottom: 0; }
  .hdr { font-size: 12px; font-weight: 600; color: #444; display: flex; gap: 12px; align-items: baseline; }
  .lg { font-weight: 400; font-size: 11px; }
  .lg b { font-weight: 700; }
  canvas { flex: 1 1 0; width: 100%; min-height: 0; }
  #status { position: fixed; top: 6px; left: 6px; z-index: 10; font-size: 11px; color: #555;
            background: rgba(255,255,255,.8); padding: 2px 8px; border-radius: 4px; }
  #yreset { position: fixed; top: 4px; right: 8px; z-index: 10; font-size: 11px; cursor: pointer;
            background: #e8730c; color: #fff; border: 0; padding: 3px 10px; border-radius: 4px; }
  #recal { position: fixed; top: 4px; right: 108px; z-index: 10; font-size: 11px; cursor: pointer;
           background: #2563eb; color: #fff; border: 0; padding: 3px 10px; border-radius: 4px; }
  #recal:disabled { background: #9ca3af; cursor: default; }
  #overlay { position: fixed; inset: 0; z-index: 100; display: none; flex-direction: column;
             align-items: center; justify-content: center; background: rgba(10,12,20,.92); color: #fff;
             text-align: center; font-family: system-ui, sans-serif; }
  #overlay.show { display: flex; }
  #ov_big { font-size: 18vh; font-weight: 800; line-height: 1; }
  #ov_msg { font-size: 3.2vh; margin-top: 3vh; max-width: 80vw; }
  #ov_sub { font-size: 2vh; margin-top: 2vh; color: #9fb3d8; }
  #ov_close { margin-top: 4vh; font-size: 2.2vh; cursor: pointer; background: #2563eb; color: #fff;
              border: 0; padding: 8px 22px; border-radius: 6px; display: none; }
</style>
</head>
<body>
<div id="wrap">
  <iframe id="scene" src="__VISER_URL__"></iframe>
  <div id="dock">
    <div id="leftcol">
      <div class="cell"><div class="hdr">Force (N) <span class="lg" id="lg_force"></span></div>
        <canvas data-ch="force"></canvas></div>
      <div class="cell"><div class="hdr">Torque (Nm) <span class="lg" id="lg_torque"></span></div>
        <canvas data-ch="torque"></canvas></div>
    </div>
    <div id="rightcol">
      <div class="cell"><div class="hdr">Accel (m/s^2) <span class="lg" id="lg_acc"></span></div>
        <canvas data-ch="acc"></canvas></div>
      <div class="cell"><div class="hdr">Gyro (rad/s) <span class="lg" id="lg_gyro"></span></div>
        <canvas data-ch="gyro"></canvas></div>
    </div>
  </div>
</div>
<div id="status">connecting...</div>
<button id="recal" title="Re-estimate the tool payload, then resume compensation">Recalibrate</button>
<button id="yreset" title="Reset the auto-scaled Y ranges back to their floors">Reset Y range</button>
<div id="overlay">
  <div id="ov_big"></div>
  <div id="ov_msg"></div>
  <div id="ov_sub"></div>
  <button id="ov_close">Close</button>
</div>
<script>
const COLORS = ["#dc3232", "#28b446", "#3c6ef0"];
const LABELS = {force:["Fx","Fy","Fz"], torque:["Tx","Ty","Tz"], acc:["Ax","Ay","Az"], gyro:["Gx","Gy","Gz"]};
// Per-channel persistent y half-range (peak-hold): grows with the largest
// magnitude ever seen, never shrinks, so the symmetric +/-range only ratchets
// UP when force gets bigger and stays put when it falls back. FLOORS are the
// minimum half-range per channel (so a quiet signal isn't a degenerate 0).
const FLOORS = {force: 1.0, torque: 0.05, acc: 1.0, gyro: 0.05};
const peaks = {force: FLOORS.force, torque: FLOORS.torque, acc: FLOORS.acc, gyro: FLOORS.gyro};
let latest = null;

const canvases = {};
document.querySelectorAll("canvas[data-ch]").forEach(c => { canvases[c.dataset.ch] = c; });

// fill legends
for (const ch of Object.keys(LABELS)) {
  const el = document.getElementById("lg_" + ch);
  if (el) el.innerHTML = LABELS[ch].map((n,i)=>`<b style="color:${COLORS[i]}">${n}</b>`).join("  ");
}

function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w*dpr || c.height !== h*dpr) { c.width = w*dpr; c.height = h*dpr; }
  return dpr;
}

function drawChart(c, x, series, chName) {
  const dpr = fitCanvas(c);
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  if (!x || x.length < 2) return;
  const padL = 46*dpr, padR = 6*dpr, padT = 6*dpr, padB = 16*dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  // SYMMETRIC peak-hold y-range: track the largest |value| across all series,
  // ratchet the per-channel peak UP (never down), then plot +/-peak centered on
  // 0. Result: bigger force pushes the range out symmetrically; falling back
  // leaves the range where it was (no jitter).
  let amax = 0;
  for (const s of series) for (const v of s) { const a = Math.abs(v); if (a > amax) amax = a; }
  if (peaks[chName] === undefined) peaks[chName] = (FLOORS[chName] || 1);
  // grow with 8% headroom so the trace doesn't touch the top edge
  peaks[chName] = Math.max(peaks[chName], amax * 1.08, FLOORS[chName] || 1);
  const ymax = peaks[chName], ymin = -ymax;
  const xmin = x[0], xmax = x[x.length-1];
  const sx = v => padL + (v - xmin)/(xmax - xmin || 1) * plotW;
  const sy = v => padT + (1 - (v - ymin)/(ymax - ymin || 1)) * plotH;
  // grid + axes
  ctx.strokeStyle = "#e6e6ec"; ctx.lineWidth = 1*dpr; ctx.fillStyle = "#888";
  ctx.font = (10*dpr)+"px system-ui"; ctx.textBaseline = "middle";
  for (let g=0; g<=4; g++) {
    const yv = ymin + (ymax-ymin)*g/4, yy = sy(yv);
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W-padR, yy); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(yv.toFixed(2), padL-4*dpr, yy);
  }
  // zero line
  if (ymin < 0 && ymax > 0) { ctx.strokeStyle="#c8c8d0"; ctx.beginPath(); ctx.moveTo(padL,sy(0)); ctx.lineTo(W-padR,sy(0)); ctx.stroke(); }
  // x ticks (seconds)
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  for (let g=0; g<=4; g++) {
    const xv = xmin + (xmax-xmin)*g/4;
    ctx.fillText(xv.toFixed(1)+"s", sx(xv), H-2*dpr);
  }
  // series lines
  ctx.lineWidth = 1.5*dpr;
  for (let si=0; si<series.length; si++) {
    const s = series[si];
    ctx.strokeStyle = COLORS[si]; ctx.beginPath();
    for (let i=0; i<s.length; i++) { const px=sx(x[i]), py=sy(s[i]); i? ctx.lineTo(px,py):ctx.moveTo(px,py); }
    ctx.stroke();
  }
}

document.getElementById("yreset").addEventListener("click", () => {
  for (const ch of Object.keys(peaks)) peaks[ch] = FLOORS[ch] || 1;
});

function render() {
  if (latest && latest.x) {
    for (const ch of ["force","torque","acc","gyro"]) {
      drawChart(canvases[ch], latest.x, latest[ch], ch);
    }
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ---- Recalibration: button -> command, overlay driven by status.recal ----
let sock = null;
const ov = document.getElementById("overlay");
const ovBig = document.getElementById("ov_big");
const ovMsg = document.getElementById("ov_msg");
const ovSub = document.getElementById("ov_sub");
const ovClose = document.getElementById("ov_close");
const recalBtn = document.getElementById("recal");
let recalActive = false;

recalBtn.addEventListener("click", () => {
  if (sock && sock.readyState === 1) {
    sock.send(JSON.stringify({cmd: "recalibrate"}));
    recalActive = true;
    recalBtn.disabled = true;
  }
});
ovClose.addEventListener("click", () => {
  ov.classList.remove("show"); recalActive = false;
  if (sock && sock.readyState === 1) sock.send(JSON.stringify({cmd: "recal_ack"}));
});

function updateOverlay(recal) {
  if (!recal) return;
  const ph = recal.phase || "idle";
  if (ph === "countdown") {
    ov.classList.add("show"); ovClose.style.display = "none";
    ovBig.textContent = Math.ceil(recal.elapsed || 0);
    ovMsg.textContent = "Recalibration starting - lift the tool, keep it free of contact";
    ovSub.textContent = recal.message || "";
  } else if (ph === "estimating") {
    ov.classList.add("show"); ovClose.style.display = "none";
    ovBig.textContent = "↻";
    ovMsg.textContent = "ROTATE the tool slowly through many orientations";
    ovSub.textContent = recal.message || "";
  } else if (ph === "done") {
    ov.classList.add("show"); ovBig.textContent = "✓";
    ovMsg.textContent = "Recalibration complete";
    ovSub.textContent = "compensation reloaded";
    ovClose.style.display = "inline-block";
    recalBtn.disabled = false;
  } else if (ph === "failed") {
    ov.classList.add("show"); ovBig.textContent = "✕";
    ovMsg.textContent = "Recalibration failed";
    ovSub.textContent = recal.message || "";
    ovClose.style.display = "inline-block";
    recalBtn.disabled = false;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(proto + "://" + location.host + "/ws");
  sock = ws;
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.x) latest = msg;  // only swap chart data when present (status-only msgs don't blank charts)
    const s = msg.status || {};
    document.getElementById("status").textContent =
      (s.mode || "RAW") + "   " +
      (s.connected ? "connected" : (s.error || "disconnected")) +
      "   " + (s.rate||0).toFixed(0) + " Hz   " + (s.temp||0).toFixed(1) + " C";
    // Recal button only meaningful when a compensator is running.
    if (!recalActive) recalBtn.disabled = !s.comp_available;
    updateOverlay(s.recal);
  };
  ws.onclose = () => { sock = null; setTimeout(connect, 1000); };
}
connect();
</script>
</body>
</html>
"""
