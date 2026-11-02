# VIVE tracker bridge

`vive_tracker_bridge.py` — reads a **single** VIVE tracker via
[`pysurvive`](https://github.com/cntools/libsurvive) and streams its raw,
yaw-aligned wrist pose as localhost-WebSocket JSON. It is one spoke in the
collection-viz hub & spoke: the viz (`viz/collection_viz.py`) is a thin WS client
that consumes this stream and drives the G1 right-hand mink IK from it.

The bridge is **deliberately minimal** — it reads one tracker, applies a single
fixed yaw rotation, and broadcasts the bare SE3. All calibration / home-anchoring
lives in the viz (it knows the robot's home `right_hand_site` pose and captures
the operator's reference at its "Recalibrate" button), so this process holds no
calibration state and never needs the two-hand L-R heading trick (impossible with
one tracker anyway).

## Output

Broadcast-only WebSocket at `ws://127.0.0.1:8772`. Each frame is one JSON object:

```json
{"seq": 1234, "t": 1718800000.123, "pos": [x, y, z], "quat": [w, x, y, z], "stale": false}
```

- `pos` — xyz metres in the robot-aligned world (libsurvive Z-up world rotated by
  `--yaw-deg` about +Z).
- `quat` — wxyz, same frame.
- `stale: true` once the tracker stops updating for longer than `--stale-timeout`
  (default `0.25` s; `0` disables).

The connection is broadcast-only; anything a client sends is drained and ignored.

## Environment

Runs under **`.venv-vive`** (py3.9) — the only reason for the py3.9 venv is that
`pysurvive` ships a `cp39`-only wheel. The pip `pysurvive` wheel **bundles
`libsurvive.so` + libusb**, so a default checkout needs **no from-source
libsurvive build**. (Optional source-build fallback: see
[`docs/HARDWARE_SETUP.md`](../../docs/HARDWARE_SETUP.md).)

## Hardware swap — which tracker is "right hand"

The user swaps trackers often, so the default mode is hands-off:

- **Leave `--right-serial` empty (recommended).** The bridge **auto-latches the
  first tracker it sees** and treats it as the right hand. A tracker swap then
  needs **no config change** — just plug in the new one and restart.
- **Pin a specific tracker** only when more than one tracker is powered and you
  must disambiguate: pass `--right-serial LHR-xxxx` (or set `RIGHT_SERIAL` on the
  launcher, which forwards it as `--right-serial`; the bridge also reads a
  `VIVE_UMI_RIGHT_SERIAL` env fallback directly). Trackers not matching the
  pinned serial are logged `-> UNMAPPED` and dropped.

### Discovering serials

Run with no `--left-serial` and watch the log; every tracker the context sees is
printed once with its serial:

```bash
# under .venv-vive
python bridges/tracker/vive_tracker_bridge.py          # no --left-serial
# [tracker] tracker T20 serial=LHR-75106789 -> left
```

Lighthouse base stations (names starting `LH`) are skipped as pose sources, so only
trackers appear on those lines — but they ARE counted (see below).

## Base stations: how many are in the solve

**libsurvive finds and uses every base station it sees, up to 16** — it adds each new
channel by itself (`survive.c::survive_get_bsd_idx` → `Adding lighthouse ch N (idx: I,
cnt: C)`). So going from two stations to three needs **no configuration**: mount it, give
it a channel no other station uses, and take one `FRESH_CALIB=1` wave. There is nothing
to "connect".

What the bridge adds is the **readout**, because a station that is powered but unseen used
to be completely silent — the trackers lock, the viz moves, takes record, and the only
symptom is glitchier takes in QC days later:

```
[tracker] cached calib: 2/3 solved (ch 2,0)  [lighthouse gen 2]
[tracker] cached calib:   slot0 ch2 id=e66022c0 (+1.12,-0.28,+0.64) [OK]
[tracker] cached calib: WARN: only 2 of 3 base stations are in the solve — …
[tracker] base station LH2 placed by the solver (lighthouses=3)
[tracker] streaming 3pt roles=['head', 'left', 'right'] lighthouses=3/3 expected
[tracker] streaming 5pt roles=['head', 'left', 'left_foot', 'right', 'right_foot'] lighthouses=4/4 expected
```

`--expect-lighthouses N` (the launcher's `EXPECT_LIGHTHOUSES`, default `auto` = whatever
the cached calibration already has, floor 2) turns a shortfall into a repeated warning and
a `VIVE base stations` row on the launch board. It is **a warning only** — the rig tracks
fine on fewer stations, it just loses that station's coverage.

The count comes from the `LH*` objects `SurviveSimpleContext` hands back: a lighthouse only
gets pose updates once the global solver has **placed** it, so "seen here" == "in the
solve". Reading `activeLighthouses`/`bsd[]` out of `SurviveContext` by ctypes would be more
direct and is **deliberately not done**: `survive_simple_get_ctx` is exported by the
bundled `.so`, but the wheel's generated `struct_SurviveContext` layout does not match that
build — a probe read `activeLighthouses == 0` while two lighthouses were streaming, with
"station ids" that were float bit patterns changing every poll.

[`lighthouse_status.py`](lighthouse_status.py) is the offline half (stdlib only, no
pysurvive, no dongle) and also runs standalone:

```bash
python3 bridges/tracker/lighthouse_status.py --log data/logs/tracker.log --expect 3
```

It parses libsurvive's `~/.config/libsurvive/config.json` — which despite the name is
**not valid JSON** — plus a tracker log, and names the failure instead of the symptom:

| It reports | Because |
|---|---|
| `only N of M in the solve` | the missing station is off, blocked, or on a taken channel |
| `SEEN but never PLACED` (OOTX yes, position no) | it decoded but the solver never located it — wave through its coverage |
| `channel C reported 2 different station ids` | **two stations share a channel**; libsurvive maps channel→one slot, so they overwrite each other and *reset* the solved pose each flip. Tracking gets WORSE than with two stations |
| `channel C answered with id X but the calibration has Y` | same collision, caught from a single decode (libsurvive prints that line only when the id on a slot changes) |
| `DISABLED by lighthouse-N-disable` | fully calibrated and silently ignored by the poser |
| `in the calibration but did NOT report this launch` | a station that was there last time and isn't now |

A station counts as usable only when **both** `OOTXSet` (factory blob decoded off the sync
bits) and `PositionSet` (solver placed it) are true.

## Single-tracker yaw

With only one tracker there is **no L-R vector** to auto-derive heading, so you
must tell the bridge which real-world direction maps to robot **+x** via
`--yaw-deg`. The bridge applies a pure rotation about world +Z (gravity / Z-up is
preserved in both frames). Because the viz mirrors the live displacement/rotation
relative to a captured reference (the "Recalibrate" anchor), this absolute yaw
only sets the **nominal forward** — get it roughly right and recalibrate in the
viz.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | WS bind host. Localhost only; a routable host exits with code 2. |
| `--port` | `8772` | WS bind port. |
| `--rate-hz` | `90.0` | WS broadcast rate (Hz). |
| `--right-serial` | none → first-seen | LHR serial of the right-hand tracker; empty auto-latches the first tracker. |
| `--yaw-deg` | `0.0` | Yaw (deg) mapping libsurvive Z-up world → robot world (+x forward). |
| `--stale-timeout` | `0.25` | Mark frames `stale` if the tracker stops updating for > this many seconds; `0` disables. |
| `--fresh` | off | Force libsurvive recalibration (wave the tracker ~30 s). |
| `--expect-lighthouses` | `0` (report only) | How many base stations SHOULD end up in the solve. Warns while fewer are placed; never gates a pose. |
| `--no-fresh-ootx` | off (blobs re-read) | With `--fresh`, re-solve base-station POSITIONS but keep each station's cached factory blob. Use when a `--fresh` launch hangs before any pose with `OOTX Decoder: Bad sync bit`. |

### The OOTX phase — why a `--fresh` launch can look hung

`--fresh` bundles **two** libsurvive resets that are not the same thing:

| flag | clears | needed when |
|---|---|---|
| `--force-calibrate 1` | the base-station **poses** → re-solve the room geometry | you moved or added a station (this is what the wave is for) |
| `--force-ootx 1` | each station's decoded **factory blob** → re-read it off the sync bits | a station was **swapped onto a channel another one used** — the only thing that re-reads a station's id |

The blob is burned into the station hardware and does **not** change when you move it, but
re-reading it is one bit per sweep and needs clean, uninterrupted line of sight. A station
the pucks cannot see cleanly logs `(N) OOTX Decoder: Bad sync bit` and retries — and
**libsurvive produces no pose at all until every attached decoder finishes**, so the launch
sits at `waiting for first Global solve...` looking dead. Observed on this rig: channels 2
and 3 decoded in ~1 min, channel 0 took ~5 more.

The wave gate now shows the phase (`decoding base stations: ootx 2/3 (ch 0 pending, 16 sync
err)`), so "be patient" is distinguishable from "one station has no line of sight". Hold the
pucks **up, facing the stations** while it decodes. If a channel will not decode, relaunch
with `FRESH_OOTX=0` (`--no-fresh-ootx`): positions still get re-solved, blobs come from the
cache. The trade-off is the one in the table — a swapped station would go unnoticed.

## Run

```bash
# under .venv-vive
# hands-off: auto-latch the first tracker, +x = real-world forward
python bridges/tracker/vive_tracker_bridge.py --yaw-deg 0

# pin a specific tracker
python bridges/tracker/vive_tracker_bridge.py --left-serial LHR-75106789 --yaw-deg 0
```

Normally launched as part of the full stack by `./run_collection_viz.sh` (which
runs it under `.venv-vive`).

## Pairing a NEW puck to its dongle (one-time, needs SteamVR's `lighthouse_console`)

**Terminal path (verified 2026-09-09, `E296900B95 <-> LHR-C68495D2`):** with the bridge
stopped and ONE new dongle plugged,

```bash
bridges/tracker/pair_puck.sh            # or: pair_puck.sh <DONGLE_SERIAL> [window_s]
```

then hold the puck's button ~2 s until it blinks blue. The script drives SteamVR's
`tools/lighthouse/bin/linux64/lighthouse_console` (`serial <dongle>` + `pair`, an
undocumented interactive command; the `/serial X /pair` flag form segfaults) and finishes
with a 15 s `--list-serials` so the new `UNASSIGNED` serial prints. SteamVR itself need not
be running; if it is, note the Linux build only sees dongles plugged BEFORE vrserver started,
which is why its pairing window may show no "HTC Vive Tracker" entry at all.

**GUI path** (same thing through SteamVR's window):

A Vive Tracker is **not** paired to the dongle in its box: HTC's own guide says "when you turn
the tracker on for the first time, it needs to be paired with the dongle", and libsurvive
cannot do it (collabora/libsurvive#162 — "pairing new controllers isn't something libsurvive
does"). The pairing is stored in the dongle + puck, not the host, so do it once on ANY machine
with SteamVR and move the dongle back to the rig:

1. Plug the dongle in **before** launching SteamVR (the Linux build does not pick up dongles
   plugged later). No headset: set `"requireHmd": false`, `"forcedDriver": "null"`,
   `"activateMultipleDrivers": true` under `steamvr` and `"enable": true` under
   `driver_null` in `steamvr.vrsettings`.
2. SteamVR ▸ Devices ▸ Pair Controller ▸ *I want to pair a different type of controller* ▸
   **HTC Vive Tracker**.
3. Hold the puck's button ~2 s → **blinking blue** (pairing mode). Wait for **green**.
   Pairing idles out after 30 s and the puck powers off — brief press to turn it back on.
4. One puck at a time. Quit SteamVR before running the bridge: both claim the dongles.

LED legend (HTC): green = on & linked · solid blue = linking · blinking blue = pairing ·
blinking red = low battery · white/orange = charging. **A brief press turns the puck on;
holding it ~2 s is pairing mode, not power-on.**
