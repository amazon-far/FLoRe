# `gripper/` — i2rt DM4310 gripper control (sole CAN writer)

This package owns the **only** process that writes the CAN bus. It drives a
single YAM `linear_4310` (DM4310) gripper motor, mirrors the open/close state
into the viz, and accepts control intents from both the viser page and the
keyboard bridge. Everything else in the stack (browser, keyboard bridge) sends
**JSON intents only** — they never touch `can0`.

Two files:

- `holosoma_hand_pipeline/keyboard_gripper.py` — the gripper process: CAN
  writer + bidirectional encoder/control WebSocket.
- `keyboard_gripper_ws.py` — a device-level `evdev` keyboard bridge that turns
  HOLD `o`/`p` into JSON intents on the gripper socket.

## Running headless for data collection

For data collection, run the gripper process **headless** in `--web-control`
mode (no curses TTY, no `xset`):

```bash
PYTHONPATH=third_party/i2rt:gripper .venv/bin/python \
  -m holosoma_hand_pipeline.keyboard_gripper \
  --channel can0 --encoder-ws-port 8774 --web-control
```

`run_collection_viz.sh` launches exactly this (plus the keyboard bridge) and
brings `can0` up for you — see below.

### The `:8774` socket is BIDIRECTIONAL

The same localhost WebSocket on port `8774` does two things at once:

- **OUT** — publishes the live encoder reading every tick. Wire frame:
  `{"seq", "t", "open", "closed", "cmd", "actual"}`, where `open`/`closed` are
  the travel endpoints (rad) so the consumer can normalize to a 0..1 open
  fraction without hard-coding the range. The viz mirrors the URDF jaws and
  draws the live encoder chart from this.
- **IN** — accepts control intents on the same connection, from BOTH the viser
  page buttons/slider AND the `keyboard_gripper_ws.py` bridge. Accepted JSON
  object frames:
  - `{"cmd": "open"|"close"|"stop"}` — ramp toward an endpoint, or freeze.
  - `{"target_frac": 0.0..1.0}` — ramp to a fraction of the open stroke.

Because the control channel lives **inside** the process that already owns the
motor, the single-CAN-writer invariant holds: the browser and keyboard never
contend on the bus.

## `keyboard_gripper_ws.py` — evdev keyboard bridge

A device-level bridge: it reads `/dev/input/event*` directly and forwards key
state to the gripper socket (`--gripper-url ws://127.0.0.1:8774`).

- **HOLD `o` = open / HOLD `p` = close / release = stop.** Press/release map to
  `{"cmd": ...}` intents.
- **GLOBAL** — events come from the kernel input layer, so it works regardless
  of which window has focus (no terminal needs the keyboard).
- **Drops autorepeat.** evdev key events carry `value` (`0`=release, `1`=press,
  `2`=autorepeat); the bridge acts ONLY on press/release and IGNORES
  `value == 2`. This is what makes hold-to-move work with **no `xset` hack** and
  **no triple-character bug** (the failure mode of the old curses teleop).
- **Auto-detects every keyboard** exposing BOTH `KEY_O` and `KEY_P` (so it
  ignores mice / FIDO keys). Moving the 2-key macropad from a USB-C to a USB-A
  port — which gives it a new `/dev/input/eventN` and device name — needs **no
  config change**. It also **live re-scans** to (re)attach keyboards that are
  plugged in after start.

### Device access

`evdev` needs read access to `/dev/input/event*`. Either:

```bash
sudo usermod -aG input $USER   # then log out and back in
```

or run the bridge via `sudo`.

## ⚠️ Load-bearing failure: endpoint drift → stall → over-temp → "dead but green"

This is the single most important thing to understand about this gripper. It
caused a real, confusing field failure (browser + keyboard both "did nothing",
green status light, then a red flash), and three defenses now guard against it.
**Do not remove them.**

### What happens

The DM4310 uses an **absolute encoder that re-references on every power loss**,
so the open/closed travel endpoints **drift** after each motor power-cycle
(observed: the real rest moved to ~−1.26 rad while the stale defaults were
`open = 4.90` / `closed = 0.0`). If the teleop loop is launched with **stale
endpoints**, commanding "open" ramps the position command ~6 rad *past* the real
mechanical stop. At the teleop `kp = 3.0` the motor then holds **saturated torque
into the stall → motor over-temperature → the i2rt control thread raises and
dies** (`chain.running` flips `False`). After that the bridge keeps publishing a
*phantom* moving command while the encoder is frozen, and the WS keeps
broadcasting at 30 Hz — so the gripper looks **connected and healthy (green)
while the motor is dead.** That is the exact symptom: keys/buttons seem dead, the
light is green, then it red-flashes when the fault finally surfaces.

> **Note:** `third_party/i2rt` is byte-for-byte identical to upstream — the bug
> was never in the driver, only in launching it with the wrong endpoints.

### The three defenses (every launch, automatic)

1. **Startup self-test** (`gripper/gripper_selftest.py`, run by the launcher
   *before* the gripper bridge starts). It **auto-homes both hard stops** safely,
   discovers the **real current endpoints**, and **verifies one full
   open→close→open cycle**. On PASS the launcher feeds the discovered endpoints to
   the bridge (so you no longer hand-map them); on FAIL it **aborts the launch**
   with the motor powered down. See [self-test](#startup-self-test-auto-home--openclose)
   below.
2. **Overheat guard** (`_clamp_to_actual_lead`, `--max-lead 0.6` rad). Caps how
   far the position command may lead the *measured* encoder, so stall torque is
   bounded to `kp·max_lead` (≈ 1.8 N·m) regardless of how far a stale endpoint
   is. A stall can no longer cook the motor. Applied in both the headless and
   curses loops.
3. **Dead-motor detection.** `_I2rtGripper.motor_ok()` returns `chain.running`;
   the headless loop polls it, and on a fault it **stops commanding** (no phantom
   ramp), **de-energizes**, and sets `motor_ok = false` on the WS snapshot — so
   the viz shows **`✖ MOTOR FAULT`** instead of a fake-green light.

### Manual endpoint override (rarely needed now)

The self-test auto-discovers endpoints, so you normally do nothing. To **pin**
them explicitly (e.g. measured by hand, or a non-standard mechanism), set the
launcher env vars — the self-test then *verifies against* your values instead of
auto-homing them:

```bash
GRIPPER_OPEN=-1.25 GRIPPER_CLOSED=-6.53 ./run_collection_viz.sh
```

(or pass `--open` / `--closed` directly to `keyboard_gripper`). The endpoints
travel over the WS to the viz inside every encoder frame, so **no viz-side
change is needed**. To read the live endpoints by hand (zero-torque, safe):
`gripper/measure_gripper_endpoints.py --channel can0`.

## Startup self-test (auto-home + open/close)

`gripper/gripper_selftest.py` runs on every `./run_collection_viz.sh` launch,
**before** the gripper bridge, and **owns the CAN bus for its duration** (then
releases it — motor powered down — so it never runs concurrently with the bridge;
single-CAN-writer holds). It:

1. **Auto-homes** both hard stops by gently driving each way. Safe by
   construction: the same **overheat lead-clamp** (`--max-lead 0.5`) bounds torque,
   plus **stall detection**, a per-sweep **timeout**, and a **rotor/MOS
   temperature guard** that aborts on heat.
2. **Classifies** the two stops into `(open, closed)` (positive sim-rad = more
   open) and rejects an implausible stroke via a two-sided band
   (`--min-travel 4.5` … `--max-travel 6.5` rad; the real YAM stroke is ~5.1–5.3).
3. **Verifies** one full open→close→open cycle covers the discovered range.
4. Prints a machine-readable `[selftest] RESULT status=PASS open=.. closed=..`
   line the launcher parses to pick up the endpoints.

**Operational notes:**

- **The jaws MOVE during the self-test** to find the stops — keep hands clear and
  **nothing held in the jaws**.
- **On FAIL the whole launch aborts** (motor de-energized) with a re-measure hint.
  Failure means the jaws couldn't complete a full open/close — a stall, a dead
  motor, or something held. Clear the jaws and relaunch.
- **Skip it** with `SELFTEST=0 ./run_collection_viz.sh` (e.g. a known-good rig in
  a hurry). This removes the over-temp gate at startup — use sparingly.

### Stall detection (why it's subtle)

A stall is declared only when the command lead has been **saturated while the
encoder stays frozen for a sustained number of consecutive ticks**
(`--min-push-ticks 25` ≈ 0.25 s at 100 Hz), and the frozen counter **resets the
instant the encoder advances**. This is deliberately *not* an instantaneous
"lead saturated" test (which fires the tick full torque first arrives, misreading
a slow static-friction breakaway as a stop) and *not* a distance-from-start
arming guard (a real stop can be closer to the start than any arming distance →
it would never arm and time out — observed on hardware). Tying the grace to
"frozen-since" makes it local to each obstruction: a free-but-sticky jaw breaks
away within the window (counter resets), only a real stop stays frozen.

## Motor enable + warm-up (DM4310 idle auto-disable)

The DM4310 latches itself **disabled** (`0x0`) after ~8 s idle (its internal
timeout). A plain `motor_on` loses a ~2-tick transient race right after
re-enable (the motor still reports `disabled` for a couple of ticks). The
i2rt driver therefore does, synchronously, BEFORE the control thread starts:

1. an explicit `motor_off → clean_error → motor_on` enable cycle (the sequence
   that reliably clears a timeout-latched disabled state — a bare `motor_on`
   does not), then
2. **warm-up holds** that push past the post-enable transient until the motor
   reports solidly enabled.

So the control loop always starts against an already-enabled motor.

## Gains, motor config, defaults

- i2rt backend gains: `kp = 3.0` / `kd = 0.3` (direct MIT PD; tracks the full
  stroke to within ~0.05 rad — `kp=1` is too soft). Override with `--kp` /
  `--kd`.
- `motor_direction = -1` (positive motor rad = open).
- `--motor-id 7` (default), `--motor-type DM4310` (default).
- `--max-delta 0.075` rad/tick, `--rate-hz 50.0` — i.e. 3.75 rad/s, so the full ~5 rad
  stroke takes ~1.3 s while a key is held (was `0.05`/2.0 s; raised 1.5× for collection).
  One default serves every hand and mode — the launcher never passes it.
- `--max-lead 0.6` rad — the **overheat guard**: the position command can never
  lead the measured encoder by more than this, so a stall is bounded to
  `kp·max_lead` torque (see the failure section above). `≤ 0` disables it.

## `can0` bring-up

`run_collection_viz.sh` auto-brings-up `can0` before launching the gripper:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

Set `GRIPPER=0` to skip the gripper + keyboard bridges entirely (no CAN, no
motor).

> The optional `gmp_i2rt` PEAK backend needs a robot-side driver package that is **not
> part of this release** and is not used for data collection. A CANable on `can0`
> auto-selects the i2rt SocketCAN backend.

## Legacy curses teleop

Dropping `--web-control` runs the original interactive curses TTY teleop
(hold-to-move in the launching terminal, `xset` autorepeat tweak). It still
exists, but **`--web-control` is the only supported data-collection mode.**

## Tests

```bash
PYTHONPATH=third_party/i2rt:gripper .venv/bin/python -m pytest \
  gripper/holosoma_hand_pipeline/tests/test_keyboard_gripper.py \
  gripper/test_gripper_selftest.py -q
```

`test_keyboard_gripper.py` (65) covers the control loop, the overheat clamp
(`_clamp_to_actual_lead`), and dead-motor detection (`motor_ok`);
`test_gripper_selftest.py` (28) covers the self-test decision logic (stall
detection, endpoint classification band, full-stroke verify). The hardware
runner (auto-home + verify) is exercised by `run_collection_viz.sh` on launch.
