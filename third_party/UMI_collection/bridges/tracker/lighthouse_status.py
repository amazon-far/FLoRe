#!/usr/bin/env python3
"""How many VIVE base stations is libsurvive ACTUALLY solving with? (read-only)

WHY THIS EXISTS
---------------
libsurvive already supports up to ``NUM_GEN2_LIGHTHOUSES`` (16) base stations and
adds each one **by itself** the moment its channel shows up in the sync stream
(``survive.c::survive_get_bsd_idx`` -> "Adding lighthouse ch N (idx: I, cnt: C)").
So going from two stations to three needs NO code change to "connect" the new one:
put it up, give it a free channel, and run one FRESH calibration.

What was missing is the other half — nothing in this repo ever reported how many
stations are in the solve. Every operator-facing string assumed exactly two ("so
libsurvive re-solves BOTH base stations"), the wave gate only checked that the
TRACKERS were live, and the recorded session kept no provenance of it. A station
that is powered but unseen (wrong channel, no line of sight, dead PSU) therefore
looks EXACTLY like a good launch: the trackers lock, the viz moves, the take
records — you just quietly get 2-station coverage with its occlusion dropouts,
and the only symptom is glitchier takes days later in QC.

This module is the missing readout. It is pure parsing over two artifacts that
libsurvive maintains on its own, so it needs neither pysurvive nor the dongle and
can run any time, in any interpreter:

  1. ``~/.config/libsurvive/config.json`` — the persisted per-station record
     (id, channel, OOTX decoded, position solved, pose, variance). NOTE: despite
     the name this is NOT valid JSON (bare newline-separated members, no commas
     between blocks), which is why ``json.load`` on it raises "Extra data" and why
     the parser below is line-oriented.
  2. the tracker bridge's log — libsurvive's own stdout, which names every channel
     it saw, every OOTX id it decoded and every lighthouse it added THIS launch.

NOT USED ON PURPOSE: reading ``activeLighthouses``/``bsd[]`` out of the live
``SurviveContext`` via ctypes. ``survive_simple_get_ctx`` is exported by the
bundled ``libsurvive.so``, but the wheel's generated ``struct_SurviveContext``
layout does not match that build — a probe read ``activeLighthouses == 0`` while
two lighthouses were streaming, and "station ids" that were float bit patterns
changing every poll. A silently mis-offset struct is worse than no readout, so
the live count comes from the stable public API instead (the ``LH*`` objects
``SurviveSimpleContext`` hands back, counted in ``vive_tracker_bridge.py``).

WHAT COUNTS AS "IN THE SOLVE"
-----------------------------
A station contributes to tracking only once BOTH halves are true:
``OOTXSet`` (its factory calibration blob was decoded off the sync bits) AND
``PositionSet`` (the global solver actually placed it in the world). A station
can sit at OOTX-yes / position-no indefinitely — that is the "seen but never
placed, wave more" case, and it is invisible without this readout.

...and it must occupy a slot BELOW the number of stations in the file. libsurvive
sets ``activeLighthouses`` to the COUNT of entries it read, never to
``max(slot)+1``, and then every poser plus pysurvive's own object list iterates
``i < activeLighthouses``. So a file holding slots 0, 1, 3 hides slot 3 from the
solver entirely and offers the empty slot 2 as a station (it enumerates as
``LHB-0``). Gaps arise unaided: ``config_save`` writes only slots with
``OOTXSet``, so one station that failed to decode leaves a hole every later
launch inherits. ``diagnose`` reports it; the repair is renumbering the
``lighthouseN`` keys to be contiguous, which keeps the cached blobs.

Run:
    python3 bridges/tracker/lighthouse_status.py                    # cached calib
    python3 bridges/tracker/lighthouse_status.py --log data/logs/tracker.log
    python3 bridges/tracker/lighthouse_status.py --expect 3         # gate on a count

Exit codes (0 unless --strict, so this can never break a launch by itself):
    0  reported fine (or problems found but --strict not given)
    1  --strict and at least one problem (missing/unsolved/collided station)
    2  no calibration file to read
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

#: libsurvive's own default config path (``DEFAULT_CONFIG_PATH`` in survive.c).
#: Overridable per launch with libsurvive's ``--configfile``, which is what the
#: probe scripts use so they can never disturb the rig's real calibration.
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/libsurvive/config.json")

#: Hard ceiling from ``survive_types.h::NUM_GEN2_LIGHTHOUSES``. Only used to
#: sanity-bound a parsed slot index; we never need to iterate it.
MAX_LIGHTHOUSES = 16


@dataclass
class Station:
    """One base station as libsurvive persisted it."""

    slot: int                      # bsd[] index (config key "lighthouseN")
    channel: int | None = None     # libsurvive "mode" == the LH2 channel
    station_id: int | None = None  # unique per physical station (from OOTX)
    ootx_set: bool = False
    position_set: bool = False
    disabled: bool = False         # top-level "lighthouse-N-disable"
    pose: list[float] = field(default_factory=list)      # 7: xyz + wxyz
    variance: list[float] = field(default_factory=list)  # 6

    @property
    def solved(self) -> bool:
        """True when this station can actually contribute to a tracker pose."""
        return self.ootx_set and self.position_set and not self.disabled

    @property
    def id_hex(self) -> str:
        return "?" if self.station_id is None else f"{self.station_id & 0xFFFFFFFF:08x}"

    def describe(self) -> str:
        pos = (f"({self.pose[0]:+.2f},{self.pose[1]:+.2f},{self.pose[2]:+.2f})"
               if len(self.pose) >= 3 else "(no pose)")
        flags = "OK" if self.solved else ",".join(
            f for f in (
                "" if self.ootx_set else "no-OOTX",
                "" if self.position_set else "NOT-PLACED",
                "DISABLED" if self.disabled else "",
            ) if f
        )
        ch = "?" if self.channel is None else self.channel
        return f"slot{self.slot} ch{ch} id={self.id_hex} {pos} [{flags}]"


@dataclass
class LogSighting:
    """What libsurvive's stdout says it saw during ONE launch."""

    channels: set[int] = field(default_factory=set)
    #: channel -> every distinct station id decoded on it. >1 id on one channel is
    #: the fingerprint of two physical stations sharing a channel.
    ids_by_channel: dict[int, set[int]] = field(default_factory=dict)
    #: ("Adding lighthouse ch N (idx: I, cnt: C)") — a channel appearing at RUNTIME,
    #: i.e. a station that was not in the cached calibration. Seeing this is how you
    #: know a newly-mounted station reached the solver at all.
    added: list[tuple[int, int | None, int | None]] = field(default_factory=list)
    #: lighthouse indices the global solver ran for ("Global solve ... for N")
    solved_indices: set[int] = field(default_factory=set)
    disabled_slots: set[int] = field(default_factory=set)
    #: our own bridge heartbeat ("lighthouses=N"), newest last
    bridge_counts: list[int] = field(default_factory=list)
    #: channels libsurvive attached an OOTX decoder to this launch, i.e. the ones whose
    #: factory blob it does NOT have and must read off the sync bits before ANY pose can
    #: be computed. This is the phase a `--force-ootx` launch sits in, and it was
    #: previously invisible: the wave gate showed "waiting for first Global solve..."
    #: with no hint that one channel's decode was failing.
    ootx_attached: set[int] = field(default_factory=set)
    #: "OOTX Decoder: Bad sync bit" per channel — a corrupt frame, restarted. A few are
    #: normal; dozens on one channel means no clean line of sight from the puck the
    #: decoder was attached to.
    bad_sync_by_channel: dict[int, int] = field(default_factory=dict)

    @property
    def live_count(self) -> int:
        """Best available live count: the bridge heartbeat, else solver indices."""
        if self.bridge_counts:
            return self.bridge_counts[-1]
        return len(self.solved_indices or self.channels)

    @property
    def known_count(self) -> int:
        """How many base stations libsurvive knew about, at the most it ever knew.

        Read off its own ``cnt:`` (see :attr:`added`), falling back to the number of
        distinct channels seen. This is a COUNT OF THE ROOM, not of the solve: a station
        whose OOTX has not decoded, or that is not in the cached calibration, is counted
        here and cannot appear in ``solved``.
        """
        counts = [c for _ch, _idx, c in self.added if c]
        return max(counts) if counts else len(self.channels)

    @property
    def ootx_pending(self) -> set[int]:
        """Channels still waiting on an OOTX decode (attached but never decoded)."""
        return self.ootx_attached - set(self.ids_by_channel)


# --- libsurvive config parsing -------------------------------------------------
# The file is JSON-ISH: quoted keys, quoted scalar values, bracketed arrays of
# quoted numbers, and "lighthouseN":{ ... } blocks separated by nothing at all.
# Line-oriented on purpose — a real JSON parser rejects it outright.
_BLOCK_OPEN = re.compile(r'^"(lighthouse(\d+))"\s*:\s*\{')
_SCALAR = re.compile(r'^"([^"]+)"\s*:\s*"([^"]*)"\s*,?\s*$')
_ARRAY = re.compile(r'^"([^"]+)"\s*:\s*\[(.*)\]\s*,?\s*$')
_DISABLE_KEY = re.compile(r"^lighthouse-(\d+)-disable$")


def _as_int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _as_floats(body: str) -> list[float]:
    out = []
    for tok in re.findall(r'"([^"]*)"', body) or body.split(","):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def parse_libsurvive_config(text: str) -> tuple[list[Station], dict[str, str]]:
    """Parse libsurvive's config file into stations + top-level scalars.

    Tolerant by design: unknown keys are ignored and a malformed line is skipped
    rather than fatal, because this is a *diagnostic* — refusing to report because
    a future libsurvive added a field would defeat the whole point.
    """
    stations: dict[int, Station] = {}
    top: dict[str, str] = {}
    cur: Station | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _BLOCK_OPEN.match(line)
        if m:
            slot = int(m.group(2))
            cur = stations.setdefault(slot, Station(slot=slot))
            continue
        if line.startswith("}"):
            cur = None
            continue
        m = _ARRAY.match(line)
        if m:
            key, body = m.group(1), m.group(2)
            if cur is not None:
                if key == "pose":
                    cur.pose = _as_floats(body)
                elif key == "variance":
                    cur.variance = _as_floats(body)
            continue
        m = _SCALAR.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if cur is None:
            top[key] = val
            continue
        if key == "mode":
            cur.channel = _as_int(val)
        elif key == "id":
            cur.station_id = _as_int(val)
        elif key == "OOTXSet":
            cur.ootx_set = val not in ("0", "", "false")
        elif key == "PositionSet":
            cur.position_set = val not in ("0", "", "false")
        elif key == "index":
            idx = _as_int(val)
            if idx is not None and 0 <= idx < MAX_LIGHTHOUSES:
                cur.slot = idx
    # "lighthouse-N-disable" is a TOP-LEVEL knob, not part of the block, so it has
    # to be applied after the fact. A disabled station is the nastiest of the
    # failure modes: fully calibrated, listed, and silently ignored by the poser.
    for key, val in top.items():
        m = _DISABLE_KEY.match(key)
        if m and val not in ("0", "", "false"):
            slot = int(m.group(1))
            stations.setdefault(slot, Station(slot=slot)).disabled = True
    return [stations[k] for k in sorted(stations)], top


def read_config(path: str = DEFAULT_CONFIG_PATH) -> tuple[list[Station], dict[str, str]]:
    """Read + parse the calibration file; ([], {}) when it does not exist."""
    if not path or not os.path.exists(path):
        return [], {}
    with open(path, errors="replace") as f:
        return parse_libsurvive_config(f.read())


# --- tracker-log parsing -------------------------------------------------------
# Every pattern below is a real line from a real rig log, not a guess:
#   Info: Adding LH 0 mode: 2 id: e66022c0                      (config load, --v 50)
#   Info: Adding lighthouse ch 5 (idx: 2, cnt: 3)               (NEW channel at runtime)
#   Info: OOTX not set for LH in channel 0; attaching ootx decoder using device WM2
#   Info: Got OOTX packet 2 e66022c0
#   Info: Global solve with 7 scenes for 0 with error of ... (acc err 0.0012)
#   Warning: Disabling LH 3
#   [tracker] streaming 3pt roles=['left'] lighthouses=2         (our own heartbeat)
#   Info: (0) OOTX Decoder: Bad sync bit                        (channel in parens)
_RE_ADD_LH = re.compile(r"Adding LH (\d+) mode:\s*(-?\d+) id:\s*([0-9a-fA-F]+)")
_RE_ADD_CH = re.compile(
    r"Adding lighthouse ch (\d+)(?:\s*\(idx:\s*(\d+)\s*,\s*cnt:\s*(\d+)\s*\))?")
_RE_OOTX_MISSING = re.compile(r"OOTX not set for LH in channel (\d+)")
_RE_OOTX_PACKET = re.compile(r"Got OOTX packet (\d+) ([0-9a-fA-F]+)")
_RE_BAD_SYNC = re.compile(r"\((\d+)\) OOTX Decoder: Bad sync bit")
_RE_GLOBAL_SOLVE = re.compile(r"Global solve with \d+ scenes for (\d+)")
_RE_DISABLING = re.compile(r"Disabling LH (\d+)")
_RE_BRIDGE_COUNT = re.compile(r"lighthouses=(\d+)")


def parse_tracker_log(text: str) -> LogSighting:
    """Extract every base-station sighting from one tracker-bridge log."""
    s = LogSighting()

    def note_id(channel: int, sid: int) -> None:
        s.channels.add(channel)
        s.ids_by_channel.setdefault(channel, set()).add(sid & 0xFFFFFFFF)

    for line in text.splitlines():
        m = _RE_ADD_LH.search(line)
        if m:
            note_id(int(m.group(2)), int(m.group(3), 16))
            continue
        m = _RE_ADD_CH.search(line)
        if m:
            ch = int(m.group(1))
            s.channels.add(ch)
            # cnt is libsurvive's OWN running total of lighthouses it knows about, and it
            # is the most direct answer there is to "how many base stations are in this
            # room" — it counts a channel seen for the first time this launch, which no
            # cached-config reading can. It used to be parsed as None (the regex stopped at
            # idx), so a launch that announced six stations could still be summarized as
            # four: the cached table's four, none of which the new pair is in.
            s.added.append((ch, int(m.group(2)) if m.group(2) else None,
                            int(m.group(3)) if m.group(3) else None))
            continue
        m = _RE_OOTX_PACKET.search(line)
        if m:
            note_id(int(m.group(1)), int(m.group(2), 16))
            continue
        m = _RE_OOTX_MISSING.search(line)
        if m:
            ch = int(m.group(1))
            s.channels.add(ch)
            s.ootx_attached.add(ch)
            continue
        m = _RE_BAD_SYNC.search(line)
        if m:
            ch = int(m.group(1))
            s.bad_sync_by_channel[ch] = s.bad_sync_by_channel.get(ch, 0) + 1
            continue
        m = _RE_GLOBAL_SOLVE.search(line)
        if m:
            s.solved_indices.add(int(m.group(1)))
            continue
        m = _RE_DISABLING.search(line)
        if m:
            s.disabled_slots.add(int(m.group(1)))
            continue
        m = _RE_BRIDGE_COUNT.search(line)
        if m:
            s.bridge_counts.append(int(m.group(1)))
    return s


# --- reporting -----------------------------------------------------------------
def summary_line(stations: list[Station], live: int | None = None,
                 expect: int | None = None) -> str:
    """One compact line for the launch board / the bridge heartbeat.

    e.g. ``2/3 solved (ch 0,2)`` — ``live`` (the count actually streaming now)
    wins over the config count when it is known, because a calibration file can
    list a station that is powered off today.
    """
    solved = [s for s in stations if s.solved]
    n = len(solved) if live is None else live
    chans = ",".join(str(s.channel) for s in solved if s.channel is not None)
    total = f"{n}/{expect}" if expect else str(n)
    listed = len(stations)
    extra = ""
    if listed > len(solved):
        extra = f", {listed - len(solved)} listed-but-unusable"
    return f"{total} solved{f' (ch {chans})' if chans else ''}{extra}"


def diagnose(stations: list[Station], log: LogSighting | None = None,
             expect: int | None = None) -> list[str]:
    """Operator-actionable problems, most important first. Empty == all good."""
    problems: list[str] = []
    solved = [s for s in stations if s.solved]

    if not stations:
        problems.append(
            "no base stations in the calibration file at all — run one FRESH_CALIB=1 "
            "launch and wave the tracker(s) ~20-30 s")
        return problems

    # An UNCACHED channel whose OOTX never decoded blocks EVERYTHING: libsurvive cannot
    # place a station it has no factory blob for, so the wave gate sits at "waiting for
    # first Global solve..." indefinitely with no other symptom. Report it FIRST — it is
    # the actual blocker, and "only N of M in the solve" below is just its consequence.
    # ⚠️ A pending decoder on a channel whose blob IS cached is BENIGN, and the log looks
    # identical. libsurvive persists OOTXSet but not OOTXChecked, so every launch
    # re-attaches a decoder and prints "OOTX not set for LH in channel N" even with a
    # valid cached blob (survive_process_gen2.c: survive_ootx_behavior gates on
    # OOTXChecked) — while the posers gate on OOTXSet (poser_mpfit.c:371,
    # driver_global_scene_solver.c:200) and solve happily from the cache. Measured on
    # this rig: three "OOTX not set" lines and an immediate MPFIT + Global solve on all
    # three stations, no waving. Calling that a blocker sent a real diagnosis (a slot
    # gap, below) chasing base-station line of sight for three launches.
    if log is not None:
        cached_ootx = {s.channel for s in stations if s.ootx_set and s.channel is not None}
        for ch in sorted(log.ootx_pending - cached_ootx):
            bad = log.bad_sync_by_channel.get(ch, 0)
            problems.append(
                f"channel {ch}'s OOTX never decoded and there is NO cached blob for it"
                + (f" ({bad} bad sync bits)" if bad else "")
                + " — libsurvive cannot place a station without its factory blob, so NO "
                  "pose is produced at all. Hold the puck(s) up with clear line of sight "
                  "to that station and wave slowly; if it will not decode, relaunch with "
                  "FRESH_OOTX=0 to keep the cached blob (only positions get re-solved)")

    live = log.live_count if log is not None else None
    have = len(solved) if live is None else min(live, len(solved)) or len(solved)
    if expect and have < expect:
        taken = sorted(s.channel for s in solved if s.channel is not None)
        problems.append(
            f"only {have} of {expect} base stations are in the solve — check the missing "
            f"one's power LED and that a tracker can SEE it"
            # Only mention channels when we actually know which are taken: with zero
            # solved stations this used to read "differs from []", which is nonsense.
            + (f", and that its CHANNEL is not one of {taken}" if taken else "")
            + " (then relaunch with FRESH_CALIB=1)")

    # A GAP in the slot numbering silently deletes the highest station(s) from the
    # solve. libsurvive sets activeLighthouses to the COUNT of entries it read, not to
    # max(slot)+1 (survive.c: the load loop just increments), and then every poser, the
    # force-calibrate/force-ootx reset loops and pysurvive's own object list iterate
    # `i < activeLighthouses` (survive_api.c:310, poser_mpfit.c, poser_epnp.c). So with
    # entries at slots 0,1,3 the count is 3 and slot 3 is NEVER reached, while the empty
    # slot 2 is offered as a station (it enumerates as "LHB-0"). Nothing else says so:
    # the channel still reports, its OOTX still decodes, and the wave gate just never
    # reaches a Global solve because only 2 of 3 stations can contribute.
    # A gap appears on its own — config_save writes only slots with OOTXSet, so one
    # station that failed to decode leaves a hole the NEXT launch inherits.
    slots = sorted(s.slot for s in stations)
    missing = [i for i in range(len(slots)) if i not in set(slots)]
    if missing:
        lost = ", ".join(
            f"slot{s.slot} (ch {s.channel}, id {s.id_hex})"
            for s in stations if s.slot >= len(slots))
        problems.append(
            f"the calibration has a GAP at slot{'s' if len(missing) > 1 else ''} "
            f"{', '.join(str(m) for m in missing)}: libsurvive counts "
            f"{len(slots)} stations and only ever iterates slots 0..{len(slots) - 1}, "
            f"so {lost or 'the highest entry'} is EXCLUDED from every solve while the "
            f"empty slot is offered as a station. Renumber the config's "
            f"\"lighthouseN\" keys to be contiguous from 0 (this preserves the cached "
            f"factory blobs, so no OOTX re-decode is needed), or delete "
            f"~/.config/libsurvive/config.json and take one FRESH_CALIB=1 wave")

    for s in stations:
        if s.disabled:
            problems.append(
                f"slot{s.slot} (ch {s.channel}, id {s.id_hex}) is DISABLED by "
                f"'lighthouse-{s.slot}-disable' in the config — it is calibrated and "
                f"still ignored; delete that key to use it")
        elif s.ootx_set and not s.position_set:
            problems.append(
                f"slot{s.slot} (ch {s.channel}, id {s.id_hex}) was SEEN but never PLACED "
                f"(OOTX yes, position no) — it contributes nothing; wave the tracker(s) "
                f"through the part of the workspace that station covers")
        elif not s.ootx_set and not s.position_set:
            problems.append(
                f"slot{s.slot} (ch {s.channel}) is a stale/empty entry — no OOTX and no "
                f"pose; a FRESH_CALIB=1 launch clears it")

    # Two physical stations on ONE channel: libsurvive maps channel -> one bsd slot,
    # so the pair alternately overwrite each other's id, which RESETS that station's
    # solved pose every time (survive_process_gen2.c: doSave -> reset position).
    # Tracking gets WORSE than with two stations, and nothing else says why.
    if log is not None:
        for ch, ids in sorted(log.ids_by_channel.items()):
            if len(ids) > 1:
                problems.append(
                    f"channel {ch} reported {len(ids)} DIFFERENT station ids "
                    f"({', '.join(f'{i:08x}' for i in sorted(ids))}) — two base stations "
                    f"are sharing that channel. They overwrite each other and reset the "
                    f"solved pose; set one to a free channel with the button on its back")
        # Same collision, caught with only ONE decode in the window: the id reporting
        # on a channel disagrees with the id the calibration has there. libsurvive
        # prints "Got OOTX packet <ch> <id>" ONLY when the id on that slot changed
        # (survive_process_gen2.c), so a single such line already means the channel
        # is not answering with the station it was calibrated against.
        by_channel = {s.channel: s for s in stations if s.channel is not None}
        for ch, ids in sorted(log.ids_by_channel.items()):
            known = by_channel.get(ch)
            if known is None or known.station_id is None or len(ids) > 1:
                continue
            seen_id = next(iter(ids))
            if seen_id != (known.station_id & 0xFFFFFFFF):
                problems.append(
                    f"channel {ch} answered with id {seen_id:08x} but the calibration has "
                    f"{known.id_hex} there — either that station was swapped/moved, or a "
                    f"second station is on channel {ch}. Its solved pose is reset on every "
                    f"flip; give one of them a free channel, then FRESH_CALIB=1")

        cfg_channels = {s.channel for s in solved if s.channel is not None}
        unseen = cfg_channels - log.channels
        if unseen and log.channels:
            problems.append(
                f"channel(s) {sorted(unseen)} are in the calibration file but did NOT "
                f"report this launch — that station is off, unplugged, or blocked")
    return problems


def ootx_line(log: LogSighting) -> str:
    """Compact OOTX-decode progress, e.g. ``ootx 2/3 (ch 0 pending, 16 sync err)``.

    Empty string when no decoder was attached (every blob came from the cache). This is
    the one phase where a launch legitimately looks hung, so it gets its own readout.
    """
    if not log.ootx_attached:
        return ""
    done = sorted(log.ootx_attached & set(log.ids_by_channel))
    pending = sorted(log.ootx_pending)
    out = f"ootx {len(done)}/{len(log.ootx_attached)}"
    if pending:
        errs = sum(log.bad_sync_by_channel.get(c, 0) for c in pending)
        out += f" (ch {','.join(map(str, pending))} pending"
        out += f", {errs} sync err)" if errs else ")"
    return out


def report(stations: list[Station], top: dict[str, str], log: LogSighting | None = None,
           expect: int | None = None, prefix: str = "[lighthouse]",
           diagnostics: bool = True) -> list[str]:
    """Full human-readable report as a list of lines.

    ``diagnostics=False`` prints the table without the WARN block — used when the state
    being described is about to be discarded anyway (a ``--fresh`` launch clears every
    cached position, so warning that they are unplaced is noise that reads like a fault).
    """
    live = None if log is None else log.live_count
    lines = [f"{prefix} {summary_line(stations, live=live, expect=expect)}"]
    gen = top.get("configed-lighthouse-gen")
    if gen:
        lines[0] += f"  [lighthouse gen {gen}]"
    for s in stations:
        lines.append(f"{prefix}   {s.describe()}")
    if log is not None:
        seen = ",".join(str(c) for c in sorted(log.channels)) or "none"
        lines.append(f"{prefix}   this launch: channels seen {seen}"
                     + (f", added at runtime {[a[0] for a in log.added]}" if log.added else "")
                     + (f"  {ootx_line(log)}" if ootx_line(log) else ""))
    if diagnostics:
        for p in diagnose(stations, log=log, expect=expect):
            lines.append(f"{prefix} WARN: {p}")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="libsurvive calibration file (default: %(default)s)")
    p.add_argument("--log", default=None,
                   help="a tracker-bridge log to cross-check against (data/logs/tracker.log)")
    p.add_argument("--expect", type=int, default=None,
                   help="how many base stations SHOULD be in the solve (warn when fewer)")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when any problem is found (default: report and exit 0, so "
                        "this can never abort a launch on its own)")
    p.add_argument("--quiet", action="store_true", help="print only the summary line")
    p.add_argument("--ootx-only", action="store_true",
                   help="print ONLY the OOTX-decode progress line for --log (empty when "
                        "every blob came from the cache). Used by the launcher's wave gate, "
                        "which polls this once a second.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.ootx_only:
        if not args.log or not os.path.exists(args.log):
            return 0
        with open(args.log, errors="replace") as f:
            print(ootx_line(parse_tracker_log(f.read())), flush=True)
        return 0

    stations, top = read_config(args.config)
    if not stations and not os.path.exists(args.config):
        print(f"[lighthouse] no calibration file at {args.config} — "
              f"has libsurvive ever run on this machine?", file=sys.stderr)
        return 2
    log = None
    if args.log:
        if not os.path.exists(args.log):
            print(f"[lighthouse] no log at {args.log} (skipping the launch cross-check)",
                  file=sys.stderr)
        else:
            with open(args.log, errors="replace") as f:
                log = parse_tracker_log(f.read())

    lines = report(stations, top, log=log, expect=args.expect)
    print(lines[0] if args.quiet else "\n".join(lines), flush=True)
    return 1 if (args.strict and diagnose(stations, log=log, expect=args.expect)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
