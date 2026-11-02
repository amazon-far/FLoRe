#!/usr/bin/env python3
"""lighthouse_status.py: base-station detection, against REAL libsurvive artifacts.

Every fixture below is a verbatim excerpt from this rig — the cached calibration
file and two tracker logs — because the whole module is a parser and a parser
tested against invented input tests nothing. In particular:

  * the calibration file is NOT valid JSON (bare newline-separated members, no
    comma between blocks); ``json.load`` on it raises "Extra data: line 1
    column 4". A test that fed it well-formed JSON would pass while the real
    file failed to parse.
  * ``Got OOTX packet <ch> <id>`` is printed ONLY when the id on that slot
    CHANGED (survive_process_gen2.c), which is why one such line already means
    the channel is not answering with the station it was calibrated against.
  * the launcher greps the bridge heartbeat with ``-oE "roles=\\[[^]]*\\]"``, so
    ``lighthouses=N`` must stay AFTER the closing bracket. Asserted here.

Run:  python3 bridges/tracker/test_lighthouse_status.py
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lighthouse_status as lhs  # noqa: E402

_CHECKS = 0


def check(cond, msg: str) -> None:
    global _CHECKS
    _CHECKS += 1
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


# --- fixtures: verbatim from this rig -----------------------------------------
# ~/.config/libsurvive/config.json, trimmed to the fields this module reads.
CONFIG_2 = '''"v":"0",
"poser":"MPFIT",
"configed-lighthouse-gen":"2",
"floor-offset":"-0.178424596786"
"lighthouse0":{
"index":"0",
"id":"3865060032",
"mode":"2",
"pose":["1.118214909156","-0.284750398992","0.641690445657","-0.700543427435","-0.366390867506","-0.302334466460","-0.532532166988"],
"variance":["0.000005555258","0.000091983517","0.000003510240","0.000061820906","0.000183638087","0.000072186606"],
"OOTXSet":"1",
"PositionSet":"1"
}
"lighthouse1":{
"index":"1",
"id":"797581664",
"mode":"0",
"pose":["0.000000000000","1.997005750880","0.640372012931","0.043241218974","0.035067370364","0.507056580346","0.860112841926"],
"OOTXSet":"1",
"PositionSet":"1"
}
'''

# A real 2-station launch (data/logs/tracker.log, FRESH_CALIB=1).
LOG_2 = """[tracker] starting pysurvive ['--lighthouse-update-velocity', '0.5', '--force-calibrate', '1', '--force-ootx', '1']
[tracker] mode=3pt roles=['left', 'right', 'head']
Info: Force calibrate flag set -- clearing position on all lighthouses
Info: Adding tracked object WM0 from HTC
Info: OOTX not set for LH in channel 0; attaching ootx decoder using device WM1
Info: OOTX not set for LH in channel 2; attaching ootx decoder using device WM1
Info: Got OOTX packet 2 e66022c0
Info: Got OOTX packet 0 2f8a2160
Info: Global solve with 7 scenes for 0 with error of 16809042.741274/1244.8896284422 (acc err 0.0012)
Info: Global solve with 7 scenes for 1 with error of 16809042.741274/1244.8896284422 (acc err 0.0011)
Info: Using LH 1 (2f8a2160) as reference lighthouse
[tracker] streaming 3pt roles=['head', 'left', 'right'] lighthouses=2/3 expected
"""


def _config_3() -> str:
    """The same file after a successful THREE-station calibration."""
    return CONFIG_2 + '''"lighthouse2":{
"index":"2",
"id":"1234abcd",
"mode":"5",
"pose":["-1.500000000000","0.500000000000","2.100000000000","1.0","0.0","0.0","0.0"],
"OOTXSet":"1",
"PositionSet":"1"
}
'''


def _config_gap() -> str:
    """Three REAL stations, but persisted at slots 0, 1, 3 — the shape that cost this
    rig three stalled launches. config_save only writes slots with OOTXSet, so a
    station that failed to decode once leaves a hole every later launch inherits."""
    return CONFIG_2 + '''"lighthouse3":{
"index":"3",
"id":"2534754941",
"mode":"5",
"pose":["-1.500000000000","0.500000000000","2.100000000000","1.0","0.0","0.0","0.0"],
"OOTXSet":"1",
"PositionSet":"1"
}
'''


def main() -> int:
    # --- the file really is not JSON -----------------------------------------
    import json
    try:
        json.loads(CONFIG_2)
        raise AssertionError("fixture parsed as JSON — it is no longer the real format")
    except json.JSONDecodeError:
        check(True, "libsurvive's config.json is NOT valid JSON (why the parser is line-based)")

    # --- parsing the cached calibration --------------------------------------
    stations, top = lhs.parse_libsurvive_config(CONFIG_2)
    check(len(stations) == 2, "two stations parsed out of the real 2-station config")
    check(top.get("configed-lighthouse-gen") == "2", "top-level scalars survive the blocks")
    by_ch = {s.channel: s for s in stations}
    check(sorted(by_ch) == [0, 2], f"channels are 0 and 2, not 0 and 1 (got {sorted(by_ch)})")
    check(by_ch[2].station_id == 0xE66022C0, "the decimal 'id' is the same number as the hex log id")
    check(by_ch[2].id_hex == "e66022c0", "id renders as the hex libsurvive prints")
    check(all(s.solved for s in stations), "both stations count as solved (OOTX + position)")
    check(len(by_ch[2].pose) == 7, "pose is xyz + wxyz (7), parsed from the quoted array")
    check(len(by_ch[2].variance) == 6, "variance is 6 long")
    # A station legitimately sitting at x == 0.0 must not read as "unplaced": this rig's
    # ch-0 station really does. A zero-component test here would be a live bug in the
    # bridge's placed check.
    check(by_ch[0].pose[0] == 0.0 and by_ch[0].solved,
          "a station at x=0.000 is still SOLVED (a single zero component is normal)")

    # --- id-parsing edge: hex-looking decimal ids ----------------------------
    check(lhs.parse_libsurvive_config('"lighthouse0":{\n"mode":"1"\n}\n')[0][0].station_id is None,
          "a station with no id parses with station_id=None instead of raising")

    # --- the solved/unsolved distinction ------------------------------------
    seen_not_placed = CONFIG_2.replace('"OOTXSet":"1",\n"PositionSet":"1"\n}\n"lighthouse1"',
                                       '"OOTXSet":"1",\n"PositionSet":"0"\n}\n"lighthouse1"')
    st2, _ = lhs.parse_libsurvive_config(seen_not_placed)
    check(sum(s.solved for s in st2) == 1, "OOTX-yes/position-no does NOT count as solved")
    probs = lhs.diagnose(st2)
    check(any("never PLACED" in p for p in probs),
          "a seen-but-unplaced station is reported (it contributes nothing, silently)")

    # --- a DISABLED station: calibrated, listed, and ignored -----------------
    st3, _ = lhs.parse_libsurvive_config(CONFIG_2 + '"lighthouse-0-disable":"1"\n')
    check(not st3[0].solved, "'lighthouse-N-disable' makes that station unusable")
    check(any("DISABLED" in p for p in lhs.diagnose(st3)),
          "a disabled station is called out (the nastiest failure: fully calibrated, ignored)")

    # --- log parsing --------------------------------------------------------
    log = lhs.parse_tracker_log(LOG_2)
    check(log.channels == {0, 2}, f"both channels found in the log (got {log.channels})")
    check(log.ids_by_channel == {0: {0x2F8A2160}, 2: {0xE66022C0}},
          "each channel's decoded station id is attributed to that channel")
    check(log.solved_indices == {0, 1}, "the global solver ran for both lighthouse indices")
    check(log.bridge_counts == [2] and log.live_count == 2,
          "the bridge heartbeat's lighthouses=N is the live count")
    check(not log.added, "no runtime-added channel in a cached 2-station launch")

    # The launcher's own grep must still see the roles list with the new field appended.
    hb = [ln for ln in LOG_2.splitlines() if "streaming" in ln][-1]
    roles = re.findall(r"roles=\[[^]]*\]", hb)
    check(roles == ["roles=['head', 'left', 'right']"],
          "launcher's -oE roles=[...] grep is unaffected by the appended lighthouses= field")
    check(hb.index("lighthouses=") > hb.index("]"),
          "lighthouses= sits AFTER the roles bracket (order the launcher grep depends on)")

    # `_locked_roles` must read the HEARTBEAT, never the startup line. The bare
    # "roles=[...]" pattern also matched "[tracker] mode=3pt roles=[...]", which lists the
    # CONFIGURED roles — so the wave gate displayed "locked: left, right, head" before a
    # single pose existed. Observed live on a hung launch: 0 heartbeats, 3 roles shown.
    STARTUP_ONLY = ("[tracker] mode=3pt roles=['left', 'right', 'head']\n"
                    "[tracker] role map: LHR-75106789 -> left\n")
    bare = re.findall(r"roles=\[[^]]*\]", STARTUP_ONLY)
    anchored = re.findall(r"streaming [0-9]pt roles=\[[^]]*\]", STARTUP_ONLY)
    check(bare and not anchored,
          "the OLD bare pattern matches the startup line; the anchored one does not")
    check(re.findall(r"streaming [0-9]pt roles=\[[^]]*\]", LOG_2)
          == ["streaming 3pt roles=['head', 'left', 'right']"],
          "the anchored pattern still matches a real heartbeat")

    # A runtime-added channel — how a NEWLY MOUNTED station shows up.
    added = lhs.parse_tracker_log("Info: Adding lighthouse ch 5 (idx: 2, cnt: 3)\n")
    check(added.channels == {5} and added.added == [(5, 2, 3)],
          "'Adding lighthouse ch N (idx: I, cnt: C)' is recognised as a NEW station")
    # `cnt` is libsurvive's OWN running total, and it is the only reading that counts a
    # station which is in NEITHER the cached table nor the solve. Parsed as None, a launch
    # announcing six stations summarized as the cached four — measured on this rig, whose
    # channels 1 and 10 were new while its cached 4 and 8 were silent the same launch.
    check(added.known_count == 3, f"cnt: is captured as the room's count ({added.known_count})")
    two = lhs.parse_tracker_log(
        "Info: Adding lighthouse ch 1 (idx: 4, cnt: 5)\n"
        "Info: Adding lighthouse ch 10 (idx: 5, cnt: 6)\n")
    check(two.known_count == 6 and sorted(two.channels) == [1, 10],
          f"the count is the MAX cnt seen, not the number of channels ({two.known_count})")
    check(lhs.parse_tracker_log("Info: Adding lighthouse ch 3 (cnt: 2)\n").channels == {3},
          "the gen1 form of that line (no idx:) also parses")
    check(lhs.parse_tracker_log("Info: Adding lighthouse ch 3 (cnt: 2)\n").known_count == 1,
          "with no parsable cnt the count falls back to the channels seen, never to 0")

    # --- expectation gating -------------------------------------------------
    check(not lhs.diagnose(stations, log=log, expect=2), "2 of 2 expected -> no problems")
    p3 = lhs.diagnose(stations, log=log, expect=3)
    check(any("only 2 of 3" in p for p in p3), "2 of 3 expected -> reported")
    check(any("CHANNEL is not one of [0, 2]" in p for p in p3),
          "the report names the channels already taken (what the new station must avoid)")
    st_three, top3 = lhs.parse_libsurvive_config(_config_3())
    log3 = lhs.parse_tracker_log(
        LOG_2.replace("lighthouses=2/3", "lighthouses=3/3")
        + "Info: Adding lighthouse ch 5 (idx: 2, cnt: 3)\nInfo: Got OOTX packet 5 1234abcd\n")
    check(len(st_three) == 3 and not lhs.diagnose(st_three, log=log3, expect=3),
          "a real 3-station calibration reports clean at expect=3")
    check("3/3 solved" in lhs.summary_line(st_three, live=3, expect=3),
          "the summary line reads N/M for the board")

    # --- a SLOT GAP: three good stations, one of them invisible to the solver -
    # Same three stations as above, persisted at slots 0/1/3. Every field reads
    # healthy (OOTXSet + PositionSet on all three), so only the numbering says
    # anything is wrong — and libsurvive's activeLighthouses = COUNT means slot 3
    # is never iterated while the empty slot 2 is offered as a station.
    st_gap, _ = lhs.parse_libsurvive_config(_config_gap())
    check([s.slot for s in st_gap] == [0, 1, 3], "the gap is parsed as-is, not renumbered")
    check(all(s.solved for s in st_gap),
          "every station in the gapped file looks solved — nothing else can catch this")
    p_gap = lhs.diagnose(st_gap, log=log3, expect=3)
    check(any("GAP at slot 2" in p for p in p_gap), "a slot gap is reported, and names the hole")
    check(any("slot3 (ch 5, id 97154a7d)" in p for p in p_gap),
          "the report names WHICH station is excluded from the solve")
    check(not any("GAP" in p for p in lhs.diagnose(st_three, log=log3, expect=3)),
          "contiguous slots 0,1,2 do NOT trigger the gap report")

    # --- channel collisions: the failure that makes tracking WORSE ----------
    # Two ids on one channel in the same launch.
    both = lhs.parse_tracker_log(LOG_2 + "Info: Got OOTX packet 0 ad3c6894\n")
    check(len(both.ids_by_channel[0]) == 2, "two ids on one channel are both retained")
    check(any("sharing that channel" in p for p in lhs.diagnose(stations, log=both, expect=3)),
          "two stations on one channel is reported as a collision")
    # Only ONE decode in the window, but it disagrees with the calibration. This is the
    # real observed case (probe log: ch 0 answered ad3c6894, config has 2f8a2160).
    swapped = lhs.parse_tracker_log(
        "Info: Got OOTX packet 0 ad3c6894\nInfo: Global solve with 3 scenes for 0 with error of 1/1 (acc err 0.1)\n")
    msgs = lhs.diagnose(stations, log=swapped, expect=2)
    check(any("answered with id ad3c6894" in p for p in msgs),
          "a single id/channel disagreement is caught (one decode is enough)")

    # --- the OOTX phase: the failure that makes a WORKING rig look hung ------
    # Verbatim from the launch that hung at the wave gate: --force-ootx cleared every
    # blob, ch 2 and ch 3 re-decoded, ch 0 did not, and libsurvive can produce NO pose
    # until it does — so the gate sat at "waiting for first Global solve..." forever with
    # no other symptom.
    STUCK = """Info: Force ootx flag set -- clearing ootx on all lighthouses
Info: OOTX not set for LH in channel 0; attaching ootx decoder using device WM0
Info: Adding lighthouse ch 3 (idx: 2, cnt: 3)
Info: OOTX not set for LH in channel 3; attaching ootx decoder using device WM0
Info: OOTX not set for LH in channel 2; attaching ootx decoder using device WM0
Info: Got OOTX packet 2 e66022c0
Info: Got OOTX packet 3 ad3c6894
Info: (0) Preamble found
Info: (0) OOTX Decoder: Bad sync bit
Info: (0) OOTX Decoder: Bad sync bit
Info: (2) OOTX Decoder: Bad sync bit
"""
    stuck = lhs.parse_tracker_log(STUCK)
    check(stuck.ootx_attached == {0, 2, 3}, "every channel with a decoder attached is tracked")
    check(stuck.ootx_pending == {0}, f"only ch 0 is still pending (got {stuck.ootx_pending})")
    check(stuck.bad_sync_by_channel == {0: 2, 2: 1},
          "'(N) OOTX Decoder: Bad sync bit' is counted PER CHANNEL")
    check(lhs.ootx_line(stuck) == "ootx 2/3 (ch 0 pending, 2 sync err)",
          f"the gate's one-line readout names the stuck channel (got {lhs.ootx_line(stuck)!r})")
    # A stuck decode only BLOCKS when there is no cached blob to fall back on, so the
    # blocker fixture needs OOTXSet=0 on that channel.
    fresh = [
        lhs.Station(slot=0, channel=2, station_id=0xE66022C0, ootx_set=True, position_set=True),
        lhs.Station(slot=1, channel=0, ootx_set=False, position_set=False),
    ]
    msgs = lhs.diagnose(fresh, log=stuck, expect=3)
    check("OOTX never decoded" in msgs[0],
          "the stuck decode is reported FIRST — it is the blocker, not a side effect")
    check("FRESH_OOTX=0" in msgs[0], "the report names the escape hatch")
    # ...and with the blob CACHED the identical log is benign. libsurvive re-attaches a
    # decoder every launch (OOTXChecked is not persisted, only OOTXSet) and prints the
    # same "OOTX not set for LH in channel N" line while the posers solve from the cache.
    # Calling this a blocker is what sent a real slot-gap diagnosis chasing line of sight.
    check(not any("OOTX never decoded" in p for p in lhs.diagnose(stations, log=stuck, expect=3)),
          "a pending decode on a channel whose blob is CACHED is not reported as a blocker")
    # Once everything decodes there is nothing to say, and no line to print.
    done = lhs.parse_tracker_log(STUCK + "Info: Got OOTX packet 0 2f8a2160\n")
    check(not done.ootx_pending and lhs.ootx_line(done) == "ootx 3/3",
          "a completed decode reads 3/3 with no pending clause")
    check(not any("OOTX never decoded" in p for p in lhs.diagnose(stations, log=done)),
          "no OOTX warning once every blob is decoded")
    # LOG_2 is itself a --force-ootx launch (it has "OOTX not set for LH in channel …"),
    # so it attached two decoders and finished both.
    check(lhs.ootx_line(lhs.parse_tracker_log(LOG_2)) == "ootx 2/2",
          f"a finished 2-station re-decode reads 2/2 (got {lhs.ootx_line(lhs.parse_tracker_log(LOG_2))!r})")
    # FRESH_OOTX=0 / FRESH_CALIB=0: no decoder is ever attached, so there is nothing to
    # report and the gate must not show an empty "decoding base stations:" prefix.
    cached_only = lhs.parse_tracker_log(
        "Info: Adding LH 0 mode: 2 id: e66022c0\n"
        "Info: Global solve with 7 scenes for 0 with error of 1/1 (acc err 0.001)\n")
    check(lhs.ootx_line(cached_only) == "",
          "a launch that used only CACHED blobs prints no ootx line at all")
    check(lhs.main(["--log", "/nonexistent.log", "--ootx-only"]) == 0,
          "--ootx-only on a missing log is silent and exits 0 (it polls once a second)")

    # The zero-solved case used to render "its CHANNEL differs from []".
    wiped, _ = lhs.parse_libsurvive_config(
        CONFIG_2.replace('"PositionSet":"1"', '"PositionSet":"0"'))
    zero = [p for p in lhs.diagnose(wiped, expect=3) if "of 3" in p]
    check(zero and "[]" not in zero[0],
          f"with nothing solved the report does not say 'differs from []' (got {zero[0]!r})")

    # A station in the calibration that never reported this launch.
    quiet = lhs.parse_tracker_log(
        "Info: OOTX not set for LH in channel 2; attaching ootx decoder using device WM1\n")
    check(any("did NOT report this launch" in p for p in lhs.diagnose(stations, log=quiet)),
          "a cached station that stayed silent is reported (off / unplugged / blocked)")

    # --- empty + robustness -------------------------------------------------
    check(any("no base stations" in p for p in lhs.diagnose([])),
          "an empty calibration is reported, not silently 'fine'")
    check(lhs.read_config("/nonexistent/config.json") == ([], {}),
          "a missing calibration file is ([], {}), not an exception")
    junk = '"lighthouse0":{\n"mode":"garbage",\n"pose":[broken\n"OOTXSet":"1"\n}\n'
    st_junk, _ = lhs.parse_libsurvive_config(junk)
    check(len(st_junk) == 1 and st_junk[0].channel is None,
          "a malformed line is skipped, not fatal (a diagnostic must still report)")
    check(lhs.parse_tracker_log("") .live_count == 0, "an empty log reads as zero, not a crash")

    # --- the real files on this machine, if present --------------------------
    real, real_top = lhs.read_config()
    if real:
        check(all(s.channel is not None for s in real),
              f"the LIVE calibration parses ({len(real)} station(s), "
              f"ch {[s.channel for s in real]})")
        check(real_top.get("configed-lighthouse-gen") in ("1", "2", None),
              "the live config's lighthouse gen is a known value")

    # --- CLI ---------------------------------------------------------------
    rc = lhs.main(["--config", "/nonexistent/x.json"])
    check(rc == 2, "a missing config exits 2")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(CONFIG_2)
        cfg = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(LOG_2)
        logf = f.name
    try:
        check(lhs.main(["--config", cfg, "--log", logf, "--expect", "3"]) == 0,
              "a MISSING station still exits 0 by default (never aborts a launch)")
        check(lhs.main(["--config", cfg, "--log", logf, "--expect", "3", "--strict"]) == 1,
              "--strict is what turns a problem into a nonzero exit")
        check(lhs.main(["--config", cfg, "--log", logf, "--expect", "2", "--strict"]) == 0,
              "--strict with the expectation met still exits 0")
    finally:
        os.unlink(cfg)
        os.unlink(logf)

    print(f"\nLOGIC OK ({_CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
