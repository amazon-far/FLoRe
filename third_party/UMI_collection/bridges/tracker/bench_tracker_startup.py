#!/usr/bin/env python3
"""Measure the tracker startup, N times, and average it.

Each trial runs ``vive_tracker_bridge.py`` exactly as ``run_collection_viz.sh`` does —
calibration cache removed, ``--force-calibrate 1 --force-ootx 1``, block until every role
streams — timestamps every line it prints and records, per trial:

  * when each base station's OOTX blob decoded ("Got OOTX packet <channel> <id>")
  * when each puck produced its first pose
  * when the bridge declared READY (every role streaming), or FAIL at --timeout

then prints a table and mean / min / max over the trials that locked. The pucks have to
be MOVED during a trial (sensor face toward the stations): OOTX needs line of sight and
the from-scratch solve needs parallax — five pucks left on the bench decode a few blobs
and never lock. Run from the repo root::

    .venv/bin/python bridges/tracker/bench_tracker_startup.py --trials 3 --timeout 240

Serials default to this rig's; ``--mode`` picks how many roles must lock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BRIDGE = os.path.join(HERE, "vive_tracker_bridge.py")

DEFAULT_SERIALS = {
    "left": "LHR-75106789", "right": "LHR-60CDD21F", "head": "LHR-3DC276BA",
    "left_foot": "LHR-C68495D2", "right_foot": "LHR-FF1D3DE2",
}
ROLES_BY_MODE = {"1pt": ("left",), "2pt": ("left", "right"), "3pt": ("left", "right", "head"),
                 "5pt": ("left", "right", "head", "left_foot", "right_foot")}

RE_OOTX = re.compile(r"Got OOTX packet (\d+) ([0-9a-f]+)")
RE_FIRST = re.compile(r"tracker (\S+) serial=(LHR-\S+) produced its first pose")
RE_READY = re.compile(r"READY all roles streaming after ([\d.]+)s")
RE_REF = re.compile(r"Using LH (\d+) \(([0-9a-f]+)\) as reference")
RE_ADD = re.compile(r"Adding tracked object \S*?(WM\d+|T\d+)")


def libsurvive_holders() -> list[int]:
    """PIDs whose command line names a libsurvive holder, minus our own ancestor chain.

    ``pgrep -f`` matches command lines, so a wrapper shell that QUOTES one of these names
    (``bash -c "... vive_tracker_bridge.py ..."``) is an ancestor and must not count.
    """
    pat = r"vive_tracker_bridge\.py|survive-cli|survive-websocketd|pose_check|vive_tracker_viser"
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    cands = [int(x) for x in out.stdout.split() if x.isdigit()]
    anc = set()
    p = os.getpid()
    while p > 1:
        anc.add(p)
        try:
            with open(f"/proc/{p}/stat") as f:
                p = int(f.read().split(")")[-1].split()[1])
        except OSError:
            break
    return [c for c in cands if c not in anc]


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def run_trial(k: int, args, env: dict) -> dict:
    roles = ROLES_BY_MODE[args.mode]
    cmd = [args.python, BRIDGE, "--mode", args.mode, "--clear-cache",
           "--startup-wait", str(args.timeout + 5), "--port", str(args.port)]
    for role in roles:
        cmd += [f"--{role.replace('_', '-')}-serial", getattr(args, f"{role}_serial")]
    rec: dict = {"trial": k, "ootx": {}, "first_pose": {}, "added": [], "ready_s": None,
                 "first_solve_s": None, "reference": None, "outcome": "timeout", "log": None}
    print(f"\n=== trial {k}/{args.trials}: fresh cache, {args.mode}, up to {args.timeout}s — "
          f">>> MOVE THE PUCKS NOW, sensor face toward the stations <<<", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            preexec_fn=os.setsid)
    done = threading.Event()
    log_path = os.path.join(args.log_dir, f"bench_trial{k}.log")
    rec["log"] = log_path
    os.makedirs(args.log_dir, exist_ok=True)
    lf = open(log_path, "w")

    def reader():
        for raw in proc.stdout:
            dt = time.monotonic() - t0
            line = strip_ansi(raw.rstrip("\n"))
            lf.write(f"{dt:8.2f} {line}\n")
            lf.flush()
            if "Preamble found" in line or "Bad sync bit" in line or "LightcapMode" in line:
                continue
            m = RE_ADD.search(line)
            if m and m.group(1) not in rec["added"]:
                rec["added"].append(m.group(1))
            m = RE_OOTX.search(line)
            if m and m.group(1) not in rec["ootx"]:
                rec["ootx"][m.group(1)] = round(dt, 1)
                print(f"  [{dt:6.1f}s] OOTX blob of station ch {m.group(1)} (id {m.group(2)}) decoded "
                      f"({len(rec['ootx'])} so far)", flush=True)
            if "Global solve" in line and rec["first_solve_s"] is None:
                rec["first_solve_s"] = round(dt, 1)
                print(f"  [{dt:6.1f}s] first Global solve", flush=True)
            m = RE_REF.search(line)
            if m and rec["reference"] is None:
                rec["reference"] = f"LH{m.group(1)} ({m.group(2)})"
            m = RE_FIRST.search(line)
            if m and m.group(2) not in rec["first_pose"]:
                rec["first_pose"][m.group(2)] = round(dt, 1)
                print(f"  [{dt:6.1f}s] first pose: {m.group(1)} {m.group(2)}", flush=True)
            m = RE_READY.search(line)
            if m:
                rec["ready_s"] = round(dt, 1)
                rec["outcome"] = "ready"
                print(f"  [{dt:6.1f}s] READY — every role streaming (bridge says {m.group(1)}s)", flush=True)
                done.set()
            if "FATAL" in line:
                rec["outcome"] = "fail"
                done.set()
        done.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    done.wait(timeout=args.timeout + 10)
    if rec["outcome"] == "timeout":
        print(f"  [{time.monotonic() - t0:6.1f}s] TIMEOUT — not every role locked", flush=True)
    time.sleep(1.0)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    th.join(timeout=2)
    lf.close()
    rec["n_ootx"] = len(rec["ootx"])
    rec["n_first_pose"] = len(rec["first_pose"])
    print(f"  -> {rec['outcome'].upper()}: ootx {rec['n_ootx']}, pucks on radio {len(rec['added'])}, "
          f"first poses {rec['n_first_pose']}/{len(roles)}, ready {rec['ready_s']}s  (log {log_path})", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=240.0, help="per-trial budget (s)")
    ap.add_argument("--mode", choices=tuple(ROLES_BY_MODE), default="5pt")
    for role, ser in DEFAULT_SERIALS.items():
        ap.add_argument(f"--{role.replace('_', '-')}-serial", default=ser)
    ap.add_argument("--python", default=os.path.join(REPO, ".venv-vive", "bin", "python"))
    ap.add_argument("--libsurvive-dir", default=os.path.expanduser(os.environ.get("LIBSURVIVE_DIR", "~/libsurvive")))
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--log-dir", default=os.path.join(REPO, "data", "logs"))
    ap.add_argument("--json", default=None, help="write the per-trial records here")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between trials (dongles settle)")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{args.libsurvive_dir}/bin" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONPATH"] = f"{args.libsurvive_dir}/bindings/python" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    holders = libsurvive_holders()
    if holders:
        print(f"another libsurvive holder is running (pids {holders}; pysurvive is a singleton) — stop it first",
              file=sys.stderr)
        return 2

    print(f"bench: {args.trials} trials x {args.timeout:.0f}s, mode {args.mode}, libsurvive {args.libsurvive_dir}")
    print("       roles: " + ", ".join(f"{r}={getattr(args, r + '_serial')}" for r in ROLES_BY_MODE[args.mode]))
    recs = []
    for k in range(1, args.trials + 1):
        recs.append(run_trial(k, args, env))
        if k < args.trials:
            time.sleep(args.pause)

    print("\n=== summary ===")
    print("trial  outcome  ootx radio 1st-pose   solve   READY  reference")
    for r in recs:
        fp = min(r["first_pose"].values()) if r["first_pose"] else None
        fp_s = f"{fp:.1f}s" if fp is not None else "-"
        solve_s = f"{r['first_solve_s']:.1f}s" if r["first_solve_s"] is not None else "-"
        ready_s = f"{r['ready_s']:.1f}s" if r["ready_s"] is not None else "-"
        print(f"{r['trial']:>5} {r['outcome']:>8} {r['n_ootx']:>5} {len(r['added']):>5} "
              f"{fp_s:>8} {solve_s:>7} {ready_s:>7}  {r['reference'] or '-'}")
    ok = [r["ready_s"] for r in recs if r["ready_s"] is not None]
    if ok:
        print(f"\nREADY (cache cleared -> every {args.mode} role streaming): mean {sum(ok)/len(ok):.1f}s  "
              f"min {min(ok):.1f}s  max {max(ok):.1f}s  over {len(ok)}/{len(recs)} trials")
    else:
        print(f"\nno trial locked every role within {args.timeout:.0f}s — the pucks must be MOVED during the "
              f"trial (OOTX needs line of sight, the solve needs parallax); re-run with them in hand")
    all6 = [max(r["ootx"].values()) for r in recs if r["n_ootx"] == max(x["n_ootx"] for x in recs) and r["ootx"]]
    if all6:
        n = max(x["n_ootx"] for x in recs)
        print(f"OOTX: {n} blob(s) decoded per trial in mean {sum(all6)/len(all6):.1f}s (max {max(all6):.1f}s)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "trials": recs}, f, indent=1)
        print(f"json -> {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
