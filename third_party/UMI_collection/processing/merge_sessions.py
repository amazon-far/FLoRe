#!/usr/bin/env python3
"""Merge several recording sittings of ONE task into a single session dir.

``convert_to_umi.py`` takes exactly one ``session_dir``, so merging is a pre-step rather
than a converter flag. This builds a session dir whose ``take_*`` entries are **relative
symlinks** into the source sessions: raw bytes are never copied (a 3-sitting merge would
otherwise duplicate ~2 GB) and never modified, and ``os.path.isdir`` follows symlinks, so
the converter, the audit and the per-take QC need no changes at all.

Two things this refuses to do silently, because both would corrupt a dataset in a way that
only shows up as a bad policy weeks later:

* **A contract mismatch.** ``mode``, ``primary_role``, ``serial_role_map``, ``world_align``,
  ``quaternion_convention``, ``record_model`` and ``schema_version`` all collapse into ONE
  set of zarr attrs. Merging a session recorded with a different tracker-to-role map or a
  different world alignment would mix two coordinate conventions under one label. Asserted
  equal up front; use ``--allow-field`` to override a specific one deliberately.
* **A take-number collision.** Take numbers are the only handle QC exclusions have
  (``--exclude take_058``), so every take needs a globally unique name whose block says
  which sitting it came from. Offsets default to the next free multiple of 100.

``created_wall`` is the FIRST sitting's, so episode order stays chronological, and the
converter's ``sorted()`` discovery already yields that order given zero-padded names.

Usage
-----
    merge_sessions.py -o data/recordings/session_2026.08.14_merged \\
        data/recordings/session_2026.08.12_15.38.36 \\
        data/recordings/session_2026.08.12_16.21.52 \\
        data/recordings/session_2026.08.14_16.41.40

Sources must be RAW sittings, not other merged dirs — passing a merged dir would chain
symlinks and make ``merged_from`` claim a provenance one level away from the real bytes.
Detected and refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Every field that becomes a single zarr attr for the whole merged dataset.
CONTRACT_FIELDS = (
    "mode",
    "primary_role",
    "serial_role_map",
    "world_align",
    "quaternion_convention",
    "record_model",
    "schema_version",
)

TAKE_META_FILE = "take_meta.json"


def _load_meta(session_dir: str) -> dict:
    path = os.path.join(session_dir, "session_meta.json")
    if not os.path.exists(path):
        sys.exit(f"[merge] {session_dir}: no session_meta.json — cannot verify the tracking contract")
    with open(path) as f:
        return json.load(f)


def _committed_takes(session_dir: str) -> list[str]:
    """Same discovery rule as convert_to_umi: a take counts only once take_meta.json lands,
    which is what makes an interrupted recording invisible instead of half-ingested."""
    out = []
    for name in sorted(os.listdir(session_dir)):
        d = os.path.join(session_dir, name)
        if (name.startswith("take_") and not name.endswith(".tmp")
                and os.path.isdir(d) and os.path.exists(os.path.join(d, TAKE_META_FILE))):
            out.append(name)
    return out


def _take_index(name: str) -> int:
    base = name[len("take_"):]
    if not base.isdigit():
        sys.exit(f"[merge] non-numeric take name {name!r}: offsets are only meaningful for take_<int>")
    return int(base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="raw session dirs, in chronological order")
    ap.add_argument("-o", "--output", required=True, help="merged session dir to create")
    ap.add_argument("--offsets", type=int, nargs="*", default=None,
                    help="take-number offset per source (default: 0, then the next free "
                         "multiple of 100 after the previous source's highest take)")
    ap.add_argument("--allow-field", nargs="*", default=(), metavar="FIELD",
                    help=f"contract fields allowed to differ between sittings, from "
                         f"{list(CONTRACT_FIELDS)}. Each one you list is a coordinate or "
                         f"schema convention you are asserting does not matter for this merge.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output dir (only if it holds nothing but "
                         "symlinked takes, session_meta.json and calibration/)")
    args = ap.parse_args()

    srcs = [s.rstrip("/") for s in args.sources]
    if len(srcs) < 2:
        sys.exit("[merge] need at least two sources")
    if len(set(os.path.abspath(s) for s in srcs)) != len(srcs):
        sys.exit("[merge] duplicate source session")

    # --- refuse merged inputs: a symlinked take means the real bytes live elsewhere ---
    for s in srcs:
        meta = _load_meta(s)
        if "merged_from" in meta:
            sys.exit(f"[merge] {s} is itself a merged session. Pass its sources "
                     f"({meta['merged_from']}) instead, so merged_from names the real bytes.")

    metas = {s: _load_meta(s) for s in srcs}
    takes = {s: _committed_takes(s) for s in srcs}
    for s in srcs:
        if not takes[s]:
            sys.exit(f"[merge] {s}: no committed take_* dirs")

    # --- the contract check ---
    ref = srcs[0]
    allow = set(args.allow_field)
    bad = []
    for field in CONTRACT_FIELDS:
        want = metas[ref].get(field)
        for s in srcs[1:]:
            got = metas[s].get(field)
            if got != want:
                bad.append(
                    f"  {field}: {ref} = {json.dumps(want)}   vs   {s} = {json.dumps(got)}"
                    + ("   [ALLOWED via --allow-field]" if field in allow else ""))
    hard = [b for b in bad if "[ALLOWED" not in b]
    if bad:
        print("[merge] tracking-contract differences:\n" + "\n".join(bad), file=sys.stderr)
    if hard:
        sys.exit("[merge] refusing: the fields above become ONE set of zarr attrs, so merging "
                 "would file two conventions under one label. Re-check the recording, or pass "
                 "--allow-field for each field you are deliberately overriding.")

    # --- offsets ---
    if args.offsets is None:
        offsets, nxt = [], 0
        for s in srcs:
            offsets.append(nxt)
            hi = nxt + _take_index(takes[s][-1])
            nxt = (hi // 100 + 1) * 100
    else:
        if len(args.offsets) != len(srcs):
            sys.exit(f"[merge] got {len(args.offsets)} offsets for {len(srcs)} sources")
        offsets = list(args.offsets)

    # name -> "session/take", and the collision check that makes exclusions unambiguous
    take_map: dict[str, str] = {}
    link_target: dict[str, str] = {}
    for s, off in zip(srcs, offsets):
        for name in takes[s]:
            merged = f"take_{_take_index(name) + off:03d}"
            if merged in take_map:
                sys.exit(f"[merge] take-number collision at {merged}: "
                         f"{take_map[merged]} and {os.path.basename(s)}/{name}. "
                         f"Raise the offsets — a colliding name makes --exclude ambiguous.")
            take_map[merged] = f"{os.path.basename(s)}/{name}"
            link_target[merged] = os.path.join(s, name)

    out = args.output.rstrip("/")
    if os.path.exists(out):
        if not args.force:
            sys.exit(f"[merge] {out} exists (use --force to rebuild it)")
        strays = [e for e in os.listdir(out)
                  if not (e == "session_meta.json" or e == "calibration"
                          or os.path.islink(os.path.join(out, e)))]
        if strays:
            sys.exit(f"[merge] {out} holds real files, not just symlinks: {strays[:5]} — "
                     f"refusing to overwrite something that may be a raw recording")
        for e in os.listdir(out):
            p = os.path.join(out, e)
            if os.path.islink(p) or os.path.isfile(p):
                os.remove(p)
    os.makedirs(out, exist_ok=True)

    # Relative links, so the whole recordings/ tree stays movable.
    for merged, target in sorted(link_target.items()):
        rel = os.path.relpath(os.path.abspath(target), os.path.abspath(out))
        link = os.path.join(out, merged)
        os.symlink(rel, link)
        if not os.path.isdir(link):
            sys.exit(f"[merge] {link} -> {rel} does not resolve to a dir")

    # calibration/ is carried from the first sitting purely so the dir looks like a session;
    # every recent session ships it empty and the repo defaults are used instead.
    os.makedirs(os.path.join(out, "calibration"), exist_ok=True)

    merged_meta = {k: metas[ref][k] for k in metas[ref] if k != "created_str"}
    merged_meta["created_str"] = os.path.basename(out).replace("session_", "", 1)
    # Rebuild in the source key order, with created_str back in its slot.
    ordered = {}
    for k in metas[ref]:
        ordered[k] = merged_meta[k]
    ordered["merged_from"] = [os.path.basename(s) for s in srcs]
    ordered["merged_take_offsets"] = {os.path.basename(s): o for s, o in zip(srcs, offsets)}
    ordered["merged_take_map"] = dict(sorted(take_map.items()))
    blocks = ", ".join(f"{os.path.basename(s)} -> take_{o:03d}+" for s, o in zip(srcs, offsets))
    ordered["_note"] = (
        "MERGED session: take_* are RELATIVE SYMLINKS into the source sessions (raw bytes are "
        "never copied or modified). One task recorded in several sittings; the tracking contract "
        f"({'/'.join(CONTRACT_FIELDS)}) was asserted equal before merging. Take-number blocks: "
        f"{blocks}, so any QC exclusion traces back to its sitting by name; merged_take_map is "
        "the authoritative mapping. created_wall is the FIRST sitting's, so episode ordering is "
        "chronological across sittings. Built by processing/merge_sessions.py.")

    with open(os.path.join(out, "session_meta.json"), "w") as f:
        json.dump(ordered, f, indent=2)

    print(f"[merge] {out}: {len(take_map)} takes from {len(srcs)} sittings")
    for s, o in zip(srcs, offsets):
        lo = o + _take_index(takes[s][0])
        hi = o + _take_index(takes[s][-1])
        print(f"  {os.path.basename(s):32s} {len(takes[s]):4d} takes -> take_{lo:03d}..take_{hi:03d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
