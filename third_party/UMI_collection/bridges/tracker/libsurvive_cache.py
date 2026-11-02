#!/usr/bin/env python3
"""Read, repair and re-render ``~/.config/libsurvive/config.json``.

WHY THIS EXISTS. libsurvive's cache holds two DIFFERENT kinds of thing per base station
and only one of them is worth re-deriving on every launch:

* ``fcal*`` + ``accel`` + ``id`` + ``mode`` — the station's **factory** calibration and
  identity, read out of its OOTX radio blob. Burned in at the factory; it cannot change
  while the station is the same physical unit on the same channel.
* ``pose`` + ``variance`` + ``PositionSet`` — **where the station is in the room**. This
  is what a station being nudged invalidates, and what must be re-solved.

``--force-calibrate 1`` clears only the second group (``survive.c``: "clearing position on
all lighthouses"); ``--force-ootx 1`` clears the first (``bsd[i].OOTXSet = 0`` and
``memset(bsd[i].fcal, 0, ...)``). So a fresh POSITION solve does not require re-reading a
single OOTX blob, and re-reading them is not free:

  ``poser_mpfit.c`` refuses to attempt ANY room solve until **every** station it has seen
  has ``OOTXSet`` ("Wait til this thing gets OOTX ... canPossiblySolveLHS = false; break").
  One station whose blob never completes therefore blocks the whole room, forever.

THE SLOT GAP (the bug this module was written for). ``survive_config.c::config_save``
writes a station only ``if (ctx->bsd[i].OOTXSet)``, under the block name ``lighthouse<i>``
where ``i`` is its runtime slot. A run that decoded slots 1 and 3 but not 0 and 2 saves
``lighthouse1`` + ``lighthouse3`` — a numbering GAP. On the next load ``survive.c`` counts
``activeLighthouses`` by how many blocks parsed (2 here) while keeping each block's own
index, so the loop ``for (lh = 0; lh < activeLighthouses; lh++)`` inspects slots 0 and 1:
slot 0 is EMPTY (``OOTXSet`` 0, so the all-OOTX gate blocks forever) and the real station in
slot 3 is never looked at at all. :func:`renumber` closes the gap.

Nothing here parses JSON: libsurvive's format is JSON-ish but has no commas between
top-level entries, so ``json.load`` fails on the real file. Station blocks are carried
VERBATIM — only the block name, the ``index`` field and (on request) the pose fields are
rewritten, so no field this module does not understand can be reformatted into something
libsurvive's own hand-rolled parser reads differently.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

DEFAULT_PATH = os.path.expanduser("~/.config/libsurvive/config.json")

#: how many pose numbers each field holds (survive_config.c::config_read_lighthouse)
POSE_LEN = 7
VARIANCE_LEN = 6

_KEY_RE = re.compile(r'"([^"]+)"\s*:\s*')


def _field(block: str, name: str) -> str | None:
    m = re.search(r'"%s"\s*:\s*(\[[^\]]*\]|"[^"]*")' % re.escape(name), block)
    if not m:
        return None
    return m.group(1).strip('"')


def _set_field(block: str, name: str, rendered: str) -> str:
    """Replace one field's VALUE, keeping everything else byte-identical."""
    pat = re.compile(r'("%s"\s*:\s*)(\[[^\]]*\]|"[^"]*")' % re.escape(name))
    if not pat.search(block):
        return block
    return pat.sub(lambda m: m.group(1) + rendered, block, count=1)


@dataclass
class Station:
    """One ``lighthouse<i>`` block."""
    slot: int              # the i in the block name
    block: str             # the verbatim "{...}" text
    path: str = ""         # which file it came from (provenance for a merge)

    @property
    def index(self) -> int | None:
        v = _field(self.block, "index")
        return int(v) if v is not None else None

    @property
    def channel(self) -> int | None:
        v = _field(self.block, "mode")
        return int(v) if v is not None else None

    @property
    def station_id(self) -> int | None:
        v = _field(self.block, "id")
        return int(v) if v is not None else None

    @property
    def ootx_set(self) -> bool:
        return (_field(self.block, "OOTXSet") or "0") not in ("0", "")

    @property
    def position_set(self) -> bool:
        return (_field(self.block, "PositionSet") or "0") not in ("0", "")

    def with_slot(self, slot: int) -> Station:
        block = _set_field(self.block, "index", f'"{slot}"')
        return Station(slot, block, self.path)

    def without_pose(self) -> Station:
        """Keep the factory calibration, drop where-it-is (what --force-calibrate clears)."""
        zeros = lambda n: "[" + ",".join(['"0.000000000000"'] * n) + "]"  # noqa: E731
        block = _set_field(self.block, "pose", zeros(POSE_LEN))
        block = _set_field(block, "variance", zeros(VARIANCE_LEN))
        block = _set_field(block, "PositionSet", '"0"')
        return Station(self.slot, block, self.path)

    def describe(self) -> str:
        sid = self.station_id
        return (f"ch{self.channel} id={sid:#010x}" if sid is not None else f"ch{self.channel} id=?") + \
               f" OOTX={'y' if self.ootx_set else 'n'} pos={'y' if self.position_set else 'n'}"


@dataclass
class Config:
    globals_text: str            # the top-level entries, verbatim
    stations: list               # list[Station], in slot order
    path: str = ""

    # -- reading --
    @classmethod
    def parse(cls, text: str, path: str = "") -> Config:
        stations: list = []
        pieces: list = []
        i, n = 0, len(text)
        while i < n:
            m = _KEY_RE.search(text, i)
            if not m:
                break
            key, j = m.group(1), m.end()
            if j < n and text[j] == "{":
                depth, k = 0, j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                block = text[j:k + 1]
                if key.startswith("lighthouse"):
                    suffix = key[len("lighthouse"):]
                    stations.append(Station(int(suffix) if suffix.isdigit() else len(stations), block, path))
                i = k + 1
            else:
                k = j
                if j < n and text[j] == "[":
                    while k < n and text[k] != "]":
                        k += 1
                    k += 1
                else:
                    while k < n and text[k] not in ",\n":
                        k += 1
                pieces.append((key, text[j:k].strip().rstrip(",")))
                i = k
        globals_text = "".join(f'"{k}":{v},\n' for k, v in pieces)
        return cls(globals_text, sorted(stations, key=lambda s: s.slot), path)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> Config:
        with open(path) as f:
            return cls.parse(f.read(), path)

    @classmethod
    def load_if_present(cls, path: str = DEFAULT_PATH) -> Config | None:
        try:
            return cls.load(path)
        except OSError:
            return None

    # -- the globals, as a dict --
    def global_items(self) -> dict:
        out = {}
        for m in _KEY_RE.finditer(self.globals_text):
            k, j = m.group(1), m.end()
            end = self.globals_text.find("\n", j)
            out[k] = self.globals_text[j:end if end != -1 else None].strip().rstrip(",").strip('"')
        return out

    def set_global(self, key: str, value: str) -> Config:
        items = self.global_items()
        items[key] = value
        text = "".join(f'"{k}":"{v}",\n' for k, v in items.items())
        return Config(text, list(self.stations), self.path)

    # -- repairs --
    def renumber(self) -> Config:
        """Slots 0..n-1 with no gap — see the module docstring."""
        return Config(self.globals_text, [s.with_slot(i) for i, s in enumerate(self.stations)], self.path)

    def has_slot_gap(self) -> bool:
        return [s.slot for s in self.stations] != list(range(len(self.stations)))

    def clear_poses(self) -> Config:
        return Config(self.globals_text, [s.without_pose() for s in self.stations], self.path)

    def merge(self, other: Config) -> tuple:
        """Add stations from ``other`` this config lacks, keyed by station id (the OOTX
        identity). Ours wins — it is the more recent readout.

        ⚠️ A station whose CHANNEL is already taken is REFUSED, not added. Stations get
        re-channelled, and libsurvive keys its runtime lookup by channel
        (``survive.c``: ``bsd_map[bsd[i].mode] = i``), so two blocks claiming one channel
        make the second silently shadow the first — every sweep on that channel would be
        attributed to the wrong station's factory calibration. Measured on this rig: the
        2026-08-08 and 2026-09-05 backups both hold stations on ch0/ch1/ch2 that the
        current room has on other channels, i.e. the fleet was re-channelled and their
        calibration is not reusable by channel.

        Returns ``(config, refused)`` where ``refused`` names the collisions.
        """
        have_ids = {s.station_id for s in self.stations if s.station_id is not None}
        have_ch = {s.channel for s in self.stations if s.channel is not None}
        extra, refused = [], []
        for st in other.stations:
            if st.station_id is None or st.station_id in have_ids or not st.ootx_set:
                continue
            if st.channel in have_ch:
                refused.append(st)
                continue
            extra.append(st)
            have_ids.add(st.station_id)
            have_ch.add(st.channel)
        return Config(self.globals_text, list(self.stations) + extra, self.path), refused

    def duplicate_channels(self) -> list:
        seen, dupes = set(), []
        for st in self.stations:
            if st.channel in seen:
                dupes.append(st.channel)
            seen.add(st.channel)
        return sorted(set(dupes))

    def drop_unusable(self) -> Config:
        """Only stations whose factory calibration actually decoded are worth keeping: a
        block without ``OOTXSet`` is the empty slot the all-OOTX gate then blocks on."""
        return Config(self.globals_text, [s for s in self.stations if s.ootx_set], self.path)

    # -- writing --
    def render(self) -> str:
        out = [self.globals_text]
        for s in self.stations:
            out.append(f'"lighthouse{s.slot}":{s.block}\n')
        return "".join(out)

    def save(self, path: str | None = None) -> str:
        path = path or self.path or DEFAULT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(self.render())
        os.replace(tmp, path)   # never leave a half-written config for libsurvive to read
        return path

    # -- reporting --
    def channels(self) -> list:
        return sorted(s.channel for s in self.stations if s.channel is not None)

    def positioned(self) -> list:
        return sorted(s.channel for s in self.stations if s.position_set and s.channel is not None)

    def summary(self) -> str:
        if not self.stations:
            return "no stations cached"
        return (f"{len(self.stations)} station(s): "
                + ", ".join(s.describe() for s in self.stations)
                + (" [SLOT GAP]" if self.has_slot_gap() else ""))


def repair(path: str = DEFAULT_PATH, merge_from: list | None = None, clear_poses: bool = False,
           ootx_ignore_sync_error: bool | None = None) -> tuple:
    """Load, drop empty slots, optionally merge older readouts, renumber, save.

    Returns ``(config, notes)``; ``config`` is None when there was no cache to repair.
    """
    cfg = Config.load_if_present(path)
    notes: list = []
    if cfg is None:
        return None, ["no cache at " + path]
    before = cfg.summary()
    dropped = len(cfg.stations)
    cfg = cfg.drop_unusable()
    if len(cfg.stations) != dropped:
        notes.append(f"dropped {dropped - len(cfg.stations)} station block(s) with no factory calibration")
    for other in merge_from or []:
        oc = Config.load_if_present(other)
        if oc is None:
            continue
        merged, refused = cfg.merge(oc.drop_unusable())
        if len(merged.stations) != len(cfg.stations):
            notes.append(f"took {len(merged.stations) - len(cfg.stations)} station(s) from {other}")
        if refused:
            notes.append(f"REFUSED {len(refused)} station(s) from {other} — channel(s) "
                         + ", ".join(f"ch{s.channel}" for s in refused)
                         + " already cached under a different station id (re-channelled fleet)")
        cfg = merged
    dupes = cfg.duplicate_channels()
    if dupes:
        keep, seen = [], set()
        for st in cfg.stations:
            if st.channel in seen:
                continue
            seen.add(st.channel)
            keep.append(st)
        notes.append("dropped later block(s) duplicating " + ", ".join(f"ch{c}" for c in dupes)
                     + " — one channel cannot map to two stations (bsd_map)")
        cfg = Config(cfg.globals_text, keep, cfg.path)
    if cfg.has_slot_gap():
        notes.append("closed a slot gap " + str([s.slot for s in cfg.stations]))
    cfg = cfg.renumber()
    if clear_poses:
        cfg = cfg.clear_poses()
        notes.append("cleared every station POSITION (factory calibration kept)")
    if ootx_ignore_sync_error is not None:
        cfg = cfg.set_global("ootx-ignore-sync-error", "1" if ootx_ignore_sync_error else "0")
    cfg.path = path
    notes.append("before: " + before)
    notes.append("after:  " + cfg.summary())
    return cfg, notes


def write_globals_only(path: str = DEFAULT_PATH, ootx_ignore_sync_error: bool = True) -> str:
    """A cache with NO stations — a genuinely fresh OOTX + position solve, but still
    carrying the global knobs (``ootx-ignore-sync-error`` is read from the config file,
    not from argv, so deleting the file would silently drop it)."""
    cfg = Config('"v":"0",\n', [], path).set_global(
        "ootx-ignore-sync-error", "1" if ootx_ignore_sync_error else "0")
    cfg.path = path
    return cfg.save(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--show", action="store_true", help="print what is cached and exit")
    p.add_argument("--repair", action="store_true", help="drop empty slots, close slot gaps, save")
    p.add_argument("--merge-from", nargs="*", default=[],
                   help="older config files to take factory calibration from (by station id)")
    p.add_argument("--clear-poses", action="store_true",
                   help="keep the factory calibration, drop every solved POSITION")
    p.add_argument("--reset", action="store_true",
                   help="replace the cache with globals only (fresh OOTX + position solve)")
    p.add_argument("--ootx-ignore-sync-error", choices=("0", "1"), default=None,
                   help="libsurvive knob: tolerate a bad OOTX sync bit instead of restarting the frame")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reset:
        if args.dry_run:
            print("would reset " + args.path)
            return 0
        print("reset " + write_globals_only(args.path, args.ootx_ignore_sync_error != "0"))
        return 0
    if args.show or not (args.repair or args.clear_poses or args.ootx_ignore_sync_error):
        cfg = Config.load_if_present(args.path)
        print(f"{args.path}: " + (cfg.summary() if cfg else "missing"))
        if cfg:
            print("  globals: " + ", ".join(f"{k}={v}" for k, v in cfg.global_items().items()))
        return 0
    ignore = None if args.ootx_ignore_sync_error is None else args.ootx_ignore_sync_error == "1"
    cfg, notes = repair(args.path, args.merge_from, args.clear_poses, ignore)
    for n in notes:
        print("  " + n)
    if cfg is None:
        return 1
    if not args.dry_run:
        cfg.save(args.path)
        print("  wrote " + args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
