"""libsurvive_cache: the slot gap, the re-channelled fleet, and what a repair must not touch.

Run: .venv/bin/python -m pytest bridges/tracker/test_libsurvive_cache.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libsurvive_cache as lc  # noqa: E402


def _station(slot, ch, sid, ootx=1, pos=0, tilt="-0.046112060547"):
    return f'''"lighthouse{slot}":{{
"index":"{slot}",
"id":"{sid}",
"mode":"{ch}",
"pose":["1.000000000000","2.000000000000","3.000000000000","0.500000000000","0.500000000000","0.500000000000","0.500000000000"],
"variance":["0.100000000000","0.100000000000","0.100000000000","0.100000000000","0.100000000000","0.100000000000"],
"unlock_count":"7",
"accel":["0.000000000000","127.000000000000","104.000000000000"],
"fcaltilt":["{tilt}","0.051666259766"],
"fcalgibmag":["-0.010726928711","-0.009239196777"],
"OOTXSet":"{ootx}",
"PositionSet":"{pos}"
}}
'''


GLOBALS = '"v":"0",\n"poser":"MPFIT",\n"ootx-ignore-sync-error":"0"\n'


def test_parses_the_real_format_without_json():
    # the real file has NO comma between top-level entries — json.load raises on it
    cfg = lc.Config.parse(GLOBALS + _station(1, 1, 3959321706) + _station(3, 4, 2534754941))
    assert [s.slot for s in cfg.stations] == [1, 3]
    assert [s.channel for s in cfg.stations] == [1, 4]
    assert [s.station_id for s in cfg.stations] == [3959321706, 2534754941]
    assert all(s.ootx_set and not s.position_set for s in cfg.stations)
    assert cfg.global_items()["poser"] == "MPFIT"
    assert cfg.channels() == [1, 4] and cfg.positioned() == []


def test_renumber_closes_the_gap_that_hides_a_station():
    cfg = lc.Config.parse(GLOBALS + _station(1, 1, 111) + _station(3, 4, 222))
    assert cfg.has_slot_gap()
    fixed = cfg.renumber()
    assert not fixed.has_slot_gap()
    assert [s.slot for s in fixed.stations] == [0, 1]
    assert [s.index for s in fixed.stations] == [0, 1]      # the block's own field moved too
    assert '"lighthouse0":' in fixed.render() and '"lighthouse1":' in fixed.render()
    assert "lighthouse3" not in fixed.render()
    # everything else is byte-identical: the factory calibration must survive a repair
    assert '"fcaltilt":["-0.046112060547","0.051666259766"]' in fixed.render()
    assert lc.Config.parse(fixed.render()).channels() == [1, 4]


def test_clear_poses_keeps_the_factory_calibration():
    cfg = lc.Config.parse(GLOBALS + _station(0, 8, 111, pos=1)).clear_poses()
    st = cfg.stations[0]
    assert not st.position_set
    assert lc._field(st.block, "pose") == '["0.000000000000","0.000000000000","0.000000000000",' \
        '"0.000000000000","0.000000000000","0.000000000000","0.000000000000"]'
    assert '"variance":["0.000000000000"' in st.block
    assert '"fcaltilt":["-0.046112060547"' in st.block   # kept: --force-calibrate does not clear fcal
    assert '"accel":["0.000000000000","127.000000000000","104.000000000000"]' in st.block
    assert '"unlock_count":"7"' in st.block


def test_drop_unusable_removes_the_slot_the_ootx_gate_blocks_on():
    cfg = lc.Config.parse(GLOBALS + _station(0, 8, 0, ootx=0) + _station(1, 1, 111))
    assert len(cfg.stations) == 2
    assert [s.channel for s in cfg.drop_unusable().stations] == [1]


def test_merge_refuses_a_rechannelled_station():
    mine = lc.Config.parse(GLOBALS + _station(0, 1, 111))
    old = lc.Config.parse(GLOBALS + _station(0, 1, 999) + _station(1, 10, 888))
    merged, refused = mine.merge(old)
    # ch1 is already ours under a different station id: taking it would shadow us in bsd_map
    assert [s.station_id for s in refused] == [999]
    assert [(s.channel, s.station_id) for s in merged.stations] == [(1, 111), (10, 888)]


def test_merge_keeps_our_own_readout_for_the_same_station():
    mine = lc.Config.parse(GLOBALS + _station(0, 1, 111, tilt="-0.001"))
    old = lc.Config.parse(GLOBALS + _station(0, 1, 111, tilt="-0.999"))
    merged, refused = mine.merge(old)
    assert len(merged.stations) == 1 and refused == []
    assert '"fcaltilt":["-0.001"' in merged.stations[0].block


def test_repair_end_to_end(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(GLOBALS + _station(1, 1, 111, pos=1) + _station(3, 4, 222, pos=1)
                    + _station(5, 8, 0, ootx=0))
    cfg, notes = lc.repair(str(path), clear_poses=True, ootx_ignore_sync_error=True)
    assert cfg is not None
    joined = "\n".join(notes)
    assert "dropped 1 station block(s) with no factory calibration" in joined
    assert "closed a slot gap [1, 3]" in joined
    assert "cleared every station POSITION" in joined
    cfg.save(str(path))
    back = lc.Config.load(str(path))
    assert [s.slot for s in back.stations] == [0, 1]
    assert back.channels() == [1, 4] and back.positioned() == []
    assert back.global_items()["ootx-ignore-sync-error"] == "1"
    assert back.global_items()["poser"] == "MPFIT"      # untouched knobs survive
    assert not back.has_slot_gap() and back.duplicate_channels() == []


def test_repair_drops_a_duplicate_channel():
    cfg = lc.Config.parse(GLOBALS + _station(0, 1, 111) + _station(1, 1, 222))
    assert cfg.duplicate_channels() == [1]


def test_reset_writes_globals_only(tmp_path):
    path = str(tmp_path / "config.json")
    lc.write_globals_only(path, ootx_ignore_sync_error=True)
    cfg = lc.Config.load(path)
    assert cfg.stations == [] and cfg.global_items()["ootx-ignore-sync-error"] == "1"


def test_missing_cache_is_not_an_error(tmp_path):
    assert lc.Config.load_if_present(str(tmp_path / "nope.json")) is None
    cfg, notes = lc.repair(str(tmp_path / "nope.json"))
    assert cfg is None and "no cache" in notes[0]
