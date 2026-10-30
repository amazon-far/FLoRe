"""Multi-GPU (DDP) training plumbing: shard geometry, failure modes, single-process no-ops.

CPU + gloo throughout, so the whole file runs in CI with no GPU. What that does and does
not prove:

* **Proves** the DDP wiring — shard partitioning, collective ordering, rank-0 guards,
  coordinated abort. These are process-level properties, and the backend does not change
  them.
* **Does not prove** NCCL-specific behaviour or throughput. Those are measured separately
  on the 8x L40S box.

Every multi-rank test spawns real processes via ``torch.multiprocessing.spawn``. A
thread-based fake would not exercise the process group, which is the part that can silently
be wrong. The model-level checks (gradient parity, EMA parity, every parameter receiving a
gradient) live in ``tests/test_ddp_diffusion.py``, which reuses this file's harness.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from flore.s1_policy.training.trainer import EpochShuffleSampler


def _spawn(worker, world_size: int, *args, timeout: int = 300):
    """Run ``worker`` in ``world_size`` processes; fail the test on any child error.

    ``join=True`` inside a bounded timeout is deliberate: a DEADLOCK is one of the failure
    modes under test (unequal shards, mismatched collectives), and it must surface as a
    test failure rather than a hung CI job.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    procs = []
    for rank in range(world_size):
        p = context.Process(target=_worker_entry, args=(worker, rank, world_size, queue, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout)
    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.terminate()
        p.join(5)
    results = []
    while not queue.empty():
        results.append(queue.get())
    if alive:
        pytest.fail(f"{len(alive)} of {world_size} ranks did not finish within {timeout}s "
                    f"(deadlock); results so far: {results}")
    errors = [r for r in results if isinstance(r, tuple) and r and r[0] == "ERROR"]
    if errors:
        pytest.fail("worker error:\n" + "\n".join(str(e[1]) for e in errors))
    exit_codes = {p.exitcode for p in procs}
    if exit_codes != {0}:
        pytest.fail(f"ranks exited with {exit_codes}; results: {results}")
    return results


def _worker_entry(worker, rank: int, world_size: int, queue, args) -> None:
    """Set the env torchrun would set, join the group, run, and always tear down."""
    import traceback

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": os.environ.get("_TEST_PORT", "29517"),
    })
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        queue.put(worker(rank, world_size, *args))
    except Exception:
        queue.put(("ERROR", f"rank {rank}: {traceback.format_exc()}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Coordinated abort and collective ordering -- a hang must surface as a failure
# ---------------------------------------------------------------------------
def _nan_abort_worker(rank: int, world_size: int):
    """Rank 1 alone reports a non-finite loss; both ranks must conclude "abort"."""
    from flore.s1_policy.training.distributed import any_rank_flagged

    saw_nan_locally = rank == 1
    return {"rank": rank, "local": saw_nan_locally,
            "should_abort": any_rank_flagged(saw_nan_locally)}


def _uneven_collective_worker(rank: int, world_size: int):
    """Reproduce the shape of the validation loop: unequal per-rank batch counts, then ONE
    reduction after the loop.

    Regression guard for a real defect found during review of this work: ``evaluate`` used to
    RAISE inside its batch loop on a non-finite loss. Because ranks hold different numbers of
    validation batches, one rank raising is a collective its peers never make -- they walk
    into the post-loop all-reduce and block until the 10-minute NCCL timeout. A hang, not a
    crash. The fix records the condition and reduces it once, after the loop, which is what
    this asserts: per-rank work may differ, collective COUNT may not.
    """
    from flore.s1_policy.training.distributed import all_reduce_sum, any_rank_flagged

    # Deliberately uneven, exactly as a contiguous validation shard can be.
    n_batches = 2 + rank
    local_bad = rank == world_size - 1        # only the LAST rank sees trouble
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(n_batches):
        total += 1.0                          # no collective inside the loop

    should_abort = any_rank_flagged(local_bad)     # exactly one collective per rank
    reduced = float(all_reduce_sum(total))         # exactly one more
    return {"rank": rank, "n_batches": n_batches, "should_abort": should_abort,
            "reduced": reduced}


def test_uneven_per_rank_work_still_makes_matching_collectives():
    """Ranks may do different AMOUNTS of work, but must make the same NUMBER of collectives.

    If this test hangs, the deadlock it guards against has returned -- `_spawn` bounds the
    join and fails rather than hanging CI.
    """
    world_size = 3
    results = _spawn(_uneven_collective_worker, world_size)
    assert len(results) == world_size, results
    assert sorted(r["n_batches"] for r in results) == [2, 3, 4], "shards should be uneven"
    # Every rank agrees on the abort decision, and on the reduced total (2+3+4).
    assert all(r["should_abort"] for r in results), results
    for result in results:
        assert result["reduced"] == pytest.approx(9.0), result


def test_a_nan_on_one_rank_aborts_every_rank():
    """One rank's numerical failure must stop the whole job, not hang it.

    Without the all-reduce, the rank that raises leaves the group and its peers block at the
    next collective until the NCCL timeout (10 min) -- so a legible crash becomes a hang
    that reads as slow training.
    """
    results = _spawn(_nan_abort_worker, 2)
    assert len(results) == 2, results
    assert all(r["should_abort"] for r in results), results
    # And exactly one rank actually saw it, so the flag really did propagate.
    assert sum(r["local"] for r in results) == 1, results


# ---------------------------------------------------------------------------
# Shard geometry -- pure logic, no processes needed
# ---------------------------------------------------------------------------
def test_shards_are_disjoint_equal_and_cover_the_truncated_permutation():
    """Every rank must get an equal, disjoint slice.

    Equal lengths are what prevent the classic silent DDP deadlock: with drop_last=True,
    unequal shards mean one rank runs fewer batches, exits the loop early, and never reaches
    the final gradient all-reduce.
    """
    size, world_size = 1000, 3
    samplers = [EpochShuffleSampler(size, seed=7, rank=r, world_size=world_size)
                for r in range(world_size)]
    for sampler in samplers:
        sampler.set_epoch(4)
    shards = [list(iter(s)) for s in samplers]

    assert {len(s) for s in shards} == {size // world_size}
    for sampler, shard in zip(samplers, shards):
        assert len(shard) == len(sampler) == sampler.shard_size
    flat = [i for shard in shards for i in shard]
    assert len(set(flat)) == len(flat), "shards overlap: a sample would be seen twice"
    # The union is the permutation truncated to a multiple of world_size.
    assert len(flat) == (size // world_size) * world_size
    assert set(flat) <= set(range(size))
    # Truncation drops fewer than world_size samples.
    assert size - len(flat) == size % world_size < world_size


def test_all_ranks_shard_the_same_underlying_permutation():
    """The ranks partition ONE global permutation without communicating.

    They agree because the permutation is seeded by (seed + epoch) alone. If a rank's order
    depended on its rank, the shards could overlap or miss samples silently.
    """
    size, world_size = 60, 4
    union = set()
    for rank in range(world_size):
        sampler = EpochShuffleSampler(size, seed=11, rank=rank, world_size=world_size)
        sampler.set_epoch(2)
        union |= set(iter(sampler))
    solo = EpochShuffleSampler(size, seed=11, rank=0, world_size=1)
    solo.set_epoch(2)
    assert union == set(iter(solo)), "sharded union differs from the single-process order"


def test_a_different_epoch_drops_a_different_truncated_tail():
    """The truncation must not exclude the same samples every epoch.

    With a fixed permutation, `size % world_size` samples would be permanently invisible.
    Reseeding per epoch makes the dropped tail rotate.
    """
    size, world_size = 101, 4  # 101 % 4 == 1, so exactly one sample is dropped per epoch
    dropped = set()
    for epoch in range(12):
        seen = set()
        for rank in range(world_size):
            sampler = EpochShuffleSampler(size, seed=3, rank=rank, world_size=world_size)
            sampler.set_epoch(epoch)
            seen |= set(iter(sampler))
        missing = set(range(size)) - seen
        assert len(missing) == 1
        dropped |= missing
    assert len(dropped) > 1, "the same sample is dropped every epoch"


def test_mid_epoch_offset_preserves_the_remaining_shard_order():
    """Resume must continue a rank's shard, not restart or reshuffle it."""
    sampler = EpochShuffleSampler(120, seed=5, rank=1, world_size=3)
    sampler.set_epoch(2)
    full = list(iter(sampler))
    sampler.set_epoch(2, offset=7)
    assert list(iter(sampler)) == full[7:]
    assert len(sampler) == len(full) - 7


def test_offset_is_bounded_by_the_shard_not_the_dataset():
    """An offset valid for the whole dataset but past this rank's shard must be refused.

    Accepting it would silently yield an empty rank while its peers still had batches --
    i.e. a deadlock at the next collective.
    """
    sampler = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    assert sampler.shard_size == 30
    sampler.set_epoch(0, offset=30)          # exactly at the end is legal
    with pytest.raises(ValueError, match=r"outside \[0, 30\]"):
        sampler.set_epoch(0, offset=31)


def test_resume_refuses_a_different_world_size():
    """The shard depends on world_size, so resuming on a different one replays other data."""
    saved = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    saved.set_epoch(3, offset=2)
    state = saved.state_dict()
    assert state["world_size"] == 4

    with pytest.raises(ValueError, match="world_size=4"):
        EpochShuffleSampler(120, seed=5, rank=0, world_size=2).load_state_dict(state)

    same = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    same.load_state_dict(state)
    assert (same.epoch, same.offset) == (3, 2)


def test_a_pre_ddp_checkpoint_still_resumes():
    """A state written before DDP existed has no world_size key; treat it as 1."""
    legacy = {"size": 64, "seed": 5, "epoch": 2, "offset": 3}   # no world_size
    sampler = EpochShuffleSampler(64, seed=5)
    sampler.load_state_dict(legacy)
    assert (sampler.epoch, sampler.offset, sampler.world_size) == (2, 3, 1)


def test_single_process_sampler_is_unchanged_by_the_ddp_plumbing():
    """world_size=1 must behave exactly as before: full permutation, no truncation.

    Guards the promise that a non-torchrun run is bit-identical to the pre-DDP trainer.
    """
    sampler = EpochShuffleSampler(97, seed=13)
    sampler.set_epoch(6)
    order = list(iter(sampler))
    expected = torch.randperm(97, generator=torch.Generator().manual_seed(13 + 6)).tolist()
    assert order == expected
    assert sampler.shard_size == 97 and len(order) == 97


def test_sampler_rejects_an_impossible_topology():
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        EpochShuffleSampler(10, seed=0, world_size=0)
    with pytest.raises(ValueError, match=r"rank 4 is outside \[0, 4\)"):
        EpochShuffleSampler(10, seed=0, rank=4, world_size=4)


# ---------------------------------------------------------------------------
# Single-process no-op guarantees
# ---------------------------------------------------------------------------
def test_distributed_helpers_are_inert_without_a_process_group():
    """Every helper must be safe and side-effect-free in a plain single-process run."""
    from flore.s1_policy.training.distributed import (
        all_reduce_sum,
        any_rank_flagged,
        barrier,
        broadcast_flag,
        destroy_process_group,
        get_dist_info,
        wrap_model,
    )

    info = get_dist_info()
    assert not info.enabled and info.is_main and info.world_size == 1

    barrier()                      # must not raise
    destroy_process_group()        # must not raise
    assert any_rank_flagged(True) is True
    assert any_rank_flagged(False) is False
    assert broadcast_flag(True) is True

    tensor = torch.tensor([2.0, 3.0], dtype=torch.float64)
    out = all_reduce_sum(tensor)
    torch.testing.assert_close(out, tensor)
    assert out is not tensor, "must not alias the caller's accumulator"

    from tests.test_ddp_diffusion import _model

    model = _model()
    assert wrap_model(model, info) is model, "single process must not be DDP-wrapped"


def test_wrap_model_keeps_state_dict_keys_unprefixed():
    """A DDP-wrapped save would prefix every key with 'module.'.

    flore.s1_policy.inference.common loads with strict=True, so such a checkpoint is written
    successfully and then refused by every reader. The trainer therefore keeps `self.model`
    unwrapped and saves from it; this pins the property that makes that necessary.
    """
    from flore.s1_policy.training.distributed import DistInfo
    from tests.test_ddp_diffusion import _model

    model = _model()
    unwrapped_keys = set(model.state_dict())
    assert not any(k.startswith("module.") for k in unwrapped_keys)

    # Simulate what a wrapped save would produce, without needing a process group.
    class _FakeWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapped_keys = set(_FakeWrapper(model).state_dict())
    assert all(k.startswith("module.") for k in wrapped_keys)
    assert wrapped_keys != unwrapped_keys
    assert DistInfo(world_size=4).enabled and not DistInfo(rank=1, world_size=4).is_main
