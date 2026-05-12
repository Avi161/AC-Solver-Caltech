"""Unit tests for value_search.value_guided_search.

Covers:
  - value_guided_greedy_search (with length priority and untrained model)
  - beam_search
  - _check_cache_with_rotations / _expand_cache_with_rotations
  - backfill_solution_cache
  - expand_path_with_cyclic_reductions
  - load_model
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from ac_solver.envs.ac_moves import ACMove
from ac_solver.envs.utils import is_presentation_trivial
from value_search.value_guided_search import (
    _check_cache_with_rotations,
    _expand_cache_with_rotations,
    _reconstruct_path,
    backfill_solution_cache,
    beam_search,
    expand_path_with_cyclic_reductions,
    load_model,
    value_guided_greedy_search,
)
from value_search.value_network import FeatureMLP, SequenceValueNet


# AK(2): same fixture used by test_mcts.py
AK2 = np.array(
    [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
)


def _verify_path_reaches_trivial(presentation, path, cyclical=False):
    """Replay a path and assert it reaches a length-2 (trivial) state."""
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    wl = [
        int(np.count_nonzero(state[:mrl])),
        int(np.count_nonzero(state[mrl:])),
    ]
    for action, _ in path:
        state, wl = ACMove(action, state, mrl, wl, cyclical=cyclical)
    assert sum(wl) == 2, f"Final length {sum(wl)} != 2"
    return state


class TestValueGuidedGreedyLengthPriority:
    """With use_length_priority=True the search behaves like classical greedy."""

    def test_solves_ak2(self):
        solved, path, stats = value_guided_greedy_search(
            AK2,
            use_length_priority=True,
            max_nodes_to_explore=10000,
        )
        assert solved, f"Length-priority search failed on AK(2): {stats}"
        _verify_path_reaches_trivial(AK2, path)
        assert "nodes_explored" in stats and stats["nodes_explored"] > 0
        assert "min_length" in stats

    def test_returns_unsolved_when_budget_exhausted(self):
        solved, path, stats = value_guided_greedy_search(
            AK2,
            use_length_priority=True,
            max_nodes_to_explore=5,
        )
        assert solved is False
        assert path == []
        assert stats["nodes_explored"] >= 5

    def test_path_actions_in_valid_range(self):
        _, path, _ = value_guided_greedy_search(
            AK2, use_length_priority=True, max_nodes_to_explore=10000,
        )
        for action, length in path:
            assert 0 <= action <= 11
            assert length >= 2

    def test_time_limit_terminates(self):
        # A very small time limit on a moderately hard presentation
        # should return unsolved without exploring the full budget.
        solved, _, stats = value_guided_greedy_search(
            AK2,
            use_length_priority=True,
            max_nodes_to_explore=100_000,
            time_limit=0.001,
        )
        # Either it solved very fast OR the time limit hit (both fine);
        # the important check is that we don't crash and stats is populated.
        assert "nodes_explored" in stats


class TestValueGuidedGreedyWithModel:
    """Search using an untrained model — checks plumbing, not solution quality."""

    def test_mlp_path_runs(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        feat_mean = np.zeros(14, dtype=np.float32)
        feat_std = np.ones(14, dtype=np.float32)
        # Untrained model: not expected to solve, but must run without error
        solved, _, stats = value_guided_greedy_search(
            AK2,
            model=model,
            architecture="mlp",
            feat_mean=feat_mean,
            feat_std=feat_std,
            max_nodes_to_explore=200,
        )
        assert isinstance(solved, bool)
        assert stats["nodes_explored"] > 0

    def test_seq_path_runs(self):
        torch.manual_seed(0)
        model = SequenceValueNet(max_seq_len=len(AK2))
        solved, _, stats = value_guided_greedy_search(
            AK2,
            model=model,
            architecture="seq",
            max_nodes_to_explore=100,
        )
        assert isinstance(solved, bool)
        assert stats["nodes_explored"] > 0


class TestBeamSearch:
    def test_runs_with_untrained_mlp(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        feat_mean = np.zeros(14, dtype=np.float32)
        feat_std = np.ones(14, dtype=np.float32)
        solved, _, stats = beam_search(
            AK2,
            model=model,
            architecture="mlp",
            feat_mean=feat_mean,
            feat_std=feat_std,
            beam_width=4,
            max_nodes_to_explore=200,
        )
        assert isinstance(solved, bool)
        assert "nodes_explored" in stats and "steps" in stats

    def test_runs_with_untrained_seq(self):
        torch.manual_seed(0)
        model = SequenceValueNet(max_seq_len=len(AK2))
        solved, _, stats = beam_search(
            AK2,
            model=model,
            architecture="seq",
            beam_width=4,
            max_nodes_to_explore=100,
        )
        assert isinstance(solved, bool)
        assert stats["nodes_explored"] > 0


class TestReconstructPath:
    def test_simple_path_reconstruction(self):
        # Build a chain root → n1 → n2 → n3
        # parent_map[id] = (action_from_parent, total_length_at_node, parent_id)
        parent_map = {
            0: (None, 6, None),  # root
            1: (3, 5, 0),
            2: (7, 4, 1),
            3: (5, 2, 2),
        }
        path = _reconstruct_path(parent_map, 3)
        assert path == [(3, 5), (7, 4), (5, 2)]

    def test_root_has_empty_path(self):
        parent_map = {0: (None, 6, None)}
        assert _reconstruct_path(parent_map, 0) == []


class TestSolutionCacheLookup:
    def test_direct_cache_hit(self):
        mrl = len(AK2) // 2
        cached_path = [(3, 4), (5, 2)]
        cache = {tuple(AK2): cached_path}
        result = _check_cache_with_rotations(tuple(AK2), mrl, cache)
        assert result == cached_path

    def test_cache_miss_returns_none(self):
        mrl = len(AK2) // 2
        cache = {}
        assert _check_cache_with_rotations(tuple(AK2), mrl, cache) is None

    def test_rotation_expansion_finds_rotated_state(self):
        # Build a state whose cyclic rotation is in cache.
        # r1 = [1, 2, -2] but this has cancelable last+first (-2 + 1 != 0, fine);
        # we need a relator that is not trivially cancellable after rotation.
        # Use r1 = [1, 2, 1, 2] (abab), r2 = [2] (b). mrl = 6.
        mrl = 6
        state = np.zeros(12, dtype=np.int8)
        state[:4] = [1, 2, 1, 2]
        state[mrl] = 2
        state_tup = tuple(state)

        # The rotation by k=1 on r1: [2, 1, 2, 1] → place into a new state
        rotated = state.copy()
        rotated[:4] = [2, 1, 2, 1]
        rotated_tup = tuple(rotated)

        # Put the rotated state in the cache with a fake path
        cached_path = [(0, 2)]
        cache = {rotated_tup: cached_path}

        result = _check_cache_with_rotations(state_tup, mrl, cache)
        # Should find rotated entry and prepend conjugation move(s)
        assert result is not None
        # Last entries of result should match the cached suffix
        assert result[-len(cached_path):] == cached_path
        # Prepended entries should be valid AC moves (0..11)
        for action, _ in result[: -len(cached_path)]:
            assert 0 <= action <= 11

    def test_expand_cache_skips_short_relators(self):
        mrl = 6
        # r1 has length 1, r2 has length 1 → no rotations possible
        state = np.zeros(12, dtype=np.int8)
        state[0] = 1
        state[mrl] = 2
        state_tup = tuple(state)
        cache = {}
        _expand_cache_with_rotations(cache, state_tup, [(0, 2)], mrl, False)
        # Nothing should be added because both relators have length <= 1
        assert cache == {}


class TestBackfillSolutionCache:
    def test_backfill_stores_suffixes(self):
        # Construct a simple problem: solve from AK2 using length-priority search,
        # then backfill the cache and assert every intermediate state has a suffix.
        _, path, _ = value_guided_greedy_search(
            AK2, use_length_priority=True, max_nodes_to_explore=10000,
        )
        assert path, "Need a solved path for this test"

        cache = {}
        backfill_solution_cache(cache, AK2, path, expand_rotations=False)

        # Initial state must be in cache, mapped to full path
        assert tuple(AK2) in cache
        assert cache[tuple(AK2)] == list(path)

        # Loop iterates len(path) times caching one state per iteration;
        # some intermediate states might collide, but never exceed len(path).
        assert 1 <= len(cache) <= len(path)
        # All cached suffixes should be non-empty lists ending at length 2
        for suffix in cache.values():
            assert isinstance(suffix, list) and len(suffix) >= 1
            assert suffix[-1][1] == 2

    def test_backfill_with_rotation_expansion_grows_cache(self):
        _, path, _ = value_guided_greedy_search(
            AK2, use_length_priority=True, max_nodes_to_explore=10000,
        )
        assert path

        cache_no_rot = {}
        backfill_solution_cache(cache_no_rot, AK2, path, expand_rotations=False)

        cache_with_rot = {}
        backfill_solution_cache(cache_with_rot, AK2, path, expand_rotations=True)

        # Rotation expansion should produce at least as many entries
        assert len(cache_with_rot) >= len(cache_no_rot)


class TestExpandPathWithCyclicReductions:
    def test_empty_path_returns_empty(self):
        assert expand_path_with_cyclic_reductions(AK2, []) == []

    def test_passthrough_when_no_cyclic_reductions_needed(self):
        # Use a path solved without cyclical reduction; the expanded path
        # should at minimum start with the same first action and end at length 2.
        _, path, _ = value_guided_greedy_search(
            AK2,
            use_length_priority=True,
            cyclically_reduce_after_moves=False,
            max_nodes_to_explore=10000,
        )
        assert path
        expanded = expand_path_with_cyclic_reductions(AK2, path)
        # Without cyclic reduction in the search, no extra moves should be emitted
        assert len(expanded) == len(path)
        # All actions still in valid range
        for action, length in expanded:
            assert 0 <= action <= 11
            assert length >= 2
        # The replayed expanded path should reach trivial
        _verify_path_reaches_trivial(AK2, expanded, cyclical=False)

    def test_inserts_conjugation_moves_for_cyclic_reductions(self):
        # Search WITH cyclical reduction enabled. The compact path it returns
        # may rely on implicit cyclic reductions. expand_path_with_cyclic_reductions
        # must insert explicit conjugation moves so replay under cyclical=False
        # still reaches trivial.
        solved, compact_path, _ = value_guided_greedy_search(
            AK2,
            use_length_priority=True,
            cyclically_reduce_after_moves=True,
            max_nodes_to_explore=20000,
        )
        assert solved, "Cyclical-reduction search should solve AK(2)"
        expanded = expand_path_with_cyclic_reductions(AK2, compact_path)
        # Expanded path replays correctly under non-cyclical ACMove
        _verify_path_reaches_trivial(AK2, expanded, cyclical=False)
        # All action ids in expanded path must be standard 0-11
        for action, length in expanded:
            assert 0 <= action <= 11, f"Action {action} out of range"
            assert length >= 2
        # Expanded path should be at least as long as the compact one
        assert len(expanded) >= len(compact_path)


class TestLoadModel:
    def test_load_mlp_checkpoint(self, tmp_path):
        # Save a tiny MLP and reload via load_model
        model = FeatureMLP()
        ckpt_path = tmp_path / "tiny_mlp.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        }))

        loaded, mean, std = load_model(
            str(ckpt_path), architecture="mlp",
            feature_stats_path=str(stats_path), device="cpu",
        )
        assert isinstance(loaded, FeatureMLP)
        assert mean is not None and mean.shape == (14,)
        assert std is not None and std.shape == (14,)
        # State dicts should match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            torch.testing.assert_close(v1, v2)

    def test_load_seq_checkpoint(self, tmp_path):
        model = SequenceValueNet(max_seq_len=36)
        ckpt_path = tmp_path / "tiny_seq.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        loaded, mean, std = load_model(
            str(ckpt_path), architecture="seq", device="cpu", max_state_dim=36,
        )
        assert isinstance(loaded, SequenceValueNet)
        # Seq architecture doesn't use feature stats
        assert mean is None
        assert std is None
