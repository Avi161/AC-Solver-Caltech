"""
Unit tests for MCTS search on AC presentations.

Tests that MCTS can solve known-easy presentations (AK(2))
and that the returned paths are valid.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ac_solver.envs.ac_moves import ACMove
from ac_solver.envs.utils import is_presentation_trivial
from value_search.mcts import mcts_search
from value_search.value_guided_search import load_model


# AK(2): Akbulut-Kirby n=2, known easy presentation
AK2 = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8)

# Simple 2-step solvable presentations (from PPO solutions)
EASY_PRESENTATIONS = {
    # idx 1 in Miller-Schupp: solvable in 2 moves
    "ms_1": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      dtype=np.int8),
}


def verify_path(presentation, path):
    """Replay a path and verify it reaches a trivial state."""
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    len_r1 = int(np.count_nonzero(state[:mrl]))
    len_r2 = int(np.count_nonzero(state[mrl:]))
    wl = [len_r1, len_r2]

    for action, expected_length in path:
        state, wl = ACMove(action, state, mrl, wl, cyclical=False)
        actual_length = sum(wl)
        assert actual_length == expected_length, (
            f"Path length mismatch: expected {expected_length}, got {actual_length}"
        )

    assert is_presentation_trivial(state), (
        f"Path did not reach trivial state. Final state: {list(state)}"
    )
    return True


@pytest.fixture(scope="module")
def mlp_model():
    """Load the MLP value network checkpoint."""
    ckpt = os.path.join(PROJECT_ROOT, "value_search", "checkpoints", "best_mlp.pt")
    stats = os.path.join(PROJECT_ROOT, "value_search", "checkpoints", "feature_stats.json")
    if not os.path.exists(ckpt):
        pytest.skip("MLP checkpoint not found")
    model, feat_mean, feat_std = load_model(ckpt, "mlp", stats)
    return model, feat_mean, feat_std


class TestMCTSSearch:
    """Test MCTS search correctness."""

    def test_mcts_solves_ak2(self, mlp_model):
        """MCTS should solve AK(2) with a small node budget."""
        model, feat_mean, feat_std = mlp_model
        solved, path, stats = mcts_search(
            AK2, model=model, architecture='mlp',
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=10000, c_explore=1.41,
        )
        assert solved, f"MCTS failed to solve AK(2) with 10K nodes. Stats: {stats}"
        assert len(path) > 0, "Path should not be empty"
        verify_path(AK2, path)
        print(f"AK(2) solved in {len(path)} moves, {stats['nodes_explored']} nodes explored")

    def test_mcts_path_format(self, mlp_model):
        """Verify path format is list of (action, total_length) tuples."""
        model, feat_mean, feat_std = mlp_model
        solved, path, stats = mcts_search(
            AK2, model=model, architecture='mlp',
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=10000,
        )
        if solved:
            for entry in path:
                assert isinstance(entry, tuple), f"Path entry should be tuple, got {type(entry)}"
                assert len(entry) == 2, f"Path entry should have 2 elements, got {len(entry)}"
                action, length = entry
                assert 0 <= action <= 11, f"Action {action} out of range [0, 11]"
                assert length >= 2, f"Length {length} should be >= 2"

    def test_mcts_easy_presentation(self, mlp_model):
        """MCTS should solve an easy Miller-Schupp presentation."""
        model, feat_mean, feat_std = mlp_model
        pres = EASY_PRESENTATIONS["ms_1"]
        solved, path, stats = mcts_search(
            pres, model=model, architecture='mlp',
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=5000,
        )
        assert solved, f"MCTS failed to solve easy presentation. Stats: {stats}"
        verify_path(pres, path)
        print(f"Easy presentation solved in {len(path)} moves, "
              f"{stats['nodes_explored']} nodes explored")

    def test_mcts_stats_keys(self, mlp_model):
        """Verify stats dict has expected keys."""
        model, feat_mean, feat_std = mlp_model
        _, _, stats = mcts_search(
            AK2, model=model, architecture='mlp',
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=100,
        )
        assert 'nodes_explored' in stats
        assert 'iterations' in stats
        assert stats['nodes_explored'] > 0
        assert stats['iterations'] > 0

    def test_mcts_with_solution_cache(self, mlp_model):
        """Test that solution cache integration works."""
        model, feat_mean, feat_std = mlp_model
        cache = {}
        solved, path, stats = mcts_search(
            AK2, model=model, architecture='mlp',
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=10000,
            solution_cache=cache,
        )
        assert solved, "MCTS should solve AK(2) even with empty cache"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
