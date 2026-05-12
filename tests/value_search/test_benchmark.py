"""Unit tests for value_search.benchmark helper functions.

The heavy run_* benchmark runners are not exercised end-to-end (they take
minutes per algorithm). The pure helpers and file-IO loaders are covered
here, plus a single-presentation smoke test for each runner.
"""

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.benchmark import (
    compute_metrics,
    load_all_presentations,
    load_greedy_solved_set,
    run_beam,
    run_vguided_greedy,
)
from value_search.value_network import FeatureMLP


AK2 = np.array(
    [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
)


class TestComputeMetrics:
    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["solved"] == 0
        assert metrics["total"] == 0
        assert metrics["avg_path_length"] == 0
        assert metrics["median_path_length"] == 0
        assert metrics["max_path_length"] == 0
        assert metrics["solved_indices"] == []

    def test_mixed_results(self):
        results = [
            {"idx": 0, "solved": True, "path_length": 5, "time": 1.0},
            {"idx": 1, "solved": False, "path_length": 0, "time": 0.5},
            {"idx": 2, "solved": True, "path_length": 9, "time": 2.0},
            {"idx": 3, "solved": True, "path_length": 7, "time": 0.25},
        ]
        m = compute_metrics(results)
        assert m["solved"] == 3
        assert m["total"] == 4
        assert m["avg_path_length"] == pytest.approx((5 + 9 + 7) / 3)
        assert m["median_path_length"] == pytest.approx(7)
        assert m["max_path_length"] == 9
        assert m["avg_time"] == pytest.approx((1.0 + 0.5 + 2.0 + 0.25) / 4)
        assert m["total_time"] == pytest.approx(3.75)
        assert sorted(m["solved_indices"]) == [0, 2, 3]

    def test_all_unsolved(self):
        results = [
            {"idx": 0, "solved": False, "path_length": 0, "time": 0.1},
            {"idx": 1, "solved": False, "path_length": 0, "time": 0.2},
        ]
        m = compute_metrics(results)
        assert m["solved"] == 0
        assert m["total"] == 2
        # Empty path-length list keeps these at 0
        assert m["avg_path_length"] == 0
        assert m["median_path_length"] == 0
        assert m["max_path_length"] == 0
        assert m["solved_indices"] == []


class TestDataLoaders:
    """Real file IO from miller_schupp/data — checks the loaders work."""

    def test_load_all_presentations_returns_arrays(self):
        pres = load_all_presentations()
        assert isinstance(pres, list)
        assert len(pres) > 0
        # Each entry is an int8 numpy array with even length
        for p in pres[:5]:
            assert isinstance(p, np.ndarray)
            assert p.dtype == np.int8
            assert len(p) % 2 == 0

    def test_load_greedy_solved_set_returns_tuples(self):
        solved = load_greedy_solved_set()
        assert isinstance(solved, set)
        assert len(solved) > 0
        # Each entry is a hashable tuple of ints
        sample = next(iter(solved))
        assert isinstance(sample, tuple)
        for v in sample:
            assert isinstance(v, int)


class TestRunWrappersSinglePresentation:
    """Smoke-tests for run_vguided_greedy and run_beam on a single presentation."""

    def test_run_vguided_greedy_shape(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        feat_mean = np.zeros(14, dtype=np.float32)
        feat_std = np.ones(14, dtype=np.float32)
        results = run_vguided_greedy(
            [AK2], model=model, architecture="mlp",
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes=100, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert set(r.keys()) >= {
            "idx", "solved", "path_length", "nodes_explored", "time"
        }
        assert r["idx"] == 0
        assert isinstance(r["solved"], bool)
        assert r["path_length"] >= 0
        assert r["nodes_explored"] > 0

    def test_run_beam_shape(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        feat_mean = np.zeros(14, dtype=np.float32)
        feat_std = np.ones(14, dtype=np.float32)
        results = run_beam(
            [AK2], model=model, architecture="mlp",
            feat_mean=feat_mean, feat_std=feat_std,
            beam_width=4, max_nodes=100, device="cpu",
        )
        assert len(results) == 1
        r = results[0]
        assert r["idx"] == 0
        assert isinstance(r["solved"], bool)
        assert r["nodes_explored"] > 0
