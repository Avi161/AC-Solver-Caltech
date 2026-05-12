"""Unit tests for value_search.data_extraction.

Covers:
  - load_presentations / load_paths
  - replay_path (1-indexed format with sentinel)
  - replay_path_from_action_list (0-indexed format)
  - generate_negative_examples
  - build_dataset_from_dict
"""

import os
import pickle
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.data_extraction import (
    build_dataset_from_dict,
    generate_negative_examples,
    load_paths,
    load_presentations,
    replay_path,
    replay_path_from_action_list,
)
from value_search.value_guided_search import value_guided_greedy_search


AK2 = np.array(
    [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
)


@pytest.fixture(scope="module")
def ak2_path_0indexed():
    """Solve AK(2) with length-priority search; returns the 0-indexed path."""
    solved, path, _ = value_guided_greedy_search(
        AK2, use_length_priority=True, max_nodes_to_explore=10000,
    )
    assert solved, "Fixture must solve AK(2)"
    return path


@pytest.fixture
def ak2_path_1indexed_with_sentinel(ak2_path_0indexed):
    """Convert the 0-indexed path to the 1-indexed format expected by replay_path."""
    mrl = len(AK2) // 2
    init_len = int(np.count_nonzero(AK2[:mrl]) + np.count_nonzero(AK2[mrl:]))
    # Sentinel followed by 1-indexed actions
    return [(0, init_len)] + [(a + 1, length) for a, length in ak2_path_0indexed]


class TestLoadIO:
    def test_load_presentations_round_trip(self, tmp_path):
        # Write a file with two presentation literals
        p = tmp_path / "pres.txt"
        p.write_text("[1, 0, 2, 0]\n[1, 1, 0, 2, -2, 0]\n")
        loaded = load_presentations(str(p))
        assert len(loaded) == 2
        np.testing.assert_array_equal(loaded[0], np.array([1, 0, 2, 0], dtype=np.int8))
        np.testing.assert_array_equal(
            loaded[1], np.array([1, 1, 0, 2, -2, 0], dtype=np.int8)
        )
        assert loaded[0].dtype == np.int8

    def test_load_presentations_skips_blank_lines(self, tmp_path):
        p = tmp_path / "pres.txt"
        p.write_text("\n[1, 0, 2, 0]\n\n[1, 2, 0, 0]\n\n")
        loaded = load_presentations(str(p))
        assert len(loaded) == 2

    def test_load_paths_round_trip(self, tmp_path):
        p = tmp_path / "paths.txt"
        p.write_text("[(3, 4), (5, 2)]\n[(1, 6), (7, 4), (9, 2)]\n")
        paths = load_paths(str(p))
        assert paths == [[(3, 4), (5, 2)], [(1, 6), (7, 4), (9, 2)]]


class TestReplayPath:
    def test_replay_legacy_format_reaches_trivial(
        self, ak2_path_1indexed_with_sentinel
    ):
        examples = replay_path(AK2, ak2_path_1indexed_with_sentinel, source_idx=42)
        # Number of examples = 1 (initial) + len(moves)
        assert len(examples) == len(ak2_path_1indexed_with_sentinel)
        # Final state must be trivial (length 2)
        assert examples[-1]["total_length"] == 2
        # All examples carry the source_idx
        for ex in examples:
            assert ex["source_idx"] == 42
        # Features have length 14
        for ex in examples:
            assert ex["features"].shape == (14,)
        # steps_remaining decreases monotonically and ends at 0
        steps = [ex["steps_remaining"] for ex in examples]
        assert steps == sorted(steps, reverse=True)
        assert steps[-1] == 0

    def test_replay_legacy_asserts_on_bad_path(self):
        # A path that does not reach trivial should raise AssertionError
        bad_path = [(0, 5), (1, 5)]  # sentinel + one move, won't reach length 2
        with pytest.raises(AssertionError):
            replay_path(AK2, bad_path, source_idx=0)


class TestReplayPathFromActionList:
    def test_replay_0indexed_reaches_trivial(self, ak2_path_0indexed):
        examples = replay_path_from_action_list(AK2, ak2_path_0indexed, source_idx=7)
        assert len(examples) == len(ak2_path_0indexed) + 1  # +1 for initial state
        assert examples[-1]["total_length"] == 2
        assert all(ex["source_idx"] == 7 for ex in examples)
        # steps_remaining of initial == len(path), of final == 0
        assert examples[0]["steps_remaining"] == len(ak2_path_0indexed)
        assert examples[-1]["steps_remaining"] == 0

    def test_replay_0indexed_initial_state_preserved(self, ak2_path_0indexed):
        examples = replay_path_from_action_list(AK2, ak2_path_0indexed, source_idx=0)
        # First example is the initial presentation
        np.testing.assert_array_equal(examples[0]["state"], AK2)


class TestGenerateNegativeExamples:
    def test_unsolved_get_label_value(self):
        # All presentations are int8 length-4 arrays
        pres_a = np.array([1, 0, 2, 0], dtype=np.int8)
        pres_b = np.array([1, 1, 2, 0], dtype=np.int8)
        pres_c = np.array([1, -1, 2, -2], dtype=np.int8)
        all_pres = [pres_a, pres_b, pres_c]
        # Only pres_a is "solved"
        solved_set = {tuple(pres_a)}

        negatives = generate_negative_examples(all_pres, solved_set, label_value=137.0)
        assert len(negatives) == 2
        for n in negatives:
            assert n["steps_remaining"] == 137.0
            # Unsolved examples have negative source indices
            assert n["source_idx"] < 0
            assert n["features"].shape == (14,)

    def test_all_solved_returns_empty(self):
        pres_a = np.array([1, 0, 2, 0], dtype=np.int8)
        negatives = generate_negative_examples(
            [pres_a], {tuple(pres_a)}, label_value=100.0
        )
        assert negatives == []


class TestBuildDatasetFromDict:
    def test_builds_pickle_with_expected_keys(
        self, tmp_path, ak2_path_0indexed
    ):
        # Two presentations: one is AK2 (solved) and one is a dummy unsolved
        unsolved_pres = np.zeros_like(AK2)
        unsolved_pres[0] = 1
        unsolved_pres[len(AK2) // 2] = 2  # <a, b | a, b> — trivial-shaped

        solved_paths = {tuple(AK2): ak2_path_0indexed}
        all_pres = [AK2, unsolved_pres]
        output_path = tmp_path / "ds.pkl"

        ds = build_dataset_from_dict(
            solved_paths, all_pres, str(output_path),
            negative_label=99.0,
        )
        # Pickle was written
        assert output_path.exists()

        # Returned dataset keys
        for key in (
            "states", "features", "labels", "source_idx",
            "state_lengths", "metadata",
        ):
            assert key in ds

        # Metadata accuracy
        meta = ds["metadata"]
        assert meta["num_all"] == 2
        assert meta["num_solved"] == 1
        assert meta["num_unsolved"] == 1
        assert meta["feature_dim"] == 14
        assert meta["positive_examples"] == len(ak2_path_0indexed) + 1
        assert meta["negative_examples"] == 1
        assert meta["total_examples"] == meta["positive_examples"] + meta["negative_examples"]

        # Label values
        # At least one label equals the negative label value
        assert (ds["labels"] == 99.0).any()
        # All states pad to max_state_dim
        assert ds["states"].shape[1] == meta["max_state_dim"]

    def test_max_path_length_filters(self, tmp_path, ak2_path_0indexed):
        # With a max_path_length smaller than AK2's solution, the AK2 path
        # is skipped. An unsolved presentation is included so the dataset
        # still has data (and the function doesn't fail on empty arrays).
        unsolved_pres = np.zeros_like(AK2)
        unsolved_pres[0] = 1
        unsolved_pres[len(AK2) // 2] = 2
        solved_paths = {tuple(AK2): ak2_path_0indexed}
        all_pres = [AK2, unsolved_pres]
        output_path = tmp_path / "ds_filtered.pkl"

        ds = build_dataset_from_dict(
            solved_paths, all_pres, str(output_path),
            negative_label=10.0,
            max_path_length=1,  # Smaller than AK2 solution length
        )
        # AK2 path is too long → filtered out
        assert ds["metadata"]["positive_examples"] == 0
        # The other presentation is unsolved → becomes a negative example
        assert ds["metadata"]["negative_examples"] == 1
        assert ds["metadata"]["total_examples"] == 1
