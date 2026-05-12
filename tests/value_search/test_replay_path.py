"""Unit tests for value_search.replay_path verification utilities.

Covers state_to_algebra, convert_greedy_path, replay_path (in this module),
and verify_single_path (end-to-end).
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.replay_path import (
    AC_MOVE_NAMES,
    convert_greedy_path,
    format_presentation,
    replay_path,
    state_to_algebra,
    verify_single_path,
)
from value_search.value_guided_search import value_guided_greedy_search


AK2 = np.array(
    [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
)


class TestStateToAlgebra:
    def test_trivial_presentation(self):
        # <a, b | a, b>
        state = np.array([1, 0, 0, 2, 0, 0], dtype=np.int8)
        r1, r2 = state_to_algebra(state, max_relator_length=3)
        assert r1 == "a"
        assert r2 == "b"

    def test_inverses(self):
        # r1 = aAbB
        state = np.array([1, -1, 2, -2, 0, 0, 0, 0], dtype=np.int8)
        r1, r2 = state_to_algebra(state, max_relator_length=4)
        assert r1 == "aAbB"
        assert r2 == "ε"  # empty relator → epsilon

    def test_ak2_relators(self):
        r1, r2 = state_to_algebra(AK2, max_relator_length=7)
        assert r1 == "aaBBB"
        assert r2 == "abaBAB"


class TestFormatPresentation:
    def test_default_format(self):
        result = format_presentation("aB", "ab")
        assert "aB" in result and "ab" in result


class TestConvertGreedyPath:
    def test_sentinel_prepended_and_actions_1indexed(self):
        raw_path = [(0, 5), (3, 4), (5, 2)]
        converted = convert_greedy_path(raw_path, initial_length=6)
        assert converted[0] == (0, 6)  # sentinel
        assert converted[1] == (1, 5)
        assert converted[2] == (4, 4)
        assert converted[3] == (6, 2)

    def test_empty_path_returns_only_sentinel(self):
        converted = convert_greedy_path([], initial_length=10)
        assert converted == [(0, 10)]


class TestReplayPathModule:
    def test_replay_solved_ak2(self):
        solved, raw_path, _ = value_guided_greedy_search(
            AK2, use_length_priority=True, max_nodes_to_explore=10000,
        )
        assert solved
        # Convert 0-indexed greedy path → 1-indexed legacy format with sentinel
        initial_length = 11  # AK(2) has length 5 + 6 = 11
        converted = convert_greedy_path(raw_path, initial_length)
        examples = replay_path(AK2, converted, source_idx=99)
        assert examples[-1]["total_length"] == 2
        assert examples[0]["source_idx"] == 99
        for ex in examples:
            assert ex["features"].shape == (14,)


class TestVerifySinglePath:
    def test_verifies_real_solution(self, capsys):
        solved, raw_path, _ = value_guided_greedy_search(
            AK2, use_length_priority=True, max_nodes_to_explore=10000,
        )
        assert solved
        ok = verify_single_path(AK2, raw_path, idx=0, verbose=False, show_algebra=False)
        assert ok is True

    def test_rejects_invalid_path(self):
        # A path with only 1 action that does not reach trivial should fail
        bad_path = [(0, 5)]
        ok = verify_single_path(AK2, bad_path, idx=0, verbose=False)
        assert ok is False


class TestAcMoveNamesTable:
    def test_table_covers_all_12_actions(self):
        for action in range(12):
            assert action in AC_MOVE_NAMES, f"Missing AC move name for action {action}"
            assert isinstance(AC_MOVE_NAMES[action], str)
            assert AC_MOVE_NAMES[action]  # non-empty
