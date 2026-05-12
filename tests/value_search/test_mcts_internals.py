"""Unit tests for value_search.mcts internal helpers.

The end-to-end mcts_search is already covered in tests/test_mcts.py.
This file targets the smaller building blocks that don't require a
trained checkpoint.
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

from value_search.mcts import (
    MCTSNode,
    backpropagate,
    evaluate_leaves,
    expand,
    extract_path,
    mcts_search,
    select,
)
from value_search.value_network import FeatureMLP, SequenceValueNet


AK2 = np.array(
    [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0], dtype=np.int8
)


def _make_root_for(presentation):
    mrl = len(presentation) // 2
    return MCTSNode(
        state=tuple(presentation),
        word_lengths=(
            int(np.count_nonzero(presentation[:mrl])),
            int(np.count_nonzero(presentation[mrl:])),
        ),
        max_relator_length=mrl,
    )


class TestMCTSNode:
    def test_terminal_when_length_two(self):
        node = MCTSNode(state=(1, 0, 2, 0), word_lengths=(1, 1), max_relator_length=2)
        assert node.is_terminal is True

    def test_non_terminal_otherwise(self):
        node = _make_root_for(AK2)
        assert node.is_terminal is False

    def test_initial_state_not_expanded(self):
        node = _make_root_for(AK2)
        assert node.is_expanded is False

    def test_mean_value_falls_back_to_prior(self):
        node = _make_root_for(AK2)
        node.prior_value = 7.5
        # Before any visits, mean_value returns the prior
        assert node.mean_value == pytest.approx(7.5)

    def test_mean_value_uses_value_sum(self):
        node = _make_root_for(AK2)
        node.visit_count = 4
        node.value_sum = 20.0
        assert node.mean_value == pytest.approx(5.0)


class TestExpand:
    def test_expand_produces_up_to_12_children(self):
        root = _make_root_for(AK2)
        visited = {root.state}
        children = expand(root, visited, cyclically_reduce=False)
        # Some duplicates among children may collapse, but for AK(2) we
        # expect at least one valid child and at most 12.
        assert 1 <= len(children) <= 12
        # Children registered on root
        assert len(root.children) == len(children)
        # All children have root as parent and a valid action_from_parent
        for child in children:
            assert child.parent is root
            assert 0 <= child.action_from_parent <= 11

    def test_expand_skips_visited_states(self):
        root = _make_root_for(AK2)
        # Seed visited with a state that one of the children will produce —
        # we don't know which without expansion, but pre-poisoning the set
        # with the root state alone should still leave at least some new
        # children (none of the moves map AK2 back to itself in one step
        # in practice).
        visited = {root.state}
        first = expand(root, visited, cyclically_reduce=False)
        n_first = len(first)
        # Now expand again — visited has all first-level children registered,
        # so a second expand from the same root yields no new children.
        second = expand(root, visited, cyclically_reduce=False)
        assert second == [] or len(second) < n_first

    def test_expand_does_not_run_on_terminal(self):
        node = MCTSNode(state=(1, 0, 2, 0), word_lengths=(1, 1), max_relator_length=2)
        children = expand(node, set(), cyclically_reduce=False)
        assert children == []


class TestEvaluateLeaves:
    def test_mlp_evaluation_assigns_prior_values(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        feat_mean = np.zeros(14, dtype=np.float32)
        feat_std = np.ones(14, dtype=np.float32)
        root = _make_root_for(AK2)
        evaluate_leaves(
            [root], model, architecture="mlp",
            feat_mean=feat_mean, feat_std=feat_std, device="cpu",
        )
        # prior_value should now be a finite, non-negative float
        assert np.isfinite(root.prior_value)
        assert root.prior_value >= 0.0

    def test_terminal_gets_zero_prior(self):
        torch.manual_seed(0)
        model = FeatureMLP()
        terminal = MCTSNode(
            state=(1, 0, 2, 0), word_lengths=(1, 1), max_relator_length=2,
        )
        evaluate_leaves(
            [terminal], model, architecture="mlp",
            feat_mean=np.zeros(14, dtype=np.float32),
            feat_std=np.ones(14, dtype=np.float32),
            device="cpu",
        )
        assert terminal.prior_value == 0.0

    def test_seq_evaluation_runs(self):
        torch.manual_seed(0)
        model = SequenceValueNet(max_seq_len=len(AK2))
        root = _make_root_for(AK2)
        evaluate_leaves(
            [root], model, architecture="seq",
            device="cpu", max_state_dim=len(AK2),
        )
        assert np.isfinite(root.prior_value)
        assert root.prior_value >= 0.0

    def test_empty_input_is_noop(self):
        # Should not raise
        evaluate_leaves(
            [], model=None, architecture="mlp",
            feat_mean=np.zeros(14, dtype=np.float32),
            feat_std=np.ones(14, dtype=np.float32),
        )


class TestBackpropagate:
    def test_updates_visit_counts_and_value_sums_to_root(self):
        root = _make_root_for(AK2)
        child = MCTSNode(
            state=(0,) * len(AK2), word_lengths=(0, 0),
            max_relator_length=len(AK2) // 2, parent=root,
        )
        grandchild = MCTSNode(
            state=(1,) * len(AK2), word_lengths=(0, 0),
            max_relator_length=len(AK2) // 2, parent=child,
        )
        backpropagate(grandchild, value=5.0)
        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 5.0
        assert child.visit_count == 1
        assert child.value_sum == 5.0
        assert root.visit_count == 1
        assert root.value_sum == 5.0


class TestExtractPath:
    def test_root_has_empty_path(self):
        root = _make_root_for(AK2)
        assert extract_path(root) == []

    def test_chain_path_extracted_in_order(self):
        root = _make_root_for(AK2)
        child = MCTSNode(
            state=(2,) * len(AK2), word_lengths=(2, 3),
            max_relator_length=len(AK2) // 2, parent=root,
            action_from_parent=3,
        )
        grandchild = MCTSNode(
            state=(3,) * len(AK2), word_lengths=(1, 1),
            max_relator_length=len(AK2) // 2, parent=child,
            action_from_parent=7,
        )
        path = extract_path(grandchild)
        # Path should be [(child's action, child's total_length),
        #                 (grandchild's action, grandchild's total_length)]
        assert path == [(3, 5), (7, 2)]


class TestSelect:
    def test_select_unexpanded_returns_root(self):
        root = _make_root_for(AK2)
        root.prior_value = 1.0
        chosen = select(root)
        assert chosen is root

    def test_select_picks_unvisited_with_lowest_prior(self):
        # Build root with two children, both unvisited; lower prior_value wins
        root = _make_root_for(AK2)
        root.prior_value = 0.0
        root.visit_count = 1
        child_a = MCTSNode(
            state=(2,) * len(AK2), word_lengths=(2, 2),
            max_relator_length=len(AK2) // 2, parent=root, action_from_parent=0,
        )
        child_a.prior_value = 10.0
        child_b = MCTSNode(
            state=(3,) * len(AK2), word_lengths=(2, 2),
            max_relator_length=len(AK2) // 2, parent=root, action_from_parent=1,
        )
        child_b.prior_value = 1.0
        root.children = {0: child_a, 1: child_b}
        chosen = select(root)
        assert chosen is child_b
