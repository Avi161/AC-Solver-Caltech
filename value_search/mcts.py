"""
Simplified MCTS for AC trivialization using value network leaf evaluation.

No rollouts — the value network V(P) provides leaf evaluations directly.
"""

import math
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from ac_solver.envs.ac_moves import ACMove
from value_search.feature_extraction import compute_features
from value_search.value_guided_search import _check_cache_with_rotations


@dataclass
class MCTSNode:
    """Single node in the MCTS tree."""
    state: tuple
    word_lengths: tuple  # (len_r1, len_r2)
    max_relator_length: int
    parent: Optional['MCTSNode'] = None
    action_from_parent: Optional[int] = None
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior_value: float = float('inf')  # V(state) from network, cached

    @property
    def is_terminal(self):
        return sum(self.word_lengths) == 2

    @property
    def is_expanded(self):
        return len(self.children) > 0

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return self.prior_value
        return self.value_sum / self.visit_count

    def best_child_ucb(self, c_explore=1.41):
        """Select child with highest UCB1 score (lower value = better)."""
        best_score = float('-inf')
        best_child = None
        for child in self.children.values():
            if child.visit_count == 0:
                return child  # Always explore unvisited children first
            exploitation = -child.mean_value  # Negate: lower distance = better
            exploration = c_explore * math.sqrt(
                math.log(self.visit_count) / child.visit_count
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


def select(root, c_explore=1.41):
    """Traverse tree from root to leaf using UCB1."""
    node = root
    while node.is_expanded and not node.is_terminal:
        node = node.best_child_ucb(c_explore)
    return node


def expand(node, visited_global, cyclically_reduce=False):
    """
    Expand node by applying all 12 AC moves.
    Returns list of new child nodes (skips already-visited states).
    """
    if node.is_terminal:
        return []

    state = np.array(node.state, dtype=np.int8)
    wl = list(node.word_lengths)
    mrl = node.max_relator_length
    new_children = []

    for action in range(12):
        new_state, new_lengths = ACMove(
            action, state, mrl, wl, cyclical=cyclically_reduce
        )
        state_tup = tuple(new_state)

        if state_tup in visited_global:
            continue

        visited_global.add(state_tup)
        child = MCTSNode(
            state=state_tup,
            word_lengths=tuple(new_lengths),
            max_relator_length=mrl,
            parent=node,
            action_from_parent=action,
        )
        node.children[action] = child
        new_children.append(child)

    return new_children


def evaluate_leaves(nodes, model, architecture='mlp', feat_mean=None,
                    feat_std=None, device='cpu', max_state_dim=72):
    """Batch evaluate leaf nodes with the value network."""
    if not nodes:
        return

    # Separate terminal from non-terminal
    non_terminal = [n for n in nodes if not n.is_terminal]
    for n in nodes:
        if n.is_terminal:
            n.prior_value = 0.0

    if not non_terminal:
        return

    if architecture == 'mlp':
        features = []
        for n in non_terminal:
            state = np.array(n.state, dtype=np.int8)
            feat = compute_features(state, n.max_relator_length)
            features.append(feat)
        features = np.stack(features)
        features = (features - feat_mean) / feat_std
        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32, device=device)
            preds = model(t).squeeze(-1)
            preds = torch.expm1(preds).cpu().numpy()
    else:
        token_ids = []
        for n in non_terminal:
            state = np.array(n.state, dtype=np.int8)
            padded = np.zeros(max_state_dim, dtype=np.int64)
            padded[:len(state)] = state.astype(np.int64) + 2
            padded[len(state):] = 2
            token_ids.append(padded)
        token_ids = np.stack(token_ids)
        with torch.no_grad():
            t = torch.tensor(token_ids, dtype=torch.long, device=device)
            preds = model(t).squeeze(-1)
            preds = torch.expm1(preds).cpu().numpy()

    for n, v in zip(non_terminal, preds):
        n.prior_value = float(max(v, 0))  # Clamp to non-negative


def backpropagate(node, value):
    """Update visit counts and value sums from leaf to root."""
    current = node
    while current is not None:
        current.visit_count += 1
        current.value_sum += value
        current = current.parent


def extract_path(node):
    """Extract (action, total_length) path from root to this node."""
    path = []
    current = node
    while current.parent is not None:
        path.append((current.action_from_parent, sum(current.word_lengths)))
        current = current.parent
    path.reverse()
    return path


def mcts_search(
    presentation,
    model,
    architecture='mlp',
    feat_mean=None,
    feat_std=None,
    max_nodes_to_explore=10000,
    c_explore=1.41,
    device='cpu',
    cyclically_reduce_after_moves=False,
    verbose=False,
    solution_cache=None,
    time_limit=None,
):
    """
    Simplified MCTS for AC trivialization.

    Parameters:
        presentation: initial state (numpy array)
        model: trained value network
        architecture: 'mlp' or 'seq'
        feat_mean, feat_std: feature normalization stats
        max_nodes_to_explore: total unique states budget
        c_explore: UCB1 exploration constant
        device: 'cpu' or 'cuda'
        cyclically_reduce_after_moves: cyclic reduction flag
        verbose: print progress
        solution_cache: optional dict mapping state_tuple -> path_to_trivial
        time_limit: optional float — stop if wall time exceeds this many seconds

    Returns:
        (solved, path, stats)
    """
    presentation = np.array(presentation, dtype=np.int8)
    max_relator_length = len(presentation) // 2
    max_state_dim = len(presentation)

    len_r1 = int(np.count_nonzero(presentation[:max_relator_length]))
    len_r2 = int(np.count_nonzero(presentation[max_relator_length:]))

    root = MCTSNode(
        state=tuple(presentation),
        word_lengths=(len_r1, len_r2),
        max_relator_length=max_relator_length,
    )

    # Check if starting state (or any rotation) is already in solution cache
    if solution_cache is not None:
        cached_path = _check_cache_with_rotations(
            tuple(presentation), max_relator_length, solution_cache)
        if cached_path is not None:
            stats = {'nodes_explored': 1, 'iterations': 0, 'cache_hit': True}
            return True, cached_path, stats

    # Evaluate root
    evaluate_leaves([root], model, architecture, feat_mean, feat_std,
                    device, max_state_dim)
    root.visit_count = 1
    root.value_sum = root.prior_value

    visited_global = {tuple(presentation)}
    iterations = 0
    found_terminal = None
    max_iterations = max_nodes_to_explore * 10  # Safety limit to prevent infinite loops
    stale_count = 0
    t_start = time.time()

    while len(visited_global) < max_nodes_to_explore and iterations < max_iterations:
        if time_limit is not None and (time.time() - t_start) > time_limit:
            break
        iterations += 1

        # SELECT
        leaf = select(root, c_explore)

        if leaf.is_terminal:
            found_terminal = leaf
            break

        # EXPAND
        new_children = expand(leaf, visited_global, cyclically_reduce_after_moves)

        if not new_children:
            # No new states to explore from this leaf — backprop a high value
            backpropagate(leaf, leaf.prior_value + 50)  # Penalize dead ends
            stale_count += 1
            if stale_count > 1000:
                break  # All reachable states explored
            continue
        stale_count = 0

        # Check for terminal children or cache hits
        cache_hit_path = None
        for child in new_children:
            if child.is_terminal:
                found_terminal = child
                break
            # Check solution cache (with rotation variants)
            if solution_cache is not None:
                cached = _check_cache_with_rotations(
                    child.state, max_relator_length, solution_cache)
                if cached is not None:
                    tree_path = extract_path(child)
                    cache_hit_path = tree_path + cached
                    break

        if found_terminal:
            break

        if cache_hit_path is not None:
            stats = {
                'nodes_explored': len(visited_global),
                'iterations': iterations,
                'cache_hit': True,
            }
            return True, cache_hit_path, stats

        # EVALUATE
        evaluate_leaves(new_children, model, architecture, feat_mean, feat_std,
                        device, max_state_dim)

        # BACKPROPAGATE: use the best child's value
        best_child = min(new_children, key=lambda n: n.prior_value)
        backpropagate(best_child, best_child.prior_value)

        if verbose and iterations % 1000 == 0:
            print(f"  MCTS iter {iterations}: nodes={len(visited_global)}, "
                  f"root_visits={root.visit_count}")

    stats = {
        'nodes_explored': len(visited_global),
        'iterations': iterations,
    }

    if found_terminal:
        path = extract_path(found_terminal)
        return True, path, stats

    # No solution found; return the most-visited path
    return False, [], stats
