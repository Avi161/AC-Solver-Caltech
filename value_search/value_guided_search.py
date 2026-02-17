"""
Value-guided search algorithms for AC trivialization.

Implements:
  - value_guided_greedy_search: greedy search with pluggable priority (V(P) or length)
  - beam_search: parallel beam search scored by V(P)
"""

import json
import numpy as np
import heapq
import torch

from ac_solver.envs.ac_moves import ACMove
from value_search.feature_extraction import compute_features
from value_search.value_network import FeatureMLP, SequenceValueNet


def load_model(checkpoint_path, architecture='mlp', feature_stats_path=None,
               device='cpu', max_state_dim=72):
    """
    Load a trained value network from checkpoint.

    Returns (model, feat_mean, feat_std) where feat_mean/std are None for seq models.
    """
    if architecture == 'mlp':
        model = FeatureMLP(input_dim=14, hidden_dims=[256, 256, 128], dropout=0.0)
    else:
        model = SequenceValueNet(max_seq_len=max_state_dim, dropout=0.0)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    feat_mean = None
    feat_std = None
    if architecture == 'mlp' and feature_stats_path:
        with open(feature_stats_path, 'r') as f:
            stats = json.load(f)
        feat_mean = np.array(stats['mean'], dtype=np.float32)
        feat_std = np.array(stats['std'], dtype=np.float32)

    return model, feat_mean, feat_std


def _score_states_mlp(states_list, word_lengths_list, model, feat_mean, feat_std, device):
    """Score a batch of states using the MLP model. Returns list of float scores."""
    if not states_list:
        return []
    features = []
    for state, wl in zip(states_list, word_lengths_list):
        mrl = len(state) // 2
        feat = compute_features(state, mrl)
        features.append(feat)
    features = np.stack(features)
    features = (features - feat_mean) / feat_std
    with torch.no_grad():
        t = torch.tensor(features, dtype=torch.float32, device=device)
        preds = model(t).squeeze(-1)
        # Convert from log-space back to original scale
        preds = torch.expm1(preds)
    return preds.cpu().numpy().tolist()


def _score_states_seq(states_list, model, device, max_state_dim=72):
    """Score a batch of states using the sequence model. Returns list of float scores."""
    if not states_list:
        return []
    token_ids = []
    for state in states_list:
        # Pad to max_state_dim
        padded = np.zeros(max_state_dim, dtype=np.int64)
        padded[:len(state)] = state.astype(np.int64) + 2
        padded[len(state):] = 2  # padding token (0 -> 2)
        token_ids.append(padded)
    token_ids = np.stack(token_ids)
    with torch.no_grad():
        t = torch.tensor(token_ids, dtype=torch.long, device=device)
        preds = model(t).squeeze(-1)
        preds = torch.expm1(preds)
    return preds.cpu().numpy().tolist()


def value_guided_greedy_search(
    presentation,
    model=None,
    architecture='mlp',
    feat_mean=None,
    feat_std=None,
    max_nodes_to_explore=10000,
    device='cpu',
    use_length_priority=False,
    cyclically_reduce_after_moves=False,
    verbose=False,
):
    """
    Greedy search with value network priority.

    Mirrors ac_solver/search/greedy.py but uses V(P) as priority instead of length.
    When use_length_priority=True, uses total length (matching original greedy search).

    Parameters:
        presentation: initial presentation (numpy array)
        model: trained value network (None if use_length_priority=True)
        architecture: 'mlp' or 'seq'
        feat_mean, feat_std: feature normalization stats (for MLP)
        max_nodes_to_explore: node budget
        device: 'cpu' or 'cuda'
        use_length_priority: if True, use length as priority (baseline)
        cyclically_reduce_after_moves: apply cyclic reduction after moves
        verbose: print progress

    Returns:
        (solved, path, stats) where:
            solved: bool
            path: list of (action, total_length) tuples (no sentinel)
            stats: dict with nodes_explored, etc.
    """
    presentation = np.array(presentation, dtype=np.int8)
    max_relator_length = len(presentation) // 2
    max_state_dim = len(presentation)

    # Compute initial word lengths
    len_r1 = int(np.count_nonzero(presentation[:max_relator_length]))
    len_r2 = int(np.count_nonzero(presentation[max_relator_length:]))
    word_lengths = [len_r1, len_r2]
    total_initial_length = sum(word_lengths)

    # Compute initial priority
    if use_length_priority:
        init_priority = float(total_initial_length)
    else:
        if architecture == 'mlp':
            scores = _score_states_mlp(
                [presentation], [word_lengths], model, feat_mean, feat_std, device)
        else:
            scores = _score_states_seq([presentation], model, device, max_state_dim)
        init_priority = scores[0]

    # Priority queue: (priority, path_length, state_tuple, word_lengths_tuple, path)
    to_explore = [(init_priority, 0, tuple(presentation), tuple(word_lengths), [])]
    heapq.heapify(to_explore)

    tree_nodes = set()
    tree_nodes.add(tuple(presentation))
    min_length = total_initial_length

    while to_explore:
        _, path_length, state_tuple, wl_tuple, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8)
        word_lengths = list(wl_tuple)

        # Expand: apply all 12 actions
        children = []
        for action in range(12):
            new_state, new_lengths = ACMove(
                action, state, max_relator_length, word_lengths,
                cyclical=cyclically_reduce_after_moves,
            )
            new_length = sum(new_lengths)
            state_tup = tuple(new_state)

            if new_length < min_length:
                min_length = new_length
                if verbose:
                    print(f"New min length: {min_length} (nodes: {len(tree_nodes)})")

            # Check for trivial
            if new_length == 2:
                final_path = path + [(action, new_length)]
                stats = {'nodes_explored': len(tree_nodes), 'min_length': min_length}
                return True, final_path, stats

            if state_tup not in tree_nodes:
                children.append((new_state, new_lengths, action, new_length, state_tup))

        # Score all children in a batch
        if children and not use_length_priority:
            child_states = [c[0] for c in children]
            child_wls = [c[1] for c in children]
            if architecture == 'mlp':
                scores = _score_states_mlp(
                    child_states, child_wls, model, feat_mean, feat_std, device)
            else:
                scores = _score_states_seq(child_states, model, device, max_state_dim)
        else:
            scores = [c[3] for c in children]  # Use total_length as score

        # Push children to heap
        for (new_state, new_lengths, action, new_length, state_tup), score in zip(children, scores):
            tree_nodes.add(state_tup)
            new_path = path + [(action, new_length)]
            heapq.heappush(
                to_explore,
                (score, path_length + 1, state_tup, tuple(new_lengths), new_path),
            )

        if len(tree_nodes) >= max_nodes_to_explore:
            if verbose:
                print(f"Budget exhausted: {len(tree_nodes)} nodes explored")
            break

    stats = {'nodes_explored': len(tree_nodes), 'min_length': min_length}
    return False, [], stats


def beam_search(
    presentation,
    model,
    architecture='mlp',
    feat_mean=None,
    feat_std=None,
    beam_width=50,
    max_nodes_to_explore=10000,
    device='cpu',
    cyclically_reduce_after_moves=False,
    verbose=False,
):
    """
    Beam search: maintain top-k candidates, expand all, score, keep top-k.

    Parameters:
        presentation: initial presentation (numpy array)
        model: trained value network
        architecture: 'mlp' or 'seq'
        feat_mean, feat_std: feature normalization stats
        beam_width: number of candidates to maintain
        max_nodes_to_explore: total node budget
        device: 'cpu' or 'cuda'
        cyclically_reduce_after_moves: apply cyclic reduction
        verbose: print progress

    Returns:
        (solved, path, stats)
    """
    presentation = np.array(presentation, dtype=np.int8)
    max_relator_length = len(presentation) // 2
    max_state_dim = len(presentation)

    # Initialize beam with the starting presentation
    # Each candidate: (state, word_lengths, path)
    len_r1 = int(np.count_nonzero(presentation[:max_relator_length]))
    len_r2 = int(np.count_nonzero(presentation[max_relator_length:]))

    beam = [(presentation.copy(), [len_r1, len_r2], [])]
    visited = set()
    visited.add(tuple(presentation))
    total_nodes = 1

    step = 0
    while beam and total_nodes < max_nodes_to_explore:
        step += 1
        # Expand all candidates
        all_children = []
        for state, wl, path in beam:
            for action in range(12):
                new_state, new_lengths = ACMove(
                    action, state, max_relator_length, wl,
                    cyclical=cyclically_reduce_after_moves,
                )
                new_length = sum(new_lengths)
                state_tup = tuple(new_state)

                # Check trivial
                if new_length == 2:
                    final_path = path + [(action, new_length)]
                    stats = {'nodes_explored': total_nodes, 'steps': step}
                    return True, final_path, stats

                if state_tup not in visited:
                    visited.add(state_tup)
                    total_nodes += 1
                    all_children.append(
                        (new_state, new_lengths, path + [(action, new_length)])
                    )

                    if total_nodes >= max_nodes_to_explore:
                        break
            if total_nodes >= max_nodes_to_explore:
                break

        if not all_children:
            break

        # Score all children
        child_states = [c[0] for c in all_children]
        child_wls = [c[1] for c in all_children]
        if architecture == 'mlp':
            scores = _score_states_mlp(
                child_states, child_wls, model, feat_mean, feat_std, device)
        else:
            scores = _score_states_seq(child_states, model, device, max_state_dim)

        # Keep top-k (lowest score = closest to trivial)
        scored = list(zip(scores, all_children))
        scored.sort(key=lambda x: x[0])
        beam = [(c[0], c[1], c[2]) for _, c in scored[:beam_width]]

        if verbose and step % 10 == 0:
            best_score = scored[0][0] if scored else float('inf')
            min_len = min(sum(c[1]) for c in beam) if beam else 0
            print(f"  Step {step}: beam_size={len(beam)}, nodes={total_nodes}, "
                  f"best_score={best_score:.1f}, min_len={min_len}")

    stats = {'nodes_explored': total_nodes, 'steps': step}
    return False, [], stats
