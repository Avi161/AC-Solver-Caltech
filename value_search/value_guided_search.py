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


def _reconstruct_path(parent_map, node_id):
    """Trace parent pointers back to root and return path."""
    path = []
    while node_id is not None:
        action, length, parent_id = parent_map[node_id]
        if parent_id is None:
            break  # root node, no action
        path.append((action, length))
        node_id = parent_id
    path.reverse()
    return path


def _check_cache_with_rotations(state_tup, mrl, solution_cache):
    """
    Check if this state or any cyclic rotation of its relators is in the cache.

    For relator r = a₁a₂...aₙ, cyclic rotation a₂...aₙa₁ is equivalent to
    conjugating by a₁⁻¹. So if we find a rotated version in the cache, we
    can solve the original by prepending the conjugation undo moves.

    Returns the path to trivial if found (including undo moves), or None.
    """
    # Direct check first
    if state_tup in solution_cache:
        return list(solution_cache[state_tup])

    state = np.array(state_tup, dtype=np.int8)
    total_length = int(np.count_nonzero(state[:mrl])) + int(np.count_nonzero(state[mrl:]))

    # Conjugation move lookup: (rel_idx, generator) -> move_id
    # These conjugate r_i by g: r_i → g * r_i * g⁻¹
    CONJ_MOVE = {
        (0, 1): 7, (0, -1): 11, (0, 2): 9, (0, -2): 5,
        (1, 1): 8, (1, -1): 4, (1, 2): 10, (1, -2): 6,
    }

    for rel_idx in range(2):
        offset = rel_idx * mrl
        relator = state[offset:offset + mrl]
        nz = relator[relator != 0]
        L = len(nz)
        if L <= 1:
            continue
        # Skip if first and last letters cancel (rotation won't be free-reduced)
        if nz[-1] + nz[0] == 0:
            continue

        undo_moves = []
        for k in range(1, L):
            gen = int(nz[k - 1])
            move_key = (rel_idx, gen)
            if move_key not in CONJ_MOVE:
                break
            undo_moves.append((CONJ_MOVE[move_key], total_length))

            # Build rotated state via numpy slicing
            rotated_state = state.copy()
            rotated_state[offset:offset + L] = np.concatenate([nz[k:], nz[:k]])
            rot_tup = tuple(rotated_state)

            if rot_tup in solution_cache:
                # Undo rotation: conjugate by a_k, a_{k-1}, ..., a_1 then follow cached path
                return list(reversed(undo_moves)) + list(solution_cache[rot_tup])

    return None


def _expand_cache_with_rotations(cache, state_tup, suffix_path, mrl, cyclically_reduce):
    """
    Expand cache with all cyclic rotations of both relators.

    For relator r = a₁a₂...aₙ, the rotation a₂...aₙa₁ is equivalent to
    conjugating r by a₁⁻¹ (an AC move). So if we know how to solve a state,
    we also know how to solve all its cyclic rotation variants — just prepend
    the conjugation moves to undo the rotation.

    Generates rotations via direct numpy array slicing (no ACMove calls)
    for speed, then computes the appropriate undo conjugation move sequence.
    """
    state = np.array(state_tup, dtype=np.int8)
    total_length = int(np.count_nonzero(state[:mrl])) + int(np.count_nonzero(state[mrl:]))

    # Lookup: conjugation move that does r_i → g * r_i * g⁻¹
    CONJ_MOVE = {
        (0, 1): 7, (0, -1): 11, (0, 2): 9, (0, -2): 5,
        (1, 1): 8, (1, -1): 4, (1, 2): 10, (1, -2): 6,
    }

    for rel_idx in range(2):
        offset = rel_idx * mrl
        relator = state[offset:offset + mrl]
        nz = relator[relator != 0]
        L = len(nz)
        if L <= 1:
            continue

        # If first and last letters cancel (nz[-1] = -nz[0]), rotations
        # won't be free-reduced — skip this relator entirely
        if nz[-1] + nz[0] == 0:
            continue

        # Generate all L-1 cyclic left rotations.
        # Rotation by k: [a_{k+1}, ..., a_n, a_1, ..., a_k]
        # Undo: conjugate by a_k (restores a_k to front, removes from back),
        #   then by a_{k-1}, ..., then by a_1.
        undo_moves = []
        for k in range(1, L):
            gen = int(nz[k - 1])
            move_key = (rel_idx, gen)
            if move_key not in CONJ_MOVE:
                break
            undo_moves.append((CONJ_MOVE[move_key], total_length))

            # Build rotated state directly via numpy slicing
            rotated_state = state.copy()
            rotated_state[offset:offset + L] = np.concatenate([nz[k:], nz[:k]])
            rot_tup = tuple(rotated_state)

            if rot_tup not in cache:
                # Undo path: conjugate by a_k, a_{k-1}, ..., a_1 then follow suffix
                cache[rot_tup] = list(reversed(undo_moves)) + suffix_path


def backfill_solution_cache(cache, presentation, path, cyclically_reduce=False,
                            expand_rotations=True):
    """
    Replay a solved path and cache every intermediate state's suffix.

    After solving presentation P via path [m0, m1, ..., mn], this replays the
    moves to recover intermediate states S0, S1, ..., Sn and caches:
        cache[S0] = [(m0, l0), (m1, l1), ..., (mn, ln)]
        cache[S1] = [(m1, l1), ..., (mn, ln)]
        ...etc

    When expand_rotations=True (default), also caches all cyclic rotation
    variants of each intermediate state's relators. If r0 = aAbbB is cached,
    then AbbBa, bbBaA, etc. are also cached with the appropriate conjugation
    moves prepended. This exploits the fact that cyclic rotation of a relator
    is equivalent to conjugation by one of its letters.
    """
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    len_r1 = int(np.count_nonzero(state[:mrl]))
    len_r2 = int(np.count_nonzero(state[mrl:]))
    wl = [len_r1, len_r2]

    # Cache every intermediate state → remaining suffix
    for i in range(len(path)):
        state_tup = tuple(state)
        suffix = list(path[i:])

        if state_tup not in cache:
            cache[state_tup] = suffix

        # Expand with cyclic rotation variants
        if expand_rotations:
            _expand_cache_with_rotations(cache, state_tup, suffix, mrl, cyclically_reduce)

        action, _ = path[i]
        state, wl = ACMove(action, state, mrl, wl, cyclical=cyclically_reduce)

    return cache


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
    solution_cache=None,
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
        solution_cache: optional dict mapping state_tuple -> path_to_trivial,
            shared across searches to enable cross-presentation memoization

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

    # Parent-pointer tree: node_id -> (action, total_length, parent_id)
    # Root has parent_id=None
    parent_map = {0: (None, total_initial_length, None)}
    next_node_id = 1

    # Map state tuples to node IDs for dedup
    state_to_id = {}
    root_tup = tuple(presentation)
    state_to_id[root_tup] = 0

    # Check if starting state (or any cyclic rotation) is in the solution cache
    if solution_cache is not None:
        cached_path = _check_cache_with_rotations(root_tup, max_relator_length, solution_cache)
        if cached_path is not None:
            stats = {'nodes_explored': 1, 'min_length': 2, 'cache_hit': True}
            return True, cached_path, stats

    # Priority queue: (priority, path_length, node_id, state_tuple, word_lengths_tuple)
    to_explore = [(init_priority, 0, 0, root_tup, tuple(word_lengths))]
    heapq.heapify(to_explore)

    min_length = total_initial_length

    while to_explore:
        _, path_length, cur_node_id, state_tuple, wl_tuple = heapq.heappop(to_explore)
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
                    print(f"New min length: {min_length} (nodes: {len(state_to_id)})")

            # Check for trivial
            if new_length == 2:
                # Build path: parent chain + this final action
                path = _reconstruct_path(parent_map, cur_node_id)
                path.append((action, new_length))
                stats = {'nodes_explored': len(state_to_id), 'min_length': min_length}
                return True, path, stats

            # Check solution cache (with rotation variants) for this child state
            if solution_cache is not None:
                cached = _check_cache_with_rotations(state_tup, max_relator_length, solution_cache)
                if cached is not None:
                    path = _reconstruct_path(parent_map, cur_node_id)
                    path.append((action, new_length))
                    path.extend(cached)
                    stats = {'nodes_explored': len(state_to_id), 'min_length': 2,
                             'cache_hit': True}
                    return True, path, stats

            if state_tup not in state_to_id:
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
            node_id = next_node_id
            next_node_id += 1
            parent_map[node_id] = (action, new_length, cur_node_id)
            state_to_id[state_tup] = node_id
            heapq.heappush(
                to_explore,
                (score, path_length + 1, node_id, state_tup, tuple(new_lengths)),
            )

        if len(state_to_id) >= max_nodes_to_explore:
            if verbose:
                print(f"Budget exhausted: {len(state_to_id)} nodes explored")
            break

    stats = {'nodes_explored': len(state_to_id), 'min_length': min_length}
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
    solution_cache=None,
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
        solution_cache: optional dict mapping state_tuple -> path_to_trivial

    Returns:
        (solved, path, stats)
    """
    presentation = np.array(presentation, dtype=np.int8)
    max_relator_length = len(presentation) // 2
    max_state_dim = len(presentation)

    # Initialize beam with the starting presentation
    len_r1 = int(np.count_nonzero(presentation[:max_relator_length]))
    len_r2 = int(np.count_nonzero(presentation[max_relator_length:]))

    # Check if starting state (or any cyclic rotation) is in the solution cache
    root_tup = tuple(presentation)
    if solution_cache is not None:
        cached_path = _check_cache_with_rotations(root_tup, max_relator_length, solution_cache)
        if cached_path is not None:
            stats = {'nodes_explored': 1, 'steps': 0, 'cache_hit': True}
            return True, cached_path, stats

    # Parent-pointer tree: node_id -> (action, total_length, parent_id)
    parent_map = {0: (None, len_r1 + len_r2, None)}
    next_node_id = 1

    # Each beam candidate: (state, word_lengths, node_id)
    beam = [(presentation.copy(), [len_r1, len_r2], 0)]
    visited = set()
    visited.add(root_tup)
    total_nodes = 1

    step = 0
    while beam and total_nodes < max_nodes_to_explore:
        step += 1
        # Expand all candidates
        all_children = []  # (state, word_lengths, node_id)
        for state, wl, parent_node_id in beam:
            for action in range(12):
                new_state, new_lengths = ACMove(
                    action, state, max_relator_length, wl,
                    cyclical=cyclically_reduce_after_moves,
                )
                new_length = sum(new_lengths)
                state_tup = tuple(new_state)

                # Check trivial
                if new_length == 2:
                    path = _reconstruct_path(parent_map, parent_node_id)
                    path.append((action, new_length))
                    stats = {'nodes_explored': total_nodes, 'steps': step}
                    return True, path, stats

                # Check solution cache (with rotation variants)
                if solution_cache is not None:
                    cached = _check_cache_with_rotations(state_tup, max_relator_length, solution_cache)
                    if cached is not None:
                        path = _reconstruct_path(parent_map, parent_node_id)
                        path.append((action, new_length))
                        path.extend(cached)
                        stats = {'nodes_explored': total_nodes, 'steps': step,
                                 'cache_hit': True}
                        return True, path, stats

                if state_tup not in visited:
                    visited.add(state_tup)
                    total_nodes += 1
                    node_id = next_node_id
                    next_node_id += 1
                    parent_map[node_id] = (action, new_length, parent_node_id)
                    all_children.append((new_state, new_lengths, node_id))

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

        # Prune parent_map: only keep ancestors of beam nodes
        # (prevents unbounded memory growth across steps)
        if step % 50 == 0 and len(parent_map) > total_nodes // 2:
            keep = set()
            for _, _, node_id in beam:
                nid = node_id
                while nid is not None and nid not in keep:
                    keep.add(nid)
                    _, _, nid = parent_map[nid]
            parent_map = {k: v for k, v in parent_map.items() if k in keep}

        if verbose and step % 10 == 0:
            best_score = scored[0][0] if scored else float('inf')
            min_len = min(sum(c[1]) for c in beam) if beam else 0
            print(f"  Step {step}: beam_size={len(beam)}, nodes={total_nodes}, "
                  f"best_score={best_score:.1f}, min_len={min_len}")

    stats = {'nodes_explored': total_nodes, 'steps': step}
    return False, [], stats
