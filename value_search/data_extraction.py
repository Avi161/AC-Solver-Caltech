"""
Data extraction pipeline for value network training.

Replays greedy search solution paths to collect intermediate states
with distance-to-trivial labels, then builds a training dataset.
"""

import os
import numpy as np
import pickle
from ast import literal_eval

from ac_solver.envs.ac_moves import ACMove
from ac_solver.envs.utils import is_presentation_trivial
from value_search.feature_extraction import compute_features

MAX_RELATOR_LENGTH = 18
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ac_solver", "search", "miller_schupp", "data"
)


def load_presentations(filepath):
    """Load presentations from a text file. Each line is a Python list literal."""
    presentations = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pres = np.array(literal_eval(line), dtype=np.int8)
                presentations.append(pres)
    return presentations


def load_paths(filepath):
    """Load solution paths from a text file.

    Each line is a list of (action_id, total_length) tuples.
    These paths do NOT include the (-1, init_length) sentinel.
    """
    paths = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                path = literal_eval(line)
                paths.append(path)
    return paths


def replay_path(presentation, path, source_idx=0):
    """
    Replay a solution path, collecting all intermediate states.

    The stored path format is [(sentinel_action, init_length), (action1, len1), ...].
    The first entry is a sentinel whose length matches the initial presentation.
    Actual AC moves start from index 1. Stored actions are 1-indexed.

    Parameters:
        presentation: initial presentation (int8 array)
        path: list of (action_id, total_length) tuples (first entry is sentinel)
        source_idx: index identifying the source presentation

    Returns:
        list of dicts, one per state (including initial), each containing:
            'state': np.ndarray (int8)
            'features': np.ndarray (float32, length 14)
            'steps_remaining': int
            'total_length': int
            'source_idx': int
    """
    state = np.array(presentation, dtype=np.int8)
    max_relator_length = len(state) // 2
    len_r1 = int(np.count_nonzero(state[:max_relator_length]))
    len_r2 = int(np.count_nonzero(state[max_relator_length:]))
    word_lengths = [len_r1, len_r2]

    # The first entry is a sentinel (0, init_length); actual moves start from index 1.
    # Stored paths use 1-indexed actions (1-12), while ACMove uses 0-indexed (0-11).
    moves = path[1:]
    total_steps = len(moves)

    examples = []

    # Record initial state
    examples.append({
        'state': state.copy(),
        'features': compute_features(state, max_relator_length),
        'steps_remaining': total_steps,
        'total_length': sum(word_lengths),
        'source_idx': source_idx,
    })

    # Replay each action (skipping sentinel, converting 1-indexed to 0-indexed)
    for step_idx, (action_id_1indexed, expected_length) in enumerate(moves):
        action_id = action_id_1indexed - 1  # Convert to 0-indexed
        state, word_lengths = ACMove(
            move_id=action_id,
            presentation=state,
            max_relator_length=max_relator_length,
            lengths=word_lengths,
            cyclical=False,
        )
        actual_length = sum(word_lengths)

        steps_remaining = total_steps - step_idx - 1

        examples.append({
            'state': state.copy(),
            'features': compute_features(state, max_relator_length),
            'steps_remaining': steps_remaining,
            'total_length': actual_length,
            'source_idx': source_idx,
        })

    # Verify final state is trivial
    assert sum(word_lengths) == 2, (
        f"Path replay for source_idx={source_idx} did not reach trivial state. "
        f"Final length: {sum(word_lengths)}"
    )

    return examples


def generate_negative_examples(all_presentations, solved_set, label_value=200.0):
    """
    Generate training examples for unsolved presentations with large distance labels.

    Parameters:
        all_presentations: list of all presentations
        solved_set: set of presentation tuples that are solved
        label_value: pseudo-distance for unsolved presentations

    Returns:
        list of dicts with same schema as replay_path output
    """
    examples = []
    neg_idx = 0
    for pres in all_presentations:
        pres_tuple = tuple(pres)
        if pres_tuple not in solved_set:
            pres_arr = np.array(pres, dtype=np.int8)
            mrl = len(pres_arr) // 2
            examples.append({
                'state': pres_arr,
                'features': compute_features(pres_arr, mrl),
                'steps_remaining': label_value,
                'total_length': int(np.count_nonzero(pres_arr[:mrl]) +
                                    np.count_nonzero(pres_arr[mrl:])),
                'source_idx': -1 - neg_idx,  # Negative indices for unsolved
            })
            neg_idx += 1
    return examples


def build_dataset(data_dir=None, output_path=None, include_negatives=True,
                  negative_label=200.0):
    """
    Full pipeline: load data, replay all paths, compute features, save.

    Parameters:
        data_dir: directory containing the text data files
        output_path: where to save the pickle file
        include_negatives: whether to include unsolved presentations as negative examples
        negative_label: pseudo-distance label for unsolved presentations

    Returns:
        dict with keys: 'states', 'features', 'labels', 'source_idx', 'metadata'
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "training_data.pkl"
        )

    print("Loading presentations and paths...")
    all_pres = load_presentations(os.path.join(data_dir, "all_presentations.txt"))
    solved_pres = load_presentations(os.path.join(data_dir, "greedy_solved_presentations.txt"))
    paths = load_paths(os.path.join(data_dir, "greedy_search_paths.txt"))

    print(f"  All presentations: {len(all_pres)}")
    print(f"  Solved presentations: {len(solved_pres)}")
    print(f"  Solution paths: {len(paths)}")

    assert len(solved_pres) == len(paths), (
        f"Mismatch: {len(solved_pres)} solved presentations vs {len(paths)} paths"
    )

    # Build solved set for negative example filtering
    solved_set = set(tuple(p) for p in solved_pres)

    # Replay all paths
    print("Replaying solution paths...")
    all_examples = []
    path_lengths = []
    for i, (pres, path) in enumerate(zip(solved_pres, paths)):
        examples = replay_path(pres, path, source_idx=i)
        all_examples.extend(examples)
        path_lengths.append(len(path) - 1)  # Subtract 1 for sentinel entry
        if (i + 1) % 100 == 0:
            print(f"  Replayed {i + 1}/{len(solved_pres)} paths "
                  f"({len(all_examples)} examples so far)")

    positive_count = len(all_examples)
    print(f"  Total positive examples: {positive_count}")

    # Add negative examples
    negative_count = 0
    if include_negatives:
        print("Generating negative examples for unsolved presentations...")
        neg_examples = generate_negative_examples(
            all_pres, solved_set, label_value=negative_label
        )
        all_examples.extend(neg_examples)
        negative_count = len(neg_examples)
        print(f"  Negative examples: {negative_count}")

    # Convert to arrays. States have variable lengths (36-72) so pad to max.
    print("Converting to arrays...")
    n = len(all_examples)
    max_state_dim = max(len(ex['state']) for ex in all_examples)
    feat_dim = len(all_examples[0]['features'])

    states = np.zeros((n, max_state_dim), dtype=np.int8)
    features = np.zeros((n, feat_dim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    source_idx = np.zeros(n, dtype=np.int32)
    state_lengths = np.zeros(n, dtype=np.int32)  # Track original array length

    for i, ex in enumerate(all_examples):
        s = ex['state']
        state_lengths[i] = len(s)
        states[i, :len(s)] = s
        features[i] = ex['features']
        labels[i] = ex['steps_remaining']
        source_idx[i] = ex['source_idx']

    metadata = {
        'num_solved': len(solved_pres),
        'num_unsolved': len(all_pres) - len(solved_pres),
        'num_all': len(all_pres),
        'positive_examples': positive_count,
        'negative_examples': negative_count,
        'total_examples': n,
        'feature_dim': feat_dim,
        'max_state_dim': max_state_dim,
        'avg_path_length': float(np.mean(path_lengths)),
        'max_path_length': int(np.max(path_lengths)),
        'min_path_length': int(np.min(path_lengths)),
        'median_path_length': float(np.median(path_lengths)),
    }

    dataset = {
        'states': states,
        'features': features,
        'labels': labels,
        'source_idx': source_idx,
        'state_lengths': state_lengths,
        'metadata': metadata,
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nDataset saved to {output_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset Summary")
    print(f"{'='*50}")
    print(f"  Solved: {metadata['num_solved']}/{metadata['num_all']}")
    print(f"  Total training examples: {metadata['total_examples']}")
    print(f"    Positive (from paths): {metadata['positive_examples']}")
    print(f"    Negative (unsolved):   {metadata['negative_examples']}")
    print(f"  Feature dimensions: {metadata['feature_dim']}")
    print(f"  Avg path length: {metadata['avg_path_length']:.1f}")
    print(f"  Max path length: {metadata['max_path_length']}")
    print(f"  Min path length: {metadata['min_path_length']}")
    print(f"  Median path length: {metadata['median_path_length']:.1f}")
    print(f"{'='*50}")

    return dataset


if __name__ == "__main__":
    build_dataset()
