#!/usr/bin/env python3
"""
Iterative Refinement Pipeline for AC-Solver (AlphaZero-style)
==============================================================
Automated loop: search -> collect new paths -> retrain value network -> repeat.

The value network improves each iteration by training on solutions discovered
by its own previous version, progressively learning to evaluate states that
are unreachable by simple greedy search.

Usage:
    python experiments/iterative_refinement.py                     # full run
    python experiments/iterative_refinement.py --max-iterations 1  # single iteration
    python experiments/iterative_refinement.py --resume            # resume interrupted run
    python experiments/iterative_refinement.py --enable-mcts       # include MCTS search
"""

import os
import sys
import json
import time
import subprocess
import argparse
import datetime
import numpy as np
from ast import literal_eval

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from value_search.data_extraction import (
    load_presentations, load_paths, build_dataset_from_dict,
)
from value_search.benchmark import load_all_presentations

DATA_DIR = os.path.join(
    PROJECT_ROOT, "ac_solver", "search", "miller_schupp", "data"
)
REFINEMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "refinement")
STATE_FILE = os.path.join(REFINEMENT_DIR, "refinement_state.json")


# ---------------------------------------------------------------------------
# Path collection utilities
# ---------------------------------------------------------------------------

def load_greedy_paths_as_action_lists():
    """
    Load greedy-solved paths and convert from legacy 1-indexed sentinel format
    to 0-indexed action list format.

    Returns:
        dict: {presentation_tuple: [[action_id_0indexed, total_length], ...]}
    """
    solved_pres = load_presentations(
        os.path.join(DATA_DIR, "greedy_solved_presentations.txt")
    )
    raw_paths = load_paths(
        os.path.join(DATA_DIR, "greedy_search_paths.txt")
    )

    assert len(solved_pres) == len(raw_paths), (
        f"Mismatch: {len(solved_pres)} presentations vs {len(raw_paths)} paths"
    )

    paths_dict = {}
    for pres, raw_path in zip(solved_pres, raw_paths):
        pres_tuple = tuple(pres)
        # Raw format: [(sentinel_action, init_len), (action_1indexed, len), ...]
        # Convert: skip sentinel, subtract 1 from action IDs
        actions = [[a - 1, l] for a, l in raw_path[1:]]
        paths_dict[pres_tuple] = actions

    return paths_dict


def collect_paths_from_jsonl(jsonl_path, all_presentations,
                             exclude_algorithms=None):
    """
    Read solved paths from a .jsonl progress file.

    Parameters:
        jsonl_path: path to *_progress.jsonl file
        all_presentations: list of all 1190 presentations (for idx -> presentation mapping)
        exclude_algorithms: not used here (filtering done at caller level)

    Returns:
        dict: {presentation_tuple: [[action_id, total_length], ...]}
    """
    paths = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip partial/corrupt lines (e.g. from interrupted writes)
            if record.get('solved') and 'path' in record:
                idx = record['idx']
                if idx < len(all_presentations):
                    pres_tuple = tuple(all_presentations[idx])
                    paths[pres_tuple] = record['path']
    return paths


def collect_paths_from_results_dir(results_dir, all_presentations,
                                   exclude_beam=True):
    """
    Collect all solved paths from a results directory.

    Parameters:
        results_dir: path to experiment results directory
        all_presentations: list of all 1190 presentations
        exclude_beam: if True, skip beam search results (long noisy paths)

    Returns:
        dict: {presentation_tuple: [[action_id, total_length], ...]}
    """
    all_paths = {}
    if not os.path.isdir(results_dir):
        return all_paths

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('_progress.jsonl'):
            continue

        # Skip beam search results if requested
        if exclude_beam and fname.startswith('beam_'):
            print(f"    Skipping {fname} (beam search excluded from training)")
            continue

        jsonl_path = os.path.join(results_dir, fname)
        paths = collect_paths_from_jsonl(jsonl_path, all_presentations)
        all_paths.update(paths)
        print(f"    {fname}: {len(paths)} solved paths")

    return all_paths


def merge_path_dicts(existing, new, max_path_length=None):
    """
    Merge two path dictionaries, keeping the shortest path per presentation.

    Parameters:
        existing: current paths dict
        new: newly discovered paths dict
        max_path_length: reject paths longer than this

    Returns:
        tuple: (merged_dict, num_new_presentations, num_shorter_paths)
    """
    merged = dict(existing)
    new_count = 0
    shorter_count = 0

    for pres_tuple, actions in new.items():
        if max_path_length is not None and len(actions) > max_path_length:
            continue

        if pres_tuple not in merged:
            merged[pres_tuple] = actions
            new_count += 1
        elif len(actions) < len(merged[pres_tuple]):
            merged[pres_tuple] = actions
            shorter_count += 1

    return merged, new_count, shorter_count


# ---------------------------------------------------------------------------
# State persistence (for resume)
# ---------------------------------------------------------------------------

def save_state(state):
    """Save refinement state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # Convert tuple keys to strings for JSON serialization
    serializable = {
        'iteration': state['iteration'],
        'solved_per_iteration': state['solved_per_iteration'],
        'total_solved_per_iteration': state['total_solved_per_iteration'],
        'model_paths': state.get('model_paths', []),
        'results_dirs': state.get('results_dirs', []),
    }

    # Save paths dict separately (can be large)
    paths_file = os.path.join(REFINEMENT_DIR, "all_solved_paths.json")
    paths_serializable = {
        str(list(k)): v for k, v in state['solved_paths'].items()
    }
    with open(paths_file, 'w') as f:
        json.dump(paths_serializable, f)

    with open(STATE_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_state():
    """Load refinement state from disk. Returns None if no state exists."""
    if not os.path.exists(STATE_FILE):
        return None

    with open(STATE_FILE, 'r') as f:
        serializable = json.load(f)

    # Load paths
    paths_file = os.path.join(REFINEMENT_DIR, "all_solved_paths.json")
    solved_paths = {}
    if os.path.exists(paths_file):
        with open(paths_file, 'r') as f:
            paths_raw = json.load(f)
        for k_str, v in paths_raw.items():
            pres_tuple = tuple(literal_eval(k_str))
            solved_paths[pres_tuple] = v

    return {
        'iteration': serializable['iteration'],
        'solved_paths': solved_paths,
        'solved_per_iteration': serializable['solved_per_iteration'],
        'total_solved_per_iteration': serializable['total_solved_per_iteration'],
        'model_paths': serializable.get('model_paths', []),
        'results_dirs': serializable.get('results_dirs', []),
    }


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_search(model_checkpoint, results_dir, config_overrides=None,
               enable_mcts=False, max_nodes=None, indices_file=None,
               resume_dir=None):
    """
    Run search experiments via subprocess.

    Parameters:
        model_checkpoint: path to value network checkpoint
        results_dir: where to save results (overrides config output_dir)
        config_overrides: dict of config overrides
        enable_mcts: whether to enable MCTS search
        max_nodes: override max_nodes for all algorithms
        indices_file: path to file listing presentation indices to search
        resume_dir: if set, resume from this existing results directory
    """
    # Build a temporary config for this iteration
    import yaml

    config_path = os.path.join(PROJECT_ROOT, "experiments", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    # Override model checkpoint
    cfg.setdefault('model', {})
    cfg['model']['checkpoint'] = os.path.relpath(model_checkpoint, PROJECT_ROOT)
    cfg['model']['architecture'] = 'mlp'
    cfg['model']['feature_stats'] = 'value_search/checkpoints/feature_stats.json'

    # Disable solution cache (we want the model to learn, not copy)
    cfg['solution_cache_path'] = ''

    # Clear resume_dir — iterative_refinement manages its own resume via --resume-dir flag
    cfg['resume_dir'] = ''

    # Configure algorithms
    cfg['algorithms'] = {
        'greedy': {'enabled': False, 'max_nodes': 1_000_000},
        'bfs': {'enabled': False, 'max_nodes': 100_000},
        'v_guided_greedy': {
            'enabled': True,
            'max_nodes': 1_000_000,
            'cyclically_reduce': True,
        },
        'beam_search': {
            'enabled': True,
            'max_nodes': 1_000_000,
            'beam_widths': [10],
        },
        'mcts': {
            'enabled': enable_mcts,
            'max_nodes': 100_000,
            'c_explore': 1.41,
        },
    }

    # Apply max_nodes override if specified
    if max_nodes is not None:
        for algo_cfg in cfg['algorithms'].values():
            algo_cfg['max_nodes'] = max_nodes

    # Apply any extra overrides
    if config_overrides:
        for key, val in config_overrides.items():
            cfg[key] = val

    # Write temp config
    iter_config_path = os.path.join(results_dir, "iter_config.yaml")
    os.makedirs(results_dir, exist_ok=True)
    with open(iter_config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Run experiment
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "experiments", "run_experiments.py"),
        "--config", iter_config_path,
    ]
    if indices_file is not None:
        cmd.extend(["--indices", indices_file])
    if resume_dir is not None:
        cmd.extend(["--resume-dir", resume_dir])
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"  WARNING: Search exited with code {result.returncode}")

    # Find the actual results directory (timestamped subdirectory)
    output_base = os.path.join(PROJECT_ROOT, cfg.get('output_dir', 'experiments/results'))
    subdirs = sorted([
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d))
        and not d.startswith('refinement')
    ])
    if subdirs:
        return os.path.join(output_base, subdirs[-1])
    return results_dir


def run_training(data_path, save_dir, architecture='mlp', epochs=100):
    """
    Retrain value network via subprocess.

    Parameters:
        data_path: path to training data pickle
        save_dir: where to save checkpoints
        architecture: 'mlp', 'seq', or 'both'
        epochs: max training epochs
    """
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "value_search", "train_value_net.py"),
        "--data-path", data_path,
        "--save-dir", save_dir,
        "--architecture", architecture,
        "--epochs", str(epochs),
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def run_iteration(iteration, state, all_presentations, enable_mcts=False,
                  max_path_length=300, max_nodes=None):
    """
    Execute a single iteration of the refinement loop.

    Returns:
        int: number of newly solved presentations this iteration
    """
    print(f"\n{'='*70}")
    print(f"  ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"  Total solved so far: {len(state['solved_paths'])}/1190")

    # Determine which model to use
    if iteration == 0:
        model_path = os.path.join(
            PROJECT_ROOT, "value_search", "checkpoints", "best_mlp.pt"
        )
        print(f"  Model: {model_path} (original, trained on greedy paths)")
    else:
        model_path = state['model_paths'][-1]
        print(f"  Model: {model_path} (retrained iteration {iteration - 1})")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # --- 1. Run search ---
    print(f"\n  --- Step 1: Search ---")
    iter_results_dir = os.path.join(REFINEMENT_DIR, f"iter_{iteration}")

    # Write indices file so we only search unsolved presentations
    unsolved_indices = [
        i for i, pres in enumerate(all_presentations)
        if tuple(pres) not in state['solved_paths']
    ]
    indices_file = None
    if unsolved_indices and len(unsolved_indices) < len(all_presentations):
        indices_file = os.path.join(iter_results_dir, "unsolved_indices.txt")
        os.makedirs(iter_results_dir, exist_ok=True)
        with open(indices_file, 'w') as f:
            for idx in unsolved_indices:
                f.write(f"{idx}\n")
        print(f"  Searching {len(unsolved_indices)}/{len(all_presentations)} unsolved presentations")

    # Check if a previous run for this iteration exists (for resume)
    prev_results_dir = None
    if state.get('results_dirs') and len(state['results_dirs']) > iteration:
        candidate = state['results_dirs'][iteration]
        if os.path.isdir(candidate):
            prev_results_dir = candidate
            print(f"  Resuming search from: {prev_results_dir}")

    actual_results_dir = run_search(
        model_checkpoint=model_path,
        results_dir=iter_results_dir,
        enable_mcts=enable_mcts,
        max_nodes=max_nodes,
        indices_file=indices_file,
        resume_dir=prev_results_dir,
    )
    state['results_dirs'].append(actual_results_dir)

    # --- 2. Collect paths ---
    print(f"\n  --- Step 2: Collect paths ---")
    print(f"  Reading results from: {actual_results_dir}")
    new_paths = collect_paths_from_results_dir(
        actual_results_dir, all_presentations, exclude_beam=False
    )
    print(f"  Found {len(new_paths)} solved paths this iteration")

    # --- 3. Merge paths ---
    print(f"\n  --- Step 3: Merge paths ---")
    prev_total = len(state['solved_paths'])
    state['solved_paths'], new_count, shorter_count = merge_path_dicts(
        state['solved_paths'], new_paths, max_path_length=max_path_length
    )
    total_now = len(state['solved_paths'])

    print(f"  New presentations solved: {new_count}")
    print(f"  Shorter paths found: {shorter_count}")
    print(f"  Total solved: {prev_total} -> {total_now}/1190")

    state['solved_per_iteration'].append(new_count)
    state['total_solved_per_iteration'].append(total_now)

    if new_count == 0 and shorter_count == 0:
        print(f"\n  No new solutions or improvements. Convergence reached.")
        save_state(state)
        return 0

    # --- 4. Build training data ---
    print(f"\n  --- Step 4: Build training data ---")
    data_path = os.path.join(
        REFINEMENT_DIR, f"training_data_iter_{iteration}.pkl"
    )
    build_dataset_from_dict(
        solved_paths=state['solved_paths'],
        all_presentations=all_presentations,
        output_path=data_path,
        negative_label=200.0,
        max_path_length=max_path_length,
    )

    # --- 5. Retrain value network ---
    print(f"\n  --- Step 5: Retrain value network ---")
    checkpoint_dir = os.path.join(REFINEMENT_DIR, f"checkpoints_iter_{iteration}")
    run_training(
        data_path=data_path,
        save_dir=checkpoint_dir,
        architecture='mlp',
        epochs=100,
    )

    new_model_path = os.path.join(checkpoint_dir, "best_mlp.pt")
    if not os.path.exists(new_model_path):
        raise RuntimeError(f"Training did not produce checkpoint: {new_model_path}")
    state['model_paths'].append(new_model_path)

    # --- 6. Save state ---
    state['iteration'] = iteration + 1
    save_state(state)

    print(f"\n  Iteration {iteration} complete: {new_count} new, "
          f"{total_now}/1190 total")

    return new_count


def main():
    parser = argparse.ArgumentParser(
        description='Iterative refinement pipeline for AC-Solver'
    )
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of refinement iterations')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous state file')
    parser.add_argument('--enable-mcts', action='store_true',
                        help='Include MCTS search (slower, explores differently)')
    parser.add_argument('--max-path-length', type=int, default=300,
                        help='Reject paths longer than this from training data')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Override max_nodes for all algorithms (quick smoke test)')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  AC-Solver Iterative Refinement Pipeline")
    print("=" * 70)
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  MCTS enabled:   {args.enable_mcts}")
    print(f"  Max path length: {args.max_path_length}")
    if args.max_nodes:
        print(f"  Max nodes:      {args.max_nodes:,} (override)")
    print(f"  State file:     {STATE_FILE}")
    print("=" * 70)

    # Load presentations
    all_presentations = load_all_presentations()
    print(f"\n  Total presentations: {len(all_presentations)}")

    # Initialize or resume state
    if args.resume:
        state = load_state()
        if state is None:
            print("  No previous state found. Starting fresh.")
            state = None
        else:
            print(f"  Resumed from iteration {state['iteration']}")
            print(f"  Previously solved: {len(state['solved_paths'])}/1190")

    if not args.resume or state is None:
        # Fresh start: seed with greedy-solved paths
        print("\n  Loading greedy-solved paths as seed...")
        greedy_paths = load_greedy_paths_as_action_lists()
        print(f"  Greedy baseline: {len(greedy_paths)} solved")

        state = {
            'iteration': 0,
            'solved_paths': greedy_paths,
            'solved_per_iteration': [],
            'total_solved_per_iteration': [len(greedy_paths)],
            'model_paths': [],
            'results_dirs': [],
        }

    start_iter = state['iteration']
    t_start = time.time()

    for iteration in range(start_iter, start_iter + args.max_iterations):
        try:
            new_count = run_iteration(
                iteration, state, all_presentations,
                enable_mcts=args.enable_mcts,
                max_path_length=args.max_path_length,
                max_nodes=args.max_nodes,
            )
        except KeyboardInterrupt:
            print(f"\n\n  Interrupted at iteration {iteration}. Saving state...")
            save_state(state)
            print(f"  State saved. Resume with: python experiments/iterative_refinement.py --resume")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR at iteration {iteration}: {e}")
            save_state(state)
            print(f"  State saved. Resume with: python experiments/iterative_refinement.py --resume")
            raise

        if new_count == 0:
            print(f"\n  Converged after {iteration + 1} iterations.")
            break

    # Final summary
    total_time = time.time() - t_start
    print(f"\n\n{'='*70}")
    print(f"  REFINEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Iterations run: {state['iteration'] - start_iter}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Final solved: {len(state['solved_paths'])}/1190")
    print(f"\n  Progress per iteration:")
    for i, (new, total) in enumerate(zip(
        state['solved_per_iteration'],
        state['total_solved_per_iteration'][1:] if len(state['total_solved_per_iteration']) > 1 else state['total_solved_per_iteration']
    )):
        print(f"    Iteration {i}: +{new} new, {total} total")
    print(f"\n  State saved to: {STATE_FILE}")
    print(f"  Resume with: python experiments/iterative_refinement.py --resume")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
