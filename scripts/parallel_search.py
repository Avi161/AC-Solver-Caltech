#!/usr/bin/env python3
"""
Parallelized AC-Solver search runner for GPU execution.

Runs V-guided greedy, beam search, and MCTS in parallel across all 1190
Miller-Schupp presentations using multiprocessing + GPU value network.

Designed for: Colab A100 / H100 (or any CUDA-capable GPU).

Usage:
    python scripts/parallel_search.py                        # all algorithms
    python scripts/parallel_search.py --algorithm mcts       # MCTS only
    python scripts/parallel_search.py --algorithm v_guided   # V-guided only
    python scripts/parallel_search.py --max-nodes 100000     # custom budget
    python scripts/parallel_search.py --workers 8            # 8 parallel workers
"""

import os
import sys
import json
import time
import pickle
import argparse
import datetime
import signal
import numpy as np
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove
from value_search.value_guided_search import (
    value_guided_greedy_search, beam_search, load_model,
    backfill_solution_cache,
)
from value_search.mcts import mcts_search


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_presentations():
    """Load all 1190 Miller-Schupp presentations."""
    from ast import literal_eval
    data_path = os.path.join(
        PROJECT_ROOT, "ac_solver", "search", "miller_schupp", "data",
        "all_presentations.txt"
    )
    presentations = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                presentations.append(np.array(literal_eval(line), dtype=np.int8))
    return presentations


def load_greedy_solved_set():
    """Load set of presentations solved by greedy search."""
    from ast import literal_eval
    data_path = os.path.join(
        PROJECT_ROOT, "ac_solver", "search", "miller_schupp", "data",
        "greedy_solved_presentations.txt"
    )
    solved = set()
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                solved.add(tuple(literal_eval(line)))
    return solved


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def search_single_presentation(args):
    """
    Run search on a single presentation. Designed for multiprocessing.Pool.

    Args:
        args: tuple of (idx, presentation_list, algorithm, config_dict)

    Returns:
        dict with results for this presentation
    """
    idx, pres_list, algorithm, config = args

    # Reconstruct presentation from list (numpy arrays can't be pickled easily)
    pres = np.array(pres_list, dtype=np.int8)

    # Load model in each worker process (avoids pickling issues)
    # The model is small enough that this is fast
    device = config['device']
    checkpoint = config['checkpoint']
    architecture = config['architecture']
    feature_stats = config['feature_stats']
    max_nodes = config['max_nodes']

    model, feat_mean, feat_std = load_model(
        checkpoint, architecture, feature_stats, device
    )

    t0 = time.time()

    if algorithm == 'v_guided':
        solved, path, stats = value_guided_greedy_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes, device=device,
            cyclically_reduce_after_moves=config.get('cyclically_reduce', False),
        )
    elif algorithm == 'beam':
        solved, path, stats = beam_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            beam_width=config.get('beam_width', 50),
            max_nodes_to_explore=max_nodes, device=device,
        )
    elif algorithm == 'mcts':
        solved, path, stats = mcts_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes,
            c_explore=config.get('c_explore', 1.41),
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed = time.time() - t0

    result = {
        'idx': idx,
        'solved': solved,
        'path_length': len(path) if solved else 0,
        'nodes_explored': stats.get('nodes_explored', 0),
        'time': elapsed,
    }
    if solved and path:
        result['path'] = [[int(a), int(l)] for a, l in path]

    return result


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_parallel_search(
    presentations,
    algorithm,
    config,
    num_workers=4,
    output_dir=None,
    greedy_solved_set=None,
):
    """
    Run search algorithm in parallel across all presentations.

    Args:
        presentations: list of numpy arrays
        algorithm: 'v_guided', 'beam', or 'mcts'
        config: dict with model paths, device, max_nodes, etc.
        num_workers: number of parallel worker processes
        output_dir: where to save results
        greedy_solved_set: set of tuples for comparing newly solved
    """
    n = len(presentations)
    print(f"\n{'='*60}")
    print(f"  Running {algorithm} with {num_workers} workers")
    print(f"  Max nodes: {config['max_nodes']:,}")
    print(f"  Device: {config['device']}")
    print(f"  Architecture: {config['architecture']}")
    print(f"{'='*60}\n")

    # Prepare args for pool (convert numpy to list for pickling)
    work_items = [
        (i, pres.tolist(), algorithm, config)
        for i, pres in enumerate(presentations)
    ]

    # Progress tracking
    results = []
    solved_count = 0
    newly_solved_count = 0
    t_start = time.time()

    # Open progress file
    progress_path = os.path.join(output_dir, f'{algorithm}_progress.jsonl')
    progress_fh = open(progress_path, 'w')

    try:
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better load balancing
            for result in pool.imap_unordered(search_single_presentation, work_items):
                results.append(result)
                progress_fh.write(json.dumps(result) + '\n')
                progress_fh.flush()

                if result['solved']:
                    solved_count += 1
                    if greedy_solved_set and tuple(presentations[result['idx']]) not in greedy_solved_set:
                        newly_solved_count += 1

                done = len(results)
                if done % 50 == 0 or done == n:
                    elapsed = time.time() - t_start
                    rate = elapsed / done if done > 0 else 0
                    eta = rate * (n - done)
                    eta_str = f"{eta/60:.1f}m" if eta < 3600 else f"{eta/3600:.1f}h"
                    print(f"    {algorithm}: {done}/{n}, solved={solved_count}, "
                          f"new={newly_solved_count}, ETA {eta_str}")
    finally:
        progress_fh.close()

    total_time = time.time() - t_start

    # Sort by index for consistent output
    results.sort(key=lambda r: r['idx'])

    # Compute metrics
    solved_results = [r for r in results if r['solved']]
    path_lengths = [r['path_length'] for r in solved_results]
    metrics = {
        'algorithm': algorithm,
        'solved': len(solved_results),
        'total': len(results),
        'newly_solved': newly_solved_count,
        'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0,
        'max_path_length': int(max(path_lengths)) if path_lengths else 0,
        'total_time': total_time,
        'config': {k: v for k, v in config.items()
                   if k not in ('checkpoint', 'feature_stats')},
    }

    # Print summary
    print(f"\n  => {algorithm}: Solved {metrics['solved']}/{metrics['total']} "
          f"({metrics['newly_solved']} new) in {total_time/60:.1f}m")
    if path_lengths:
        print(f"     Avg path: {metrics['avg_path_length']:.1f}, "
              f"Max path: {metrics['max_path_length']}")

    # Save detailed results
    detail_path = os.path.join(output_dir, f'{algorithm}_details.json')
    with open(detail_path, 'w') as f:
        json.dump(results, f, indent=2)

    metrics_path = os.path.join(output_dir, f'{algorithm}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print newly solved details
    if greedy_solved_set:
        newly = [r for r in solved_results
                 if tuple(presentations[r['idx']]) not in greedy_solved_set]
        if newly:
            print(f"\n  NEWLY SOLVED ({len(newly)} presentations):")
            for r in newly[:20]:
                pres = presentations[r['idx']]
                mrl = len(pres) // 2
                tl = int(np.count_nonzero(pres[:mrl]) + np.count_nonzero(pres[mrl:]))
                print(f"    idx={r['idx']:>4d}, path_len={r['path_length']}, "
                      f"word_len={tl}, nodes={r['nodes_explored']}")
            if len(newly) > 20:
                print(f"    ... and {len(newly)-20} more")

    return results, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parallelized AC-Solver search runner for GPU'
    )
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['all', 'v_guided', 'beam', 'mcts'],
                        help='Which algorithm to run')
    parser.add_argument('--architecture', type=str, default='mlp',
                        choices=['mlp', 'seq'],
                        help='Value network architecture')
    parser.add_argument('--max-nodes', type=int, default=100_000,
                        help='Max nodes to explore per presentation')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--beam-width', type=int, default=50,
                        help='Beam width for beam search')
    parser.add_argument('--c-explore', type=float, default=1.41,
                        help='UCB exploration constant for MCTS')
    parser.add_argument('--cyclically-reduce', action='store_true',
                        help='Apply cyclic reduction after AC moves')
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Resolve workers
    num_workers = args.workers or min(cpu_count(), 8)

    # Resolve checkpoint paths
    arch = args.architecture
    if arch == 'mlp':
        checkpoint = os.path.join(PROJECT_ROOT, 'value_search', 'checkpoints', 'best_mlp.pt')
    else:
        checkpoint = os.path.join(PROJECT_ROOT, 'value_search', 'checkpoints', 'best_seq.pt')
    feature_stats = os.path.join(PROJECT_ROOT, 'value_search', 'checkpoints', 'feature_stats.json')

    # Verify files exist
    for fpath in [checkpoint, feature_stats]:
        if not os.path.exists(fpath):
            print(f"ERROR: Required file not found: {fpath}")
            sys.exit(1)

    config = {
        'device': device,
        'architecture': arch,
        'checkpoint': checkpoint,
        'feature_stats': feature_stats,
        'max_nodes': args.max_nodes,
        'beam_width': args.beam_width,
        'c_explore': args.c_explore,
        'cyclically_reduce': args.cyclically_reduce,
    }

    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(PROJECT_ROOT, 'experiments', 'results', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config_used.json'), 'w') as f:
        json.dump({
            'algorithm': args.algorithm,
            'max_nodes': args.max_nodes,
            'workers': num_workers,
            'device': device,
            'architecture': arch,
            'beam_width': args.beam_width,
            'c_explore': args.c_explore,
            'cyclically_reduce': args.cyclically_reduce,
            'timestamp': timestamp,
        }, f, indent=2)

    print()
    print("=" * 60)
    print("  AC-Solver Parallel Search Runner")
    print("=" * 60)
    print(f"  Device:       {device}")
    print(f"  Workers:      {num_workers}")
    print(f"  Algorithm:    {args.algorithm}")
    print(f"  Architecture: {arch}")
    print(f"  Max nodes:    {args.max_nodes:,}")
    print(f"  Output:       {output_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading presentations...")
    presentations = load_all_presentations()
    greedy_solved_set = load_greedy_solved_set()
    print(f"  Total: {len(presentations)}")
    print(f"  Greedy solved: {len(greedy_solved_set)}")

    # Determine which algorithms to run
    if args.algorithm == 'all':
        algorithms = ['v_guided', 'beam', 'mcts']
    else:
        algorithms = [args.algorithm]

    all_metrics = {}
    all_solved_indices = {}
    t_total_start = time.time()

    for algo in algorithms:
        results, metrics = run_parallel_search(
            presentations, algo, config,
            num_workers=num_workers,
            output_dir=output_dir,
            greedy_solved_set=greedy_solved_set,
        )
        all_metrics[algo] = metrics
        all_solved_indices[algo] = set(
            r['idx'] for r in results if r['solved']
        )

    # Union analysis
    if len(algorithms) > 1:
        union = set()
        for indices in all_solved_indices.values():
            union |= indices
        print(f"\n  UNION of all algorithms: {len(union)}/{len(presentations)} solved")

        # Presentations uniquely solved by each algorithm
        for algo in algorithms:
            unique = all_solved_indices[algo] - set().union(
                *(v for k, v in all_solved_indices.items() if k != algo)
            )
            if unique:
                print(f"    Uniquely solved by {algo}: {len(unique)}")

    total_time = time.time() - t_total_start
    print(f"\n\nAll experiments completed in {total_time/60:.1f}m")
    print(f"Results saved to: {output_dir}/")

    # Save combined metrics
    with open(os.path.join(output_dir, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == '__main__':
    main()
