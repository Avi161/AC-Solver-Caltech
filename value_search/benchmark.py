"""
Benchmarking suite for comparing search algorithms on Miller-Schupp presentations.

Runs greedy (length), V-guided greedy, beam search, and MCTS on all 1190 presentations,
then produces comparison tables and identifies newly solved presentations.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from ast import literal_eval

from ac_solver.search.greedy import greedy_search
from value_search.value_guided_search import (
    value_guided_greedy_search, beam_search, load_model,
)
from value_search.mcts import mcts_search


DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "ac_solver", "search", "miller_schupp", "data"
)


def load_all_presentations(data_dir=None):
    """Load all 1190 Miller-Schupp presentations."""
    if data_dir is None:
        data_dir = DATA_DIR
    filepath = os.path.join(data_dir, "all_presentations.txt")
    presentations = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pres = np.array(literal_eval(line), dtype=np.int8)
                presentations.append(pres)
    return presentations


def load_greedy_solved_set(data_dir=None):
    """Load set of presentations solved by original greedy search."""
    if data_dir is None:
        data_dir = DATA_DIR
    filepath = os.path.join(data_dir, "greedy_solved_presentations.txt")
    solved = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                solved.add(tuple(literal_eval(line)))
    return solved


def run_original_greedy(presentations, max_nodes=1000000):
    """Run the original length-based greedy search."""
    results = []
    solved_count = 0
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path = greedy_search(
            pres, max_nodes_to_explore=max_nodes, verbose=False
        )
        elapsed = time.time() - t0
        # Original greedy returns path with sentinel; remove it
        path_len = len(path) - 1 if solved and path else 0
        results.append({
            'idx': i,
            'solved': solved,
            'path_length': path_len,
            'time': elapsed,
        })
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            print(f"    Greedy: {i+1}/{len(presentations)}, solved={solved_count}")
    return results


def run_vguided_greedy(presentations, model, architecture, feat_mean, feat_std,
                       max_nodes=1000000, device='cpu'):
    """Run value-guided greedy search."""
    results = []
    solved_count = 0
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = value_guided_greedy_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            'idx': i,
            'solved': solved,
            'path_length': len(path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        })
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            print(f"    V-Greedy: {i+1}/{len(presentations)}, solved={solved_count}")
    return results


def run_beam(presentations, model, architecture, feat_mean, feat_std,
             beam_width=50, max_nodes=1000000, device='cpu'):
    """Run beam search."""
    results = []
    solved_count = 0
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = beam_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            beam_width=beam_width, max_nodes_to_explore=max_nodes, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            'idx': i,
            'solved': solved,
            'path_length': len(path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        })
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            print(f"    Beam(k={beam_width}): {i+1}/{len(presentations)}, solved={solved_count}")
    return results


def run_mcts_benchmark(presentations, model, architecture, feat_mean, feat_std,
                       max_nodes=100000, c_explore=1.41, device='cpu'):
    """Run MCTS search."""
    results = []
    solved_count = 0
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = mcts_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes, c_explore=c_explore, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            'idx': i,
            'solved': solved,
            'path_length': len(path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        })
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            print(f"    MCTS(c={c_explore}): {i+1}/{len(presentations)}, solved={solved_count}")
    return results


def compute_metrics(results):
    """Compute aggregate metrics from results."""
    solved = [r for r in results if r['solved']]
    n_solved = len(solved)
    path_lengths = [r['path_length'] for r in solved]
    times = [r['time'] for r in results]
    return {
        'solved': n_solved,
        'total': len(results),
        'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0,
        'median_path_length': float(np.median(path_lengths)) if path_lengths else 0,
        'max_path_length': int(max(path_lengths)) if path_lengths else 0,
        'avg_time': float(np.mean(times)),
        'total_time': float(sum(times)),
        'solved_indices': [r['idx'] for r in solved],
    }


def print_table(all_metrics, greedy_solved_set, presentations):
    """Print formatted comparison table."""
    print(f"\n{'='*85}")
    print(f"{'Algorithm':<30} | {'Solved':>6} | {'Avg Path':>10} | {'Med Path':>10} | {'New':>5} | {'Time':>8}")
    print(f"{'-'*30}-+{'-'*8}+{'-'*12}+{'-'*12}+{'-'*7}+{'-'*10}")

    for name, metrics in all_metrics.items():
        # Count newly solved (not in original greedy solved set)
        newly_solved = 0
        for idx in metrics['solved_indices']:
            if tuple(presentations[idx]) not in greedy_solved_set:
                newly_solved += 1

        print(f"{name:<30} | {metrics['solved']:>6} | "
              f"{metrics['avg_path_length']:>10.1f} | "
              f"{metrics['median_path_length']:>10.1f} | "
              f"{newly_solved:>5} | "
              f"{metrics['total_time']:>7.1f}s")

    print(f"{'='*85}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark search algorithms')
    parser.add_argument('--model-path', type=str, default='value_search/checkpoints/best_mlp.pt')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'seq'])
    parser.add_argument('--feature-stats', type=str, default='value_search/checkpoints/feature_stats.json')
    parser.add_argument('--max-nodes', type=int, default=100000,
                        help='Max nodes for greedy/beam searches')
    parser.add_argument('--mcts-nodes', type=int, default=10000,
                        help='Max nodes for MCTS')
    parser.add_argument('--beam-widths', type=str, default='10,50,100',
                        help='Comma-separated beam widths')
    parser.add_argument('--skip-greedy', action='store_true',
                        help='Skip original greedy (use stored results)')
    parser.add_argument('--skip-mcts', action='store_true',
                        help='Skip MCTS (slow)')
    parser.add_argument('--output-dir', type=str, default='value_search/results')
    args = parser.parse_args()

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    print("Loading presentations...")
    presentations = load_all_presentations()
    greedy_solved_set = load_greedy_solved_set()
    print(f"  Total: {len(presentations)}, GS-solved: {len(greedy_solved_set)}")

    # Load model
    print("Loading model...")
    model, feat_mean, feat_std = load_model(
        args.model_path, args.architecture, args.feature_stats, device
    )

    all_metrics = {}
    all_results = {}

    # 1. Original greedy
    if not args.skip_greedy:
        print("\n--- Running: Greedy (length) ---")
        results = run_original_greedy(presentations, max_nodes=args.max_nodes)
        metrics = compute_metrics(results)
        all_metrics['Greedy (length)'] = metrics
        all_results['greedy_length'] = results
        print(f"  Solved: {metrics['solved']}/{metrics['total']}")

    # 2. V-guided greedy
    print("\n--- Running: V-guided Greedy ---")
    results = run_vguided_greedy(
        presentations, model, args.architecture, feat_mean, feat_std,
        max_nodes=args.max_nodes, device=device,
    )
    metrics = compute_metrics(results)
    all_metrics['V-Greedy (MLP)'] = metrics
    all_results['vguided_greedy'] = results
    print(f"  Solved: {metrics['solved']}/{metrics['total']}")

    # 3. Beam search with various widths
    beam_widths = [int(x) for x in args.beam_widths.split(',')]
    for k in beam_widths:
        print(f"\n--- Running: Beam Search (k={k}) ---")
        results = run_beam(
            presentations, model, args.architecture, feat_mean, feat_std,
            beam_width=k, max_nodes=args.max_nodes, device=device,
        )
        metrics = compute_metrics(results)
        all_metrics[f'Beam (k={k})'] = metrics
        all_results[f'beam_k{k}'] = results
        print(f"  Solved: {metrics['solved']}/{metrics['total']}")

    # 4. MCTS
    if not args.skip_mcts:
        print(f"\n--- Running: MCTS (nodes={args.mcts_nodes}) ---")
        results = run_mcts_benchmark(
            presentations, model, args.architecture, feat_mean, feat_std,
            max_nodes=args.mcts_nodes, c_explore=1.41, device=device,
        )
        metrics = compute_metrics(results)
        all_metrics['MCTS (c=1.41)'] = metrics
        all_results['mcts'] = results
        print(f"  Solved: {metrics['solved']}/{metrics['total']}")

    # Print comparison table
    print_table(all_metrics, greedy_solved_set, presentations)

    # Find newly solved presentations
    print("\n--- Newly Solved Presentations ---")
    for algo_name, metrics in all_metrics.items():
        newly_solved_indices = []
        for idx in metrics['solved_indices']:
            if tuple(presentations[idx]) not in greedy_solved_set:
                newly_solved_indices.append(idx)
        if newly_solved_indices:
            print(f"\n{algo_name} solved {len(newly_solved_indices)} NEW presentations:")
            for idx in newly_solved_indices[:10]:  # Show first 10
                pres = presentations[idx]
                mrl = len(pres) // 2
                total_len = int(np.count_nonzero(pres[:mrl]) + np.count_nonzero(pres[mrl:]))
                print(f"  idx={idx}, array_len={len(pres)}, total_length={total_len}")
            if len(newly_solved_indices) > 10:
                print(f"  ... and {len(newly_solved_indices)-10} more")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'benchmark_results.json')
    save_data = {
        'metrics': all_metrics,
        'config': {
            'max_nodes': args.max_nodes,
            'mcts_nodes': args.mcts_nodes,
            'beam_widths': beam_widths,
            'architecture': args.architecture,
        },
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
