#!/usr/bin/env python3
"""
AC-Solver Experiment Runner
============================
Runs paper baselines (greedy, BFS) and our V-guided methods side-by-side.
Reads config from experiments/config.yaml. Saves results incrementally.

Usage:
    python experiments/run_experiments.py                    # full run from config
    python experiments/run_experiments.py --max-nodes 10000  # quick test override
    python experiments/run_experiments.py --config my.yaml   # custom config

Results saved to: experiments/results/<timestamp>/
"""

import os
import sys
import json
import time
import pickle
import signal
import atexit
import argparse
import datetime
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch


from ac_solver.search.greedy import greedy_search
from ac_solver.search.breadth_first import bfs
from value_search.value_guided_search import (
    value_guided_greedy_search, beam_search, load_model,
    backfill_solution_cache, expand_path_with_cyclic_reductions,
)
from value_search.mcts import mcts_search
from value_search.benchmark import load_all_presentations, load_greedy_solved_set


# ---------------------------------------------------------------------------
# Defaults (used if config.yaml missing or incomplete)
# ---------------------------------------------------------------------------
DEFAULTS = {
    'output_dir': 'experiments/results',
    'save_incremental': True,
    'solution_cache_path': 'experiments/solution_cache.pkl',
    'device': 'auto',
    'model': {
        'architecture': 'mlp',
        'checkpoint': 'value_search/checkpoints/best_mlp.pt',
        'feature_stats': 'value_search/checkpoints/feature_stats.json',
    },
    'algorithms': {
        'greedy': {'enabled': True, 'max_nodes': 1_000_000},
        'bfs': {'enabled': True, 'max_nodes': 100_000},
        'v_guided_greedy': {'enabled': True, 'max_nodes': 1_000_000},
        'beam_search': {'enabled': True, 'max_nodes': 1_000_000, 'beam_widths': [10, 50, 100]},
        'mcts': {'enabled': False, 'max_nodes': 100_000, 'c_explore': 1.41},
    },
}


def load_config(config_path):
    """Load YAML config, falling back to defaults for missing keys."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"[WARN] Config not found at {config_path}, using defaults.")
        cfg = {}

    # Merge with defaults
    for key, default_val in DEFAULTS.items():
        if key not in cfg:
            cfg[key] = default_val
        elif isinstance(default_val, dict):
            for sub_key, sub_val in default_val.items():
                if sub_key not in cfg[key]:
                    cfg[key][sub_key] = sub_val

    # Ensure all algorithm sections have defaults
    for algo, algo_defaults in DEFAULTS['algorithms'].items():
        if algo not in cfg['algorithms']:
            cfg['algorithms'][algo] = algo_defaults
        else:
            for k, v in algo_defaults.items():
                if k not in cfg['algorithms'][algo]:
                    cfg['algorithms'][algo][k] = v

    return cfg


def get_device(device_str):
    """Resolve device string to torch device."""
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def fmt_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def eta_str(elapsed, done, total):
    """Estimate remaining time."""
    if done == 0:
        return "?"
    rate = elapsed / done
    remaining = rate * (total - done)
    return fmt_time(remaining)


# ---------------------------------------------------------------------------
# Persistent solution cache
# ---------------------------------------------------------------------------

def load_solution_cache(path):
    """Load solution cache from pickle file. Returns empty dict if not found."""
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded solution cache: {len(cache)} entries from {path}")
        return cache
    return {}


def save_solution_cache(cache, path):
    """Save solution cache to pickle file."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# .jsonl progress writer
# ---------------------------------------------------------------------------

def write_jsonl_line(fh, record):
    """Write a single JSON record to a .jsonl file handle and flush."""
    fh.write(json.dumps(record) + '\n')
    fh.flush()


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_greedy(presentations, max_nodes, progress_fh=None, original_indices=None):
    """Run paper's length-based greedy search."""
    results = []
    solved_count = 0
    t_start = time.time()
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path = greedy_search(pres, max_nodes_to_explore=max_nodes)
        elapsed = time.time() - t0
        path_len = len(path) - 1 if solved and path else 0  # remove sentinel
        result = {
            'idx': original_indices[i] if original_indices is not None else i, 'solved': solved,
            'path_length': path_len, 'time': elapsed,
        }
        if solved and path:
            # Store full move sequence: list of [action_id, length_after] pairs
            # Skip sentinel (-1, initial_length) at index 0
            result['path'] = [[int(a), int(l)] for a, l in path[1:]]
        results.append(result)
        if progress_fh:
            write_jsonl_line(progress_fh, result)
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            total_elapsed = time.time() - t_start
            print(f"    Greedy: {i+1}/{len(presentations)}, "
                  f"solved={solved_count}, "
                  f"ETA {eta_str(total_elapsed, i+1, len(presentations))}")
    return results


def run_bfs_search(presentations, max_nodes, progress_fh=None, original_indices=None):
    """Run paper's BFS search."""
    results = []
    solved_count = 0
    t_start = time.time()
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path = bfs(pres, max_nodes_to_explore=max_nodes)
        elapsed = time.time() - t0
        # BFS path includes sentinel (-1, initial_length) at index 0 when solved
        path_len = len(path) - 1 if solved and path else 0
        result = {
            'idx': original_indices[i] if original_indices is not None else i, 'solved': solved,
            'path_length': path_len, 'time': elapsed,
        }
        if solved and path:
            # Store full move sequence: list of [action_id, length_after] pairs
            # Skip sentinel (-1, initial_length) at index 0
            result['path'] = [[int(a), int(l)] for a, l in path[1:]]
        results.append(result)
        if progress_fh:
            write_jsonl_line(progress_fh, result)
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            total_elapsed = time.time() - t_start
            print(f"    BFS: {i+1}/{len(presentations)}, "
                  f"solved={solved_count}, "
                  f"ETA {eta_str(total_elapsed, i+1, len(presentations))}")
    return results


def run_vguided(presentations, model, architecture, feat_mean, feat_std,
                max_nodes, device, solution_cache=None, progress_fh=None,
                cyclically_reduce=False, original_indices=None):
    """Run our V-guided greedy search."""
    results = []
    solved_count = 0
    cache_hits = 0
    t_start = time.time()
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = value_guided_greedy_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes, device=device,
            cyclically_reduce_after_moves=cyclically_reduce,
            solution_cache=solution_cache,
        )
        elapsed = time.time() - t0

        # Expand path with explicit cyclic reduction steps for traceability
        if solved and path and cyclically_reduce:
            expanded_path = expand_path_with_cyclic_reductions(pres, path)
        else:
            expanded_path = path

        result = {
            'idx': original_indices[i] if original_indices is not None else i, 'solved': solved,
            'path_length': len(expanded_path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        }
        if solved and path:
            result['path'] = [[int(a), int(l)] for a, l in expanded_path]
            # Backfill cache with compact path (no cyclic reduction entries)
            if solution_cache is not None:
                backfill_solution_cache(
                    solution_cache, pres, path,
                    cyclically_reduce=cyclically_reduce,
                )
            if stats.get('cache_hit'):
                cache_hits += 1
        results.append(result)
        if progress_fh:
            write_jsonl_line(progress_fh, result)
        if solved:
            solved_count += 1
        orig_idx = original_indices[i] if original_indices is not None else i
        status = f"SOLVED path={result['path_length']}" if solved else "unsolved"
        total_elapsed = time.time() - t_start
        print(f"  [idx={orig_idx}] {status}, nodes={stats['nodes_explored']}, "
              f"t={elapsed:.1f}s | {i+1}/{len(presentations)} solved={solved_count} "
              f"ETA {eta_str(total_elapsed, i+1, len(presentations))}")
    return results


def run_beam_search(presentations, model, architecture, feat_mean, feat_std,
                    beam_width, max_nodes, device, solution_cache=None,
                    progress_fh=None, original_indices=None):
    """Run our beam search."""
    results = []
    solved_count = 0
    cache_hits = 0
    t_start = time.time()
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = beam_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            beam_width=beam_width, max_nodes_to_explore=max_nodes, device=device,
            solution_cache=solution_cache,
        )
        elapsed = time.time() - t0
        result = {
            'idx': original_indices[i] if original_indices is not None else i, 'solved': solved,
            'path_length': len(path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        }
        if solved and path:
            result['path'] = [[int(a), int(l)] for a, l in path]
            if solution_cache is not None:
                backfill_solution_cache(solution_cache, pres, path)
            if stats.get('cache_hit'):
                cache_hits += 1
        results.append(result)
        if progress_fh:
            write_jsonl_line(progress_fh, result)
        if solved:
            solved_count += 1
        orig_idx = original_indices[i] if original_indices is not None else i
        status = f"SOLVED path={result['path_length']}" if solved else "unsolved"
        total_elapsed = time.time() - t_start
        print(f"  [idx={orig_idx}] {status}, nodes={stats['nodes_explored']}, "
              f"t={elapsed:.1f}s | {i+1}/{len(presentations)} solved={solved_count} "
              f"ETA {eta_str(total_elapsed, i+1, len(presentations))}")
    return results


def run_mcts_search(presentations, model, architecture, feat_mean, feat_std,
                    max_nodes, c_explore, device, solution_cache=None,
                    progress_fh=None, original_indices=None):
    """Run our MCTS search."""
    results = []
    solved_count = 0
    t_start = time.time()
    for i, pres in enumerate(presentations):
        t0 = time.time()
        solved, path, stats = mcts_search(
            pres, model=model, architecture=architecture,
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=max_nodes, c_explore=c_explore, device=device,
            solution_cache=solution_cache,
        )
        elapsed = time.time() - t0
        result = {
            'idx': original_indices[i] if original_indices is not None else i, 'solved': solved,
            'path_length': len(path) if solved else 0,
            'nodes_explored': stats['nodes_explored'],
            'time': elapsed,
        }
        if solved and path:
            result['path'] = [[int(a), int(l)] for a, l in path]
            # Backfill shared cache so other algorithms benefit
            if solution_cache is not None:
                backfill_solution_cache(solution_cache, pres, path)
        results.append(result)
        if progress_fh:
            write_jsonl_line(progress_fh, result)
        if solved:
            solved_count += 1
        if (i + 1) % 100 == 0:
            total_elapsed = time.time() - t_start
            print(f"    MCTS: {i+1}/{len(presentations)}, "
                  f"solved={solved_count}, "
                  f"ETA {eta_str(total_elapsed, i+1, len(presentations))}")
    return results


# ---------------------------------------------------------------------------
# Metrics & display
# ---------------------------------------------------------------------------

def count_completed_in_jsonl(path):
    """Count valid completed entries in an existing progress JSONL file."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass  # Partial line from interrupted write
    return count


def remap_indices(results, original_indices):
    """Remap result idx fields to original presentation indices."""
    for r in results:
        r['idx'] = original_indices[r['idx']]
    return results


def compute_metrics(results):
    """Compute aggregate metrics."""
    solved = [r for r in results if r['solved']]
    path_lengths = [r['path_length'] for r in solved]
    times = [r['time'] for r in results]
    return {
        'solved': len(solved),
        'total': len(results),
        'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0,
        'median_path_length': float(np.median(path_lengths)) if path_lengths else 0,
        'max_path_length': int(max(path_lengths)) if path_lengths else 0,
        'avg_time': float(np.mean(times)),
        'total_time': float(sum(times)),
        'solved_indices': [r['idx'] for r in solved],
    }


def print_comparison_table(all_metrics, greedy_solved_set, presentations):
    """Print side-by-side comparison table."""
    print()
    print("=" * 95)
    print(f"{'Algorithm':<35} | {'Solved':>10} | {'Avg Path':>10} | "
          f"{'Med Path':>10} | {'New':>5} | {'Time':>10}")
    print(f"{'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}-+-{'-'*10}")

    for name, metrics in all_metrics.items():
        newly_solved = 0
        for idx in metrics['solved_indices']:
            if tuple(presentations[idx]) not in greedy_solved_set:
                newly_solved += 1

        solved_str = f"{metrics['solved']}/{metrics['total']}"
        print(f"{name:<35} | {solved_str:>10} | "
              f"{metrics['avg_path_length']:>10.1f} | "
              f"{metrics['median_path_length']:>10.1f} | "
              f"{newly_solved:>5} | "
              f"{fmt_time(metrics['total_time']):>10}")

    print("=" * 95)
    print()
    print("'New' = presentations solved that original greedy (length) cannot solve.")


def print_newly_solved(all_metrics, greedy_solved_set, presentations):
    """Print details about newly solved presentations."""
    print()
    print("-" * 60)
    print("NEWLY SOLVED PRESENTATIONS (not solved by paper's greedy)")
    print("-" * 60)

    for algo_name, metrics in all_metrics.items():
        newly = []
        for idx in metrics['solved_indices']:
            if tuple(presentations[idx]) not in greedy_solved_set:
                newly.append(idx)
        if newly:
            print(f"\n  {algo_name}: {len(newly)} new presentations")
            for idx in newly[:15]:
                pres = presentations[idx]
                mrl = len(pres) // 2
                total_len = int(np.count_nonzero(pres[:mrl]) + np.count_nonzero(pres[mrl:]))
                print(f"    idx={idx:>4d}, word_length={total_len}")
            if len(newly) > 15:
                print(f"    ... and {len(newly)-15} more")
        else:
            print(f"\n  {algo_name}: 0 new presentations")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_results(output_dir, all_metrics, all_details, config):
    """Save metrics and per-presentation details."""
    os.makedirs(output_dir, exist_ok=True)

    # Main results summary
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            k: v for k, v in config.items()
            if k != 'algorithms'
        },
        'algorithms_config': config['algorithms'],
        'metrics': {
            name: {k: v for k, v in m.items() if k != 'solved_indices'}
            for name, m in all_metrics.items()
        },
    }
    summary_path = os.path.join(output_dir, 'results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-algorithm detail files
    for algo_key, details in all_details.items():
        detail_path = os.path.join(output_dir, f'{algo_key}_details.json')
        with open(detail_path, 'w') as f:
            json.dump(details, f, indent=2)

    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AC-Solver Experiment Runner: paper baselines vs V-guided methods'
    )
    parser.add_argument('--config', type=str,
                        default=os.path.join(PROJECT_ROOT, 'experiments', 'config.yaml'),
                        help='Path to config YAML')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Override max_nodes for ALL algorithms (quick test)')
    parser.add_argument('--only-unsolved', action='store_true',
                        help='Skip presentations already in solution cache')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override model checkpoint path (e.g. for retrained models)')
    parser.add_argument('--indices', type=str, default=None,
                        help='Path to file listing presentation indices to run (one per line)')
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Resume from existing results dir (skip already-processed presentations)')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.max_nodes is not None:
        print(f"[CLI override] max_nodes = {args.max_nodes} for all algorithms")
        for algo_cfg in cfg['algorithms'].values():
            algo_cfg['max_nodes'] = args.max_nodes
    if args.checkpoint is not None:
        print(f"[CLI override] checkpoint = {args.checkpoint}")
        cfg['model']['checkpoint'] = args.checkpoint

    # Resolve device
    device = get_device(cfg['device'])

    # Create output directory (or reuse on resume)
    # CLI --resume-dir overrides config resume_dir
    resume_dir = args.resume_dir or cfg.get('resume_dir', '')
    if resume_dir:
        output_dir = resume_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        timestamp = os.path.basename(output_dir)
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join(PROJECT_ROOT, cfg['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config copy
    config_copy_path = os.path.join(output_dir, 'config_used.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Resolve solution cache path
    cache_path = cfg.get('solution_cache_path', 'experiments/solution_cache.pkl')
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(PROJECT_ROOT, cache_path)

    # Header
    print()
    print("=" * 60)
    print("  AC-Solver Experiment Runner")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Config:     {args.config}")
    print(f"  Output:     {output_dir}")
    if resume_dir:
        print(f"  Resume:     YES (continuing from existing progress)")
    print(f"  Cache:      {cache_path}")
    print(f"  Started:    {timestamp}")
    print("=" * 60)
    print()

    # Load presentations
    print("Loading presentations...")
    presentations = load_all_presentations()
    greedy_solved_set = load_greedy_solved_set()
    print(f"  Total presentations: {len(presentations)}")
    print(f"  Paper greedy solved: {len(greedy_solved_set)}")

    # Load model (for V-guided methods)
    model_cfg = cfg['model']
    model, feat_mean, feat_std = None, None, None
    needs_model = any(
        cfg['algorithms'].get(a, {}).get('enabled', False)
        for a in ['v_guided_greedy', 'beam_search', 'mcts']
    )
    if needs_model:
        # Preflight: check that checkpoint files exist
        checkpoint_path = os.path.join(PROJECT_ROOT, model_cfg['checkpoint'])
        stats_path = os.path.join(PROJECT_ROOT, model_cfg['feature_stats'])
        missing = []
        if not os.path.exists(checkpoint_path):
            missing.append(model_cfg['checkpoint'])
        if not os.path.exists(stats_path):
            missing.append(model_cfg['feature_stats'])
        if missing:
            print("\nERROR: Required model files not found:")
            for f in missing:
                print(f"  - {f}")
            print("\nTo generate them, run these two commands:")
            print("  python value_search/data_extraction.py")
            print("  python value_search/train_value_net.py --architecture both")
            print("\nOr disable V-guided methods in config.yaml and run baselines only.")
            sys.exit(1)

        print(f"Loading model ({model_cfg['architecture']})...")
        model, feat_mean, feat_std = load_model(
            checkpoint_path,
            model_cfg['architecture'],
            stats_path,
            device,
        )
        print("  Model loaded.")

    all_metrics = {}
    all_details = {}
    algos = cfg['algorithms']
    experiment_start = time.time()

    # Shared solution cache: state_tuple -> path_to_trivial
    # Populated by solved presentations, enables cross-search memoization.
    # Loaded from disk if available (persists across runs).
    solution_cache = load_solution_cache(cache_path)
    if not cache_path:
        solution_cache = None  # Disable cache entirely when path is empty

    # Filter to unsolved presentations only (if requested)
    original_indices = list(range(len(presentations)))
    if args.only_unsolved and solution_cache:
        unsolved = []
        unsolved_indices = []
        for i, pres in enumerate(presentations):
            if tuple(pres) not in solution_cache:
                unsolved.append(pres)
                unsolved_indices.append(i)
        print(f"\n  --only-unsolved: {len(presentations) - len(unsolved)} already solved, "
              f"running on {len(unsolved)} unsolved")
        presentations = unsolved
        original_indices = unsolved_indices

    if args.indices is not None:
        with open(args.indices, 'r') as f:
            selected_indices = [int(line.strip()) for line in f if line.strip()]
        selected = [presentations[i] for i in selected_indices if i < len(presentations)]
        print(f"\n  --indices: running on {len(selected)}/{len(presentations)} presentations")
        original_indices = [original_indices[i] for i in selected_indices if i < len(presentations)]
        presentations = selected

    # Register emergency save on Ctrl+C / SIGTERM so cache is never lost
    def _emergency_save(signum, frame):
        if solution_cache:
            print(f"\n  [interrupted] Saving solution cache ({len(solution_cache)} entries)...")
            save_solution_cache(solution_cache, cache_path)
            print(f"  [cache saved to {cache_path}]")
        sys.exit(1)

    signal.signal(signal.SIGINT, _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)
    atexit.register(lambda: save_solution_cache(solution_cache, cache_path) if solution_cache else None)

    # ---- 1. Greedy (paper baseline) ----
    if algos['greedy']['enabled']:
        max_n = algos['greedy']['max_nodes']
        print(f"\n{'='*60}")
        print(f"  [1] GREEDY (paper baseline) — {max_n:,} nodes")
        print(f"{'='*60}")
        progress_path = os.path.join(output_dir, 'greedy_progress.jsonl')
        skip = count_completed_in_jsonl(progress_path) if resume_dir else 0
        if skip >= len(presentations):
            print(f"  Already complete ({skip} entries). Skipping.")
        else:
            if skip > 0:
                print(f"  Resuming: skipping {skip} already processed")
            algo_pres = presentations[skip:]
            algo_idx = original_indices[skip:]
            with open(progress_path, 'a' if skip > 0 else 'w') as pfh:
                results = run_greedy(algo_pres, max_n, progress_fh=pfh,
                                     original_indices=algo_idx)
            remap_indices(results, algo_idx)
            metrics = compute_metrics(results)
            name = f"Greedy (paper, {max_n//1000}K)"
            all_metrics[name] = metrics
            all_details['greedy'] = results
            print(f"  => Solved: {metrics['solved']}/{metrics['total']} "
                  f"in {fmt_time(metrics['total_time'])}")
            if cfg['save_incremental']:
                save_results(output_dir, all_metrics, all_details, cfg)
                print(f"  [saved incrementally]")

    # ---- 2. BFS (paper baseline) ----
    if algos['bfs']['enabled']:
        max_n = algos['bfs']['max_nodes']
        print(f"\n{'='*60}")
        print(f"  [2] BFS (paper baseline) — {max_n:,} nodes")
        print(f"{'='*60}")
        progress_path = os.path.join(output_dir, 'bfs_progress.jsonl')
        skip = count_completed_in_jsonl(progress_path) if resume_dir else 0
        if skip >= len(presentations):
            print(f"  Already complete ({skip} entries). Skipping.")
        else:
            if skip > 0:
                print(f"  Resuming: skipping {skip} already processed")
            algo_pres = presentations[skip:]
            algo_idx = original_indices[skip:]
            with open(progress_path, 'a' if skip > 0 else 'w') as pfh:
                results = run_bfs_search(algo_pres, max_n, progress_fh=pfh,
                                         original_indices=algo_idx)
            remap_indices(results, algo_idx)
            metrics = compute_metrics(results)
            name = f"BFS (paper, {max_n//1000}K)"
            all_metrics[name] = metrics
            all_details['bfs'] = results
            print(f"  => Solved: {metrics['solved']}/{metrics['total']} "
                  f"in {fmt_time(metrics['total_time'])}")
            if cfg['save_incremental']:
                save_results(output_dir, all_metrics, all_details, cfg)
                print(f"  [saved incrementally]")

    # ---- 3. V-guided Greedy (ours) ----
    if algos['v_guided_greedy']['enabled']:
        max_n = algos['v_guided_greedy']['max_nodes']
        arch = model_cfg['architecture']
        cyc_reduce = algos['v_guided_greedy'].get('cyclically_reduce', False)
        cyc_str = " + cyclic reduce" if cyc_reduce else ""
        print(f"\n{'='*60}")
        print(f"  [3] V-GUIDED GREEDY (ours, {arch}{cyc_str}) — {max_n:,} nodes")
        print(f"{'='*60}")
        progress_path = os.path.join(output_dir, 'v_guided_greedy_progress.jsonl')
        skip = count_completed_in_jsonl(progress_path) if resume_dir else 0
        if skip >= len(presentations):
            print(f"  Already complete ({skip} entries). Skipping.")
        else:
            if skip > 0:
                print(f"  Resuming: skipping {skip} already processed")
            algo_pres = presentations[skip:]
            algo_idx = original_indices[skip:]
            with open(progress_path, 'a' if skip > 0 else 'w') as pfh:
                results = run_vguided(
                    algo_pres, model, arch, feat_mean, feat_std, max_n, device,
                    solution_cache=solution_cache, progress_fh=pfh,
                    cyclically_reduce=cyc_reduce, original_indices=algo_idx,
                )
            # remap_indices not needed: run_vguided already sets global indices
            metrics = compute_metrics(results)
            name = f"V-Greedy (ours, {max_n//1000}K)"
            all_metrics[name] = metrics
            all_details['v_guided_greedy'] = results
            print(f"  => Solved: {metrics['solved']}/{metrics['total']} "
                  f"in {fmt_time(metrics['total_time'])}")
            # Save cache to disk after V-guided (typically finds most solutions)
            save_solution_cache(solution_cache, cache_path)
            if solution_cache is not None:
                print(f"  [cache saved: {len(solution_cache)} entries]")
            if cfg['save_incremental']:
                save_results(output_dir, all_metrics, all_details, cfg)
                print(f"  [saved incrementally]")

    # ---- 4. Beam search (ours) ----
    if algos['beam_search']['enabled']:
        max_n = algos['beam_search']['max_nodes']
        arch = model_cfg['architecture']
        beam_widths = algos['beam_search']['beam_widths']
        for bi, k in enumerate(beam_widths):
            print(f"\n{'='*60}")
            print(f"  [4.{bi+1}] BEAM SEARCH k={k} (ours, {arch}) — {max_n:,} nodes")
            print(f"{'='*60}")
            progress_path = os.path.join(output_dir, f'beam_k{k}_progress.jsonl')
            skip = count_completed_in_jsonl(progress_path) if resume_dir else 0
            if skip >= len(presentations):
                print(f"  Already complete ({skip} entries). Skipping.")
            else:
                if skip > 0:
                    print(f"  Resuming: skipping {skip} already processed")
                algo_pres = presentations[skip:]
                algo_idx = original_indices[skip:]
                with open(progress_path, 'a' if skip > 0 else 'w') as pfh:
                    results = run_beam_search(
                        algo_pres, model, arch, feat_mean, feat_std, k, max_n, device,
                        solution_cache=solution_cache, progress_fh=pfh,
                        original_indices=algo_idx,
                    )
                # remap_indices not needed: run_beam_search already sets global indices
                metrics = compute_metrics(results)
                name = f"Beam k={k} (ours, {max_n//1000}K)"
                all_metrics[name] = metrics
                all_details[f'beam_k{k}'] = results
                print(f"  => Solved: {metrics['solved']}/{metrics['total']} "
                      f"in {fmt_time(metrics['total_time'])}")
                save_solution_cache(solution_cache, cache_path)
                if solution_cache is not None:
                    print(f"  [cache saved: {len(solution_cache)} entries]")
                if cfg['save_incremental']:
                    save_results(output_dir, all_metrics, all_details, cfg)
                    print(f"  [saved incrementally]")

    # ---- 5. MCTS (ours) ----
    if algos['mcts']['enabled']:
        max_n = algos['mcts']['max_nodes']
        c_exp = algos['mcts']['c_explore']
        arch = model_cfg['architecture']
        print(f"\n{'='*60}")
        print(f"  [5] MCTS c={c_exp} (ours, {arch}) — {max_n:,} nodes")
        print(f"{'='*60}")
        progress_path = os.path.join(output_dir, 'mcts_progress.jsonl')
        skip = count_completed_in_jsonl(progress_path) if resume_dir else 0
        if skip >= len(presentations):
            print(f"  Already complete ({skip} entries). Skipping.")
        else:
            if skip > 0:
                print(f"  Resuming: skipping {skip} already processed")
            algo_pres = presentations[skip:]
            algo_idx = original_indices[skip:]
            with open(progress_path, 'a' if skip > 0 else 'w') as pfh:
                results = run_mcts_search(
                    algo_pres, model, arch, feat_mean, feat_std, max_n, c_exp, device,
                    solution_cache=solution_cache, progress_fh=pfh,
                    original_indices=algo_idx,
                )
            remap_indices(results, algo_idx)
            metrics = compute_metrics(results)
            name = f"MCTS c={c_exp} (ours, {max_n//1000}K)"
            all_metrics[name] = metrics
            all_details['mcts'] = results
            print(f"  => Solved: {metrics['solved']}/{metrics['total']} "
                  f"in {fmt_time(metrics['total_time'])}")
            save_solution_cache(solution_cache, cache_path)
            if solution_cache is not None:
                print(f"  [cache saved: {len(solution_cache)} entries]")
            if cfg['save_incremental']:
                save_results(output_dir, all_metrics, all_details, cfg)
                print(f"  [saved incrementally]")

    # ---- Final output ----
    total_time = time.time() - experiment_start
    print(f"\n\nAll experiments completed in {fmt_time(total_time)}.")
    if solution_cache is not None:
        print(f"  Solution cache: {len(solution_cache)} states memoized")

    # Save cache one final time
    save_solution_cache(solution_cache, cache_path)

    # Comparison table
    print_comparison_table(all_metrics, greedy_solved_set, presentations)

    # Newly solved details
    print_newly_solved(all_metrics, greedy_solved_set, presentations)

    # Final save
    summary_path = save_results(output_dir, all_metrics, all_details, cfg)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  results.json        — summary metrics")
    print(f"  *_details.json      — per-presentation results")
    print(f"  *_progress.jsonl    — per-presentation streaming progress")
    print(f"  config_used.yaml    — config snapshot")
    if solution_cache is not None:
        print(f"  Solution cache:       {cache_path} ({len(solution_cache)} entries)")


if __name__ == '__main__':
    main()
