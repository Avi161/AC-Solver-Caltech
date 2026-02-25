"""
Worker function for Colab multiprocessing with spawn start method.

This must live in a .py file (not a notebook cell) because the 'spawn'
start method creates fresh Python processes that need to import the
worker function by module path. Notebook-defined functions can't be
imported this way.
"""

import time
import numpy as np

from value_search.value_guided_search import (
    value_guided_greedy_search, beam_search, load_model,
)
from value_search.mcts import mcts_search


def search_worker(args):
    """Search a single presentation. Runs in a subprocess."""
    idx, pres_list, algorithm, config = args
    pres = np.array(pres_list, dtype=np.int8)

    # Each worker loads model independently
    device = config['device']
    model, feat_mean, feat_std = load_model(
        config['checkpoint'], config['architecture'],
        config['feature_stats'], device
    )

    t0 = time.time()
    if algorithm == 'v_guided':
        solved, path, stats = value_guided_greedy_search(
            pres, model=model, architecture=config['architecture'],
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=config['max_nodes'], device=device,
            cyclically_reduce_after_moves=config.get('cyclically_reduce', False),
        )
    elif algorithm == 'beam':
        solved, path, stats = beam_search(
            pres, model=model, architecture=config['architecture'],
            feat_mean=feat_mean, feat_std=feat_std,
            beam_width=config.get('beam_width', 50),
            max_nodes_to_explore=config['max_nodes'], device=device,
        )
    elif algorithm == 'mcts':
        solved, path, stats = mcts_search(
            pres, model=model, architecture=config['architecture'],
            feat_mean=feat_mean, feat_std=feat_std,
            max_nodes_to_explore=config['max_nodes'],
            c_explore=config.get('c_explore', 1.41), device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed = time.time() - t0
    result = {
        'idx': idx, 'solved': solved,
        'path_length': len(path) if solved else 0,
        'nodes_explored': stats.get('nodes_explored', 0),
        'time': elapsed,
    }
    if solved and path:
        result['path'] = [[int(a), int(l)] for a, l in path]
    return result
