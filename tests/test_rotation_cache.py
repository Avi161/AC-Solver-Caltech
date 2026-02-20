#!/usr/bin/env python3
"""
Test that _check_cache_with_rotations and _expand_cache_with_rotations
produce paths that, when replayed with ACMove, actually reach length 2.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ac_solver.envs.ac_moves import ACMove
from value_search.value_guided_search import (
    _check_cache_with_rotations,
    _expand_cache_with_rotations,
    backfill_solution_cache,
)
from value_search.benchmark import load_all_presentations


def replay_path(presentation, path, cyclical=False):
    """Replay a path of (action, total_length) on a presentation.
    Returns the final state and its total length."""
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    len_r1 = int(np.count_nonzero(state[:mrl]))
    len_r2 = int(np.count_nonzero(state[mrl:]))
    wl = [len_r1, len_r2]

    for step_i, (action, _expected_len) in enumerate(path):
        state, wl = ACMove(action, state, mrl, wl, cyclical=cyclical)

    return state, sum(wl)


def test_expand_and_check_roundtrip():
    """Test: expand cache with rotations, then check_cache finds them and
    returned paths replay to length 2."""
    presentations = load_all_presentations()

    # Use a simple greedy to solve a few easy ones for testing
    from ac_solver.search.greedy import greedy_search

    cache = {}
    solved_presentations = []

    print("Solving some presentations with greedy to build cache...")
    for i in range(min(50, len(presentations))):
        pres = presentations[i]
        solved, path = greedy_search(pres, max_nodes_to_explore=10000)
        if solved and path:
            # path has sentinel at index 0: (-1, initial_length)
            clean_path = [(int(a), int(l)) for a, l in path[1:]]
            backfill_solution_cache(cache, pres, clean_path, expand_rotations=True)
            solved_presentations.append((i, pres, clean_path))

    print(f"  Solved {len(solved_presentations)} presentations, "
          f"cache has {len(cache)} entries")

    if not solved_presentations:
        print("SKIP: no presentations solved")
        return

    # Test 1: Verify that direct cache entries replay correctly
    print("\n--- Test 1: Direct cache entries ---")
    direct_failures = 0
    direct_tested = 0
    for idx, pres, original_path in solved_presentations:
        state_tup = tuple(pres)
        if state_tup in cache:
            cached_path = list(cache[state_tup])
            _, final_len = replay_path(pres, cached_path)
            direct_tested += 1
            if final_len != 2:
                direct_failures += 1
                if direct_failures <= 3:
                    print(f"  FAIL: pres[{idx}] direct cache replay -> length {final_len} != 2")
    print(f"  Tested {direct_tested} direct entries, "
          f"failures: {direct_failures}")

    # Test 2: Verify that _check_cache_with_rotations returns valid paths
    # Create rotated versions of solved presentations and check them
    print("\n--- Test 2: Rotation cache lookups via _check_cache_with_rotations ---")
    rotation_tested = 0
    rotation_failures = 0
    for idx, pres, original_path in solved_presentations[:20]:
        state = np.array(pres, dtype=np.int8)
        mrl = len(state) // 2

        for rel_idx in range(2):
            offset = rel_idx * mrl
            relator = state[offset:offset + mrl]
            nz = relator[relator != 0]
            L = len(nz)
            if L <= 1:
                continue

            # Try each rotation
            for k in range(1, L):
                rotated_state = state.copy()
                rotated_state[offset:offset + L] = np.concatenate([nz[k:], nz[:k]])
                rot_tup = tuple(rotated_state)

                # Use _check_cache_with_rotations on the ORIGINAL state
                # It should find the rotated version in cache and return a valid path
                # But we actually want to test the reverse: check from a rotated state
                # that the original is found.
                # Let's test both directions.

                # Direction A: check if rotated state can find a path via cache
                path_from_rotated = _check_cache_with_rotations(rot_tup, mrl, cache)
                if path_from_rotated is not None:
                    rotation_tested += 1
                    _, final_len = replay_path(rotated_state, path_from_rotated)
                    if final_len != 2:
                        rotation_failures += 1
                        if rotation_failures <= 5:
                            print(f"  FAIL: pres[{idx}] rel={rel_idx} rot={k} "
                                  f"replay from rotated -> length {final_len} != 2")
                            print(f"    rotated state: {rot_tup[:10]}...")
                            print(f"    path length: {len(path_from_rotated)}")

    print(f"  Tested {rotation_tested} rotation lookups, "
          f"failures: {rotation_failures}")

    # Test 3: Build a cache with ONLY the original states (no rotation expansion),
    # then test _check_cache_with_rotations from a rotated state
    print("\n--- Test 3: _check_cache_with_rotations finding rotated states in non-expanded cache ---")
    small_cache = {}
    for idx, pres, original_path in solved_presentations:
        # Only store the direct state, no rotation expansion
        state_tup = tuple(pres)
        if state_tup not in small_cache:
            small_cache[state_tup] = original_path

    check_tested = 0
    check_failures = 0
    for idx, pres, original_path in solved_presentations[:20]:
        state = np.array(pres, dtype=np.int8)
        mrl = len(state) // 2

        for rel_idx in range(2):
            offset = rel_idx * mrl
            relator = state[offset:offset + mrl]
            nz = relator[relator != 0]
            L = len(nz)
            if L <= 1:
                continue
            if nz[-1] + nz[0] == 0:
                continue

            for k in range(1, min(L, 4)):  # test first few rotations
                rotated_state = state.copy()
                rotated_state[offset:offset + L] = np.concatenate([nz[k:], nz[:k]])
                rot_tup = tuple(rotated_state)

                # The original state is in small_cache.
                # _check_cache_with_rotations on the rotated state should find it
                # by rotating back to the original.
                path_from_rotated = _check_cache_with_rotations(rot_tup, mrl, small_cache)
                if path_from_rotated is not None:
                    check_tested += 1
                    _, final_len = replay_path(rotated_state, path_from_rotated)
                    if final_len != 2:
                        check_failures += 1
                        if check_failures <= 5:
                            print(f"  FAIL: pres[{idx}] rel={rel_idx} rot={k} "
                                  f"check_cache from rotated -> length {final_len} != 2")
                            # Debug: replay step by step
                            s = np.array(rot_tup, dtype=np.int8)
                            lr1 = int(np.count_nonzero(s[:mrl]))
                            lr2 = int(np.count_nonzero(s[mrl:]))
                            wl = [lr1, lr2]
                            print(f"    start length: {sum(wl)}")
                            for si, (a, el) in enumerate(path_from_rotated[:10]):
                                s, wl = ACMove(a, s, mrl, wl, cyclical=False)
                                print(f"    step {si}: action={a}, length={sum(wl)}")

    print(f"  Tested {check_tested} rotation checks, "
          f"failures: {check_failures}")

    # Summary
    print(f"\n{'='*60}")
    total_failures = direct_failures + rotation_failures + check_failures
    if total_failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"TOTAL FAILURES: {total_failures}")
    print(f"{'='*60}")
    return total_failures


if __name__ == '__main__':
    failures = test_expand_and_check_roundtrip()
    sys.exit(1 if failures else 0)
