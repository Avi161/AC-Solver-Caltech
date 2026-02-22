"""
Replay and verify solution paths from experiment results.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ac_solver.envs.ac_moves import ACMove
from value_search.feature_extraction import compute_features
from value_search.benchmark import load_all_presentations

"""
Notation Key
Symbol    Meaning
a, b      Generators (x_0 = a, x_1 = b in source code)
A, B      Inverses (a⁻¹, b⁻¹)
ε         Empty word
⟨a,b | R1, R2⟩  Presentation (r_0 = R1, r_1 = R2 in source code)

"""
# ---------------------------------------------------------------------------
# AC Move names - CORRECTED from ac_solver/envs/ac_moves.py source
# ---------------------------------------------------------------------------
AC_MOVE_NAMES = {
    0:  "R2 ← R2 · R1",           # r_1 --> r_1 r_0
    1:  "R1 ← R1 · R2⁻¹",         # r_0 --> r_0 r_1^{-1}
    2:  "R2 ← R2 · R1⁻¹",         # r_1 --> r_1 r_0^{-1}
    3:  "R1 ← R1 · R2",           # r_0 --> r_0 r_1
    4:  "R2 ← A·R2·a (conj a⁻¹)", # r_1 --> x_0^{-1} r_1 x_0
    5:  "R1 ← B·R1·b (conj b⁻¹)", # r_0 --> x_1^{-1} r_0 x_1
    6:  "R2 ← B·R2·b (conj b⁻¹)", # r_1 --> x_1^{-1} r_1 x_1
    7:  "R1 ← a·R1·A (conj a)",   # r_0 --> x_0 r_0 x_0^{-1}
    8:  "R2 ← a·R2·A (conj a)",   # r_1 --> x_0 r_1 x_0^{-1}
    9:  "R1 ← b·R1·B (conj b)",   # r_0 --> x_1 r_0 x_1^{-1}
    10: "R2 ← b·R2·B (conj b)",   # r_1 --> x_1 r_1 x_1^{-1}
    11: "R1 ← A·R1·a (conj a⁻¹)", # r_0 --> x_0^{-1} r_0 x_0
}


def state_to_algebra(state, max_relator_length):
    """
    Convert a presentation state array to algebraic notation.
    
    Encoding: 1=a, 2=b, -1=A (a⁻¹), -2=B (b⁻¹), 0=padding
    
    Returns:
        tuple: (r1_str, r2_str) in algebraic notation
    """
    def word_to_str(word):
        """Convert a word array to string."""
        symbols = {1: 'a', 2: 'b', -1: 'A', -2: 'B'}
        chars = []
        for val in word:
            if val == 0:
                break
            chars.append(symbols.get(int(val), '?'))
        return ''.join(chars) if chars else 'ε'  # ε for empty/trivial
    
    r1 = state[:max_relator_length]
    r2 = state[max_relator_length:]
    
    return word_to_str(r1), word_to_str(r2)


def format_presentation(r1_str, r2_str):
    """Format a presentation as ⟨a,b | R1, R2⟩."""
    return f"⟨a,b | {r1_str}, {r2_str}⟩"


def replay_path(presentation, path, source_idx=0):
    """
    Replay a solution path, collecting all intermediate states.

    The stored path format is [(sentinel_action, init_length), (action1, len1), ...].
    The first entry is a sentinel whose length matches the initial presentation.
    Actual AC moves start from index 1. Stored actions are 1-indexed.
    """
    state = np.array(presentation, dtype=np.int8)
    max_relator_length = len(state) // 2
    len_r1 = int(np.count_nonzero(state[:max_relator_length]))
    len_r2 = int(np.count_nonzero(state[max_relator_length:]))
    word_lengths = [len_r1, len_r2]

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

    # Replay each action
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
            'action_applied': action_id,
        })

    assert sum(word_lengths) == 2, (
        f"Path replay for source_idx={source_idx} did not reach trivial state. "
        f"Final length: {sum(word_lengths)}"
    )

    return examples


def convert_greedy_path(raw_path, initial_length):
    """Convert path from experiment results format to replay_path format."""
    converted = [(0, initial_length)]  # Sentinel
    for action_0idx, length in raw_path:
        converted.append((action_0idx + 1, length))  # Convert to 1-indexed
    return converted


def verify_single_path(presentation, raw_path, idx, verbose=True, show_algebra=True):
    """
    Verify a single solution path from experiment results.
    
    Parameters:
        presentation: initial presentation array
        raw_path: path from greedy_details.json format [[action, len], ...]
        idx: presentation index
        verbose: print step-by-step details
        show_algebra: show algebraic notation for each state
    
    Returns:
        bool: True if path is valid and reaches trivial state
    """
    pres_array = np.array(presentation, dtype=np.int8)
    max_relator_length = len(pres_array) // 2
    
    # Calculate initial length
    len_r1 = int(np.count_nonzero(pres_array[:max_relator_length]))
    len_r2 = int(np.count_nonzero(pres_array[max_relator_length:]))
    initial_length = len_r1 + len_r2
    
    # Convert path format
    converted_path = convert_greedy_path(raw_path, initial_length)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Verifying presentation idx={idx}")
        print(f"{'='*70}")
        
        # Show initial state in algebraic form
        r1_str, r2_str = state_to_algebra(pres_array, max_relator_length)
        print(f"\nInitial presentation:")
        print(f"  Algebraic: {format_presentation(r1_str, r2_str)}")
        print(f"  R1 = {r1_str} (length {len_r1})")
        print(f"  R2 = {r2_str} (length {len_r2})")
        print(f"  Total length: {initial_length}")
        print(f"  Path length: {len(raw_path)} moves")
        print(f"\n{'-'*70}")
        print(f"Step-by-step transformation:")
        print(f"{'-'*70}")
    
    try:
        examples = replay_path(pres_array, converted_path, source_idx=idx)
        
        if verbose:
            for i, ex in enumerate(examples):
                state = ex['state']
                r1_str, r2_str = state_to_algebra(state, max_relator_length)
                
                if i == 0:
                    print(f"\n[Step 0] Initial state")
                else:
                    action_id = converted_path[i][0] - 1  # Convert back to 0-indexed
                    action_name = AC_MOVE_NAMES.get(action_id, f"Action {action_id}")
                    print(f"\n[Step {i}] Apply action {action_id}: {action_name}")
                
                print(f"  {format_presentation(r1_str, r2_str)}")
                print(f"  R1 = {r1_str:20s}  R2 = {r2_str:20s}  (total: {ex['total_length']})")

        final_length = examples[-1]['total_length']
        
        if verbose:
            print(f"\n{'-'*70}")
        
        if final_length == 2:
            if verbose:
                print(f"✓ SUCCESS: Reached trivial presentation (length=2)")
                final_r1, final_r2 = state_to_algebra(examples[-1]['state'], max_relator_length)
                print(f"  Final: {format_presentation(final_r1, final_r2)}")
            return True
        else:
            if verbose:
                print(f"✗ FAILED: Final length={final_length}, expected 2")
            return False
            
    except AssertionError as e:
        if verbose:
            print(f"\n✗ ASSERTION ERROR: {e}")
        return False
    except Exception as e:
        if verbose:
            print(f"\n✗ ERROR: {e}")
        return False


def verify_results_file(results_dir, algorithm='greedy', indices=None, verbose=True):
    """Verify solution paths from an experiment results directory."""
    presentations = load_all_presentations()
    
    details_file = os.path.join(results_dir, f"{algorithm}_details.json")
    if not os.path.exists(details_file):
        print(f"ERROR: Details file not found: {details_file}")
        print(f"Available files in {results_dir}:")
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                print(f"  - {f}")
        return None
    
    with open(details_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'#'*70}")
    print(f"Verifying {algorithm} results from:")
    print(f"  {details_file}")
    print(f"{'#'*70}")
    
    solved_results = [r for r in results if r.get('solved') and r.get('path')]
    print(f"\nTotal results: {len(results)}")
    print(f"Solved with paths: {len(solved_results)}")
    
    if indices is not None:
        solved_results = [r for r in solved_results if r['idx'] in indices]
        print(f"Filtered to indices {indices}: {len(solved_results)}")
    
    stats = {
        'total': len(solved_results),
        'verified': 0,
        'failed': 0,
        'failed_indices': []
    }
    
    for result in solved_results:
        idx = result['idx']
        raw_path = result['path']
        presentation = presentations[idx]
        
        success = verify_single_path(presentation, raw_path, idx, verbose=verbose)
        
        if success:
            stats['verified'] += 1
        else:
            stats['failed'] += 1
            stats['failed_indices'].append(idx)
    
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total verified: {stats['verified']}/{stats['total']}")
    print(f"Failed: {stats['failed']}")
    if stats['failed_indices']:
        print(f"Failed indices: {stats['failed_indices']}")
    print(f"{'='*70}")
    
    return stats


def list_available_results():
    """List all available result directories."""
    results_base = os.path.join(PROJECT_ROOT, 'experiments', 'results')
    if not os.path.exists(results_base):
        print(f"No results directory found at {results_base}")
        return []
    
    dirs = sorted(os.listdir(results_base))
    print(f"\nAvailable result directories in {results_base}:")
    for d in dirs:
        full_path = os.path.join(results_base, d)
        if os.path.isdir(full_path):
            files = os.listdir(full_path)
            detail_files = [f for f in files if f.endswith('_details.json')]
            print(f"  {d}/")
            for df in detail_files:
                print(f"    - {df}")
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Verify solution paths from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available results
  python value_search/replay_path.py --list
  
  # Verify index 12 from specific folder (with algebraic notation)
  python value_search/replay_path.py -r experiments/results/2026-02-17_22-01-04 -i 12
  
  # Verify all from latest results (summary only)
  python value_search/replay_path.py --latest -q
  
  # Verify specific indices
  python value_search/replay_path.py --latest -i 8 9 12

Notation:
  a, b     = generators
  A, B     = inverses (a⁻¹, b⁻¹)
  ε        = empty word (trivial)
  ⟨a,b | R1, R2⟩ = presentation with relators R1 and R2

AC Moves (from source code):
  0:  R2 ← R2 · R1           (concatenate)
  1:  R1 ← R1 · R2⁻¹         (concatenate)
  2:  R2 ← R2 · R1⁻¹         (concatenate)
  3:  R1 ← R1 · R2           (concatenate)
  4:  R2 ← A·R2·a            (conjugate by a⁻¹)
  5:  R1 ← B·R1·b            (conjugate by b⁻¹)
  6:  R2 ← B·R2·b            (conjugate by b⁻¹)
  7:  R1 ← a·R1·A            (conjugate by a)
  8:  R2 ← a·R2·A            (conjugate by a)
  9:  R1 ← b·R1·B            (conjugate by b)
  10: R2 ← b·R2·B            (conjugate by b)
  11: R1 ← A·R1·a            (conjugate by a⁻¹)
        """
    )
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        help='Path to results directory'
    )
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        default='greedy',
        choices=['greedy', 'bfs', 'v_guided_greedy', 'beam_k10', 'beam_k50', 'beam_k100', 'mcts', 'ppo', 'ppo_rnd'],
        help='Which algorithm results to verify (default: greedy)'
    )
    parser.add_argument(
        '--indices', '-i',
        type=int,
        nargs='+',
        help='Specific presentation indices to verify'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show summary'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available result directories'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use the latest results directory'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_results()
        return
    
    results_dir = args.results_dir
    if args.latest or results_dir is None:
        results_base = os.path.join(PROJECT_ROOT, 'experiments', 'results')
        if os.path.exists(results_base):
            dirs = sorted([d for d in os.listdir(results_base) 
                          if os.path.isdir(os.path.join(results_base, d))])
            if dirs:
                results_dir = os.path.join(results_base, dirs[-1])
                print(f"Using latest results: {results_dir}")
            else:
                print("No result directories found")
                return
        else:
            print(f"Results directory not found: {results_base}")
            return
    
    verify_results_file(
        results_dir=results_dir,
        algorithm=args.algorithm,
        indices=args.indices,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()