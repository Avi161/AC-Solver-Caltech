#!/usr/bin/env python3
"""
Convert experiment results JSON to the JSONL format expected by the
Miller-Schupp Path Viewer website (https://avi161.github.io/miller-schupp-path-viewer/).

The input is a *_details.json file (JSON array) produced by run_experiments.py.
The output is a JSONL file where each line is:

  {"idx": N, "solved": true, "path_length": L, "path": [[action, len], ...]}

For solved presentations the script replays each path step-by-step.  When the
original search used cyclic reduction (e.g. PPO/RL environment), it inserts
explicit "cyclic reduction" steps (action_id = -1) so the website can display
every algebraic sub-step of the trivialization.  When the search did NOT use
cyclic reduction (greedy, V-guided, beam, MCTS), the path is emitted as-is
since all lengths already reflect the non-cyclically-reduced states.

Detection is automatic: the script replays without cyclic reduction first; if
the lengths match the stored path, no CR steps are needed.  If they don't, it
replays with the split approach (AC move then CR) and inserts CR steps where
the length changes.

Action IDs in the output:
    0-11  : AC moves h1-h12 (as displayed on the website)
    -1    : cyclic reduction applied to the presentation

Usage:
    python scripts/convert_to_website_format.py INPUT_JSON [-o OUTPUT_JSONL]

Examples:
    # Convert v-guided greedy results (no cyclic reduction in search)
    python scripts/convert_to_website_format.py \\
        experiments/results/test_verification/v_guided_greedy_details.json

    # Convert PPO results (cyclic reduction was used during training)
    python scripts/convert_to_website_format.py \\
        experiments/results/2026-02-22_18-00-57_ppo_rnd/ppo_rnd_details.json \\
        -o ppo_results_website.jsonl
"""

import os
import sys
import json
import argparse
import numpy as np
from ast import literal_eval

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ac_solver.envs.ac_moves import ACMove
from ac_solver.envs.utils import simplify_presentation, is_presentation_trivial


# ---------------------------------------------------------------------------
# Load all 1190 Miller-Schupp presentations
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(
    PROJECT_ROOT, "ac_solver", "search", "miller_schupp", "data"
)


def load_all_presentations(data_dir=None):
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


# ---------------------------------------------------------------------------
# Detect whether a path was produced with cyclic reduction
# ---------------------------------------------------------------------------

def _path_uses_cyclic_reduction(presentation, raw_path):
    """
    Try replaying the path WITHOUT cyclic reduction.  If every step's
    resulting length matches the stored expected length, the path was
    produced without CR.  Otherwise it must have used CR.
    """
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    wl = [
        int(np.count_nonzero(state[:mrl])),
        int(np.count_nonzero(state[mrl:])),
    ]

    for action_id, expected_length in raw_path:
        state, wl = ACMove(action_id, state, mrl, wl, cyclical=False)
        if sum(wl) != expected_length:
            return True  # mismatch → CR was used
    return False


# ---------------------------------------------------------------------------
# Replay a CR-path and insert explicit cyclic-reduction steps
# ---------------------------------------------------------------------------

def _replay_with_cr_steps(presentation, raw_path):
    """
    For paths produced WITH cyclic reduction, replay each move in two phases:
      1. Apply the AC move WITHOUT CR  → record [action_id, intermediate_len]
      2. Apply CR                      → if length changed, record [-1, final_len]
    Then continue from the post-CR state (which is what the search saw).
    """
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    wl = [
        int(np.count_nonzero(state[:mrl])),
        int(np.count_nonzero(state[mrl:])),
    ]

    expanded = []

    for action_id, expected_length in raw_path:
        # Phase 1: AC move without cyclic reduction
        mid_state, mid_wl = ACMove(action_id, state, mrl, wl, cyclical=False)
        mid_len = sum(mid_wl)
        expanded.append([int(action_id), int(mid_len)])

        # Phase 2: cyclic reduction
        cr_state, cr_wl = simplify_presentation(
            mid_state.copy(), mrl, list(mid_wl), cyclical=True,
        )
        cr_len = sum(cr_wl)

        if cr_len < mid_len:
            expanded.append([-1, int(cr_len)])

        # Advance from the fully-reduced state
        state = cr_state
        wl = list(cr_wl)

    return expanded


# ---------------------------------------------------------------------------
# Simple format conversion (no CR insertion needed)
# ---------------------------------------------------------------------------

def _convert_simple(raw_path):
    """Just re-emit the path as a list of [action_id, length] pairs."""
    return [[int(a), int(l)] for a, l in raw_path]


# ---------------------------------------------------------------------------
# Process one solved entry
# ---------------------------------------------------------------------------

def process_solved(presentation, raw_path):
    """
    Returns (expanded_path, used_cr) where expanded_path is the website-ready
    path and used_cr indicates whether CR steps were inserted.
    """
    if _path_uses_cyclic_reduction(presentation, raw_path):
        return _replay_with_cr_steps(presentation, raw_path), True
    else:
        return _convert_simple(raw_path), False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def _verify_path(presentation, expanded_path):
    """
    Replay the expanded path and check that it reaches a trivial state.
    Returns (ok, final_length).
    """
    state = np.array(presentation, dtype=np.int8)
    mrl = len(state) // 2
    wl = [
        int(np.count_nonzero(state[:mrl])),
        int(np.count_nonzero(state[mrl:])),
    ]

    for action_id, _length in expanded_path:
        if action_id == -1:
            # Cyclic reduction step
            state, wl = simplify_presentation(state.copy(), mrl, list(wl), cyclical=True)
        else:
            state, wl = ACMove(action_id, state, mrl, wl, cyclical=False)

    final_len = sum(wl)
    return final_len == 2 and is_presentation_trivial(state), final_len


def convert_file(input_path, output_path, presentations):
    """Convert a details JSON file to website JSONL format."""
    with open(input_path, "r") as f:
        results = json.load(f)

    total = len(results)
    solved_count = 0
    cr_count = 0      # paths where cyclic-reduction steps were inserted
    verified = 0
    warnings = []
    errors = []

    with open(output_path, "w") as out:
        for result in results:
            idx = result["idx"]
            solved = result.get("solved", False)

            if not solved or "path" not in result:
                out.write(json.dumps({"idx": idx, "solved": False}) + "\n")
                continue

            solved_count += 1
            raw_path = result["path"]
            pres = presentations[idx]

            try:
                expanded_path, used_cr = process_solved(pres, raw_path)
            except Exception as e:
                errors.append((idx, str(e)))
                out.write(json.dumps({"idx": idx, "solved": False}) + "\n")
                continue

            if used_cr:
                cr_count += 1

            # Verify the expanded path actually reaches a trivial state
            ok, final_len = _verify_path(pres, expanded_path)
            if ok:
                verified += 1
            else:
                warnings.append((idx, final_len))

            record = {
                "idx": idx,
                "solved": True,
                "path_length": len(expanded_path),
                "path": expanded_path,
            }
            out.write(json.dumps(record) + "\n")

    print(f"Converted {total} entries ({solved_count} solved)")
    print(f"  Paths with CR steps inserted: {cr_count}")
    print(f"  Paths without CR (direct):    {solved_count - cr_count}")
    print(f"  Verified trivial:             {verified}/{solved_count}")
    if warnings:
        print(f"  WARNING: {len(warnings)} path(s) do not reach trivial state:")
        for idx, flen in warnings[:10]:
            print(f"    idx {idx}: final length = {flen}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for idx, msg in errors[:5]:
            print(f"    idx {idx}: {msg}")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert experiment results to website JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Action IDs in output:
  0-11  AC moves (displayed as h1-h12 on the website)
  -1    Cyclic reduction step

Examples:
  python scripts/convert_to_website_format.py \\
      experiments/results/test_verification/v_guided_greedy_details.json

  python scripts/convert_to_website_format.py \\
      experiments/results/2026-02-22_18-00-57_ppo_rnd/ppo_rnd_details.json \\
      -o ppo_website.jsonl
""",
    )
    parser.add_argument("input", help="Path to *_details.json file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL path (default: <input_stem>_website.jsonl)",
    )
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(args.input)[0]
        output_path = stem + "_website.jsonl"

    print("Loading all 1190 presentations...")
    presentations = load_all_presentations()
    print(f"Loaded {len(presentations)} presentations\n")

    convert_file(args.input, output_path, presentations)


if __name__ == "__main__":
    main()
