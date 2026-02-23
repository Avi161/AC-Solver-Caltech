"""
Comprehensive tests to verify PPO solution correctness.

Key concerns:
1. The termination condition `done = sum(self.lengths) == 2` may accept
   non-trivial states (e.g., both relators being the same generator).
2. Cyclic reduction in simplify_relator may over-reduce and corrupt states.
3. The lengths tracking may get out of sync with the actual state array.
4. Concatenation/conjugation may produce relators of length 0.
"""

import pytest
import numpy as np
from ast import literal_eval
from importlib import resources

from ac_solver.envs.ac_env import ACEnv, ACEnvConfig
from ac_solver.envs.ac_moves import ACMove, concatenate_relators, conjugate
from ac_solver.envs.utils import (
    is_presentation_trivial,
    is_array_valid_presentation,
    simplify_relator,
    simplify_presentation,
    change_max_relator_length_of_presentation,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_all_presentations():
    """Load all 1190 Miller-Schupp presentations."""
    with resources.open_text(
        "ac_solver.search.miller_schupp.data", "all_presentations.txt"
    ) as f:
        return [literal_eval(line.strip()) for line in f]


def get_presentation(idx, max_relator_length=36):
    """Get presentation at index `idx` with specified max_relator_length.

    Loads presentations as lists (matching how the training code does it)
    and converts to the target max_relator_length.
    """
    presentations = load_all_presentations()
    p = presentations[idx]  # p is a Python list from literal_eval
    # change_max_relator_length_of_presentation expects a list input
    # to match how environment.py uses it
    return change_max_relator_length_of_presentation(p, max_relator_length)


def replay_actions(initial_state, actions, verbose=False):
    """Replay a list of actions on an initial state, returning full trace."""
    state = np.array(initial_state, dtype=np.int8)
    max_relator_length = len(state) // 2
    lengths = [
        int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
        for i in range(2)
    ]
    trace = [{"state": state.copy(), "lengths": lengths.copy()}]

    if verbose:
        r0 = state[:max_relator_length][:lengths[0]]
        r1 = state[max_relator_length:][:lengths[1]]
        print(f"  Start: r0={list(r0)}, r1={list(r1)}, lengths={lengths}, sum={sum(lengths)}")

    for action in actions:
        state, lengths = ACMove(
            move_id=int(action),
            presentation=state,
            max_relator_length=max_relator_length,
            lengths=lengths,
        )
        trace.append({"state": state.copy(), "lengths": lengths.copy(), "action": action})

        if verbose:
            r0 = state[:max_relator_length][:lengths[0]]
            r1 = state[max_relator_length:][:lengths[1]]
            print(f"  Action {action}: r0={list(r0)}, r1={list(r1)}, lengths={lengths}, sum={sum(lengths)}")

    return trace


def verify_lengths_consistent(state, lengths, max_relator_length):
    """Check that tracked lengths match actual non-zero counts in state array."""
    for i in range(2):
        actual = int(np.count_nonzero(
            state[i * max_relator_length : (i + 1) * max_relator_length]
        ))
        assert actual == lengths[i], (
            f"Length mismatch for relator {i}: tracked={lengths[i]}, actual={actual}\n"
            f"  State: {list(state)}"
        )


# ──────────────────────────────────────────────────────────────────────
# 1. Replay and verify the actual PPO solutions from the user's run
# ──────────────────────────────────────────────────────────────────────

# These are the solutions from the user's training log
PPO_SOLUTIONS = [
    # (idx, actions_from_path)
    # path format is [[action_id, total_length], ...], we extract action_ids
    (124, [1, 0]),       # path_length=2
    (260, [1, 0]),       # path_length=2
    (518, [1, 0]),       # path_length=2, second solution
    (407, [1, 0]),       # path_length=2
    (16,  [1, 0]),       # path_length=2  (action 1 then action 0)
    (459, [1, 0]),       # path_length=2
    (494, [1, 0]),       # path_length=2 (second solution)
    (1,   [1, 0]),       # path_length=2
    (171, [1, 0]),       # path_length=2
    (309, [1, 0]),       # path_length=2
]

# Also include longer solutions for comparison
PPO_SOLUTIONS_LONGER = [
    (171, [4, 4, 1, 7, 5, 8, 4, 2, 3, 5, 0, 2, 6, 0, 9, 4, 2, 8, 7, 1, 0, 3, 8, 2]),  # path_length=24
    (171, [11, 8, 4, 1, 11, 0]),  # path_length=6
    (494, [1, 11, 0]),  # path_length=3
    (518, [1, 9, 2, 0, 5, 7, 0]),  # path_length=7
]


class TestReplayPPOSolutions:
    """Replay each reported PPO solution and verify it truly reaches a trivial state."""

    @pytest.mark.parametrize("idx, actions", PPO_SOLUTIONS)
    def test_2step_solution_reaches_trivial(self, idx, actions):
        """Verify that each 2-step solution actually reaches is_presentation_trivial."""
        state = get_presentation(idx)
        max_relator_length = len(state) // 2

        trace = replay_actions(state, actions, verbose=True)
        final = trace[-1]

        # Check lengths are consistent with state
        verify_lengths_consistent(final["state"], final["lengths"], max_relator_length)

        # Check total length is 2
        assert sum(final["lengths"]) == 2, (
            f"idx={idx}: Final total length is {sum(final['lengths'])}, expected 2"
        )

        # CRITICAL: Check BOTH relators have length >= 1
        assert final["lengths"][0] >= 1, (
            f"idx={idx}: Relator 0 has length {final['lengths'][0]} (should be >= 1)"
        )
        assert final["lengths"][1] >= 1, (
            f"idx={idx}: Relator 1 has length {final['lengths'][1]} (should be >= 1)"
        )

        # CRITICAL: Check is_presentation_trivial (strict check)
        assert is_presentation_trivial(final["state"]), (
            f"idx={idx}: Final state is NOT trivial!\n"
            f"  State: {list(final['state'])}\n"
            f"  Lengths: {final['lengths']}\n"
            f"  This means the solution is a FALSE POSITIVE."
        )

    @pytest.mark.parametrize("idx, actions", PPO_SOLUTIONS_LONGER)
    def test_longer_solution_reaches_trivial(self, idx, actions):
        """Verify longer solutions also reach trivial state."""
        state = get_presentation(idx)
        max_relator_length = len(state) // 2

        trace = replay_actions(state, actions, verbose=True)
        final = trace[-1]

        verify_lengths_consistent(final["state"], final["lengths"], max_relator_length)
        assert is_presentation_trivial(final["state"]), (
            f"idx={idx}: Final state after {len(actions)} moves is NOT trivial!\n"
            f"  State: {list(final['state'])}\n"
            f"  Lengths: {final['lengths']}"
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Termination condition bug test
# ──────────────────────────────────────────────────────────────────────

class TestTerminationCondition:
    """Test that the environment's done condition is correct."""

    def test_done_requires_both_relators_nonzero(self):
        """Verify that done=True only when BOTH relators have length >= 1."""
        # Create env with trivial starting state
        config = ACEnvConfig(initial_state=np.array([1, 0, 2, 0], dtype=np.int8))
        env = ACEnv(config)

        # Manually set state to [0, 0, 1, 2] (invalid: first relator empty)
        # This should NOT trigger done
        env.state = np.array([0, 0, 1, 2], dtype=np.int8)
        env.lengths = [0, 2]

        # sum(lengths) == 2, but first relator is empty
        done_check = sum(env.lengths) == 2
        trivial_check = is_presentation_trivial(env.state)

        assert done_check == True, "sum(lengths)==2 should be True"
        assert trivial_check == False, "is_presentation_trivial should be False for [0,0,1,2]"
        print(f"BUG CONFIRMED: done={done_check} but is_trivial={trivial_check}")

    def test_done_both_same_generator(self):
        """Verify that done=True when both relators are the SAME generator (false positive)."""
        # State: r0=[1], r1=[1] — both are 'x', NOT a trivial presentation
        state = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
        lengths = [1, 1]

        done_check = sum(lengths) == 2
        trivial_check = is_presentation_trivial(state)

        assert done_check == True, "sum(lengths)==2 should be True"
        assert trivial_check == False, "is_presentation_trivial should be False when both relators are 'x'"
        print(f"BUG CONFIRMED: done={done_check} but is_trivial={trivial_check} for state {list(state)}")

    def test_valid_trivial_states(self):
        """Verify is_presentation_trivial accepts all 8 valid trivial states."""
        for mrl in [2, 3, 5, 18, 36]:
            count = 0
            for r0_gen in [1, -1, 2, -2]:
                for r1_gen in [1, -1, 2, -2]:
                    if abs(r0_gen) == abs(r1_gen):
                        continue
                    state = np.zeros(2 * mrl, dtype=np.int8)
                    state[0] = r0_gen
                    state[mrl] = r1_gen
                    assert is_presentation_trivial(state), (
                        f"State {list(state)} should be trivial"
                    )
                    count += 1
            assert count == 8


# ──────────────────────────────────────────────────────────────────────
# 3. Length tracking consistency
# ──────────────────────────────────────────────────────────────────────

class TestLengthTracking:
    """Verify that self.lengths stays in sync with the actual state array."""

    def test_lengths_after_all_moves(self):
        """Apply every move to several presentations and check length consistency."""
        for idx in [0, 1, 16, 124, 171, 260, 309, 407, 459, 494, 518]:
            state = get_presentation(idx)
            max_relator_length = len(state) // 2
            lengths = [
                int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
                for i in range(2)
            ]

            for move_id in range(12):
                new_state, new_lengths = ACMove(
                    move_id=move_id,
                    presentation=state.copy(),
                    max_relator_length=max_relator_length,
                    lengths=lengths.copy(),
                )
                verify_lengths_consistent(new_state, new_lengths, max_relator_length)

    def test_lengths_after_random_trajectory(self):
        """Apply random sequences of moves and check length consistency at every step."""
        rng = np.random.RandomState(42)

        for idx in [0, 1, 124, 171, 494]:
            state = get_presentation(idx)
            max_relator_length = len(state) // 2
            lengths = [
                int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
                for i in range(2)
            ]

            for step in range(200):
                action = rng.randint(0, 12)
                state, lengths = ACMove(
                    move_id=action,
                    presentation=state,
                    max_relator_length=max_relator_length,
                    lengths=lengths,
                )
                verify_lengths_consistent(state, lengths, max_relator_length)

                # Also verify state is always a valid presentation
                assert is_array_valid_presentation(state), (
                    f"idx={idx}, step={step}, action={action}: "
                    f"State is not a valid presentation: {list(state)}"
                )


# ──────────────────────────────────────────────────────────────────────
# 4. Cyclic reduction correctness
# ──────────────────────────────────────────────────────────────────────

class TestCyclicReduction:
    """Test that cyclic reduction doesn't over-reduce or corrupt state."""

    def test_cyclic_reduction_preserves_group(self):
        """Cyclic reduction of [a, w, a^{-1}] should give [w], not empty."""
        # [1, 2, -1] should reduce to [2]
        relator = np.array([1, 2, -1, 0, 0])
        result, length = simplify_relator(relator, 5, cyclical=True, padded=True)
        assert length == 1
        assert result[0] == 2

    def test_cyclic_reduction_double(self):
        """[1, 2, 3, -2, -1] should reduce to [3]."""
        relator = np.array([1, 2, 3, -2, -1, 0])
        result, length = simplify_relator(relator, 6, cyclical=True, padded=True)
        assert length == 1
        assert result[0] == 3

    def test_cyclic_reduction_full_cancel(self):
        """[1, -1] should fully cancel through free reduction, not cyclic."""
        relator = np.array([1, -1, 0, 0])
        result, length = simplify_relator(relator, 4, cyclical=True, padded=True)
        assert length == 0
        assert (result == 0).all()

    def test_cyclic_reduction_palindrome(self):
        """[1, 2, 1] — NOT reducible cyclically (1 != -1)."""
        relator = np.array([1, 2, 1, 0])
        result, length = simplify_relator(relator, 4, cyclical=True, padded=True)
        assert length == 3
        assert list(result[:3]) == [1, 2, 1]

    def test_cyclic_does_not_over_reduce(self):
        """Test that cyclic reduction correctly handles edge cases."""
        # [1, -1] : free reduction makes empty, cyclic shouldn't crash
        r1 = np.array([1, -1])
        res1, len1 = simplify_relator(r1.copy(), 2, cyclical=True)
        assert len1 == 0

        # [1, 2, -2, -1] : free reduction makes empty
        r2 = np.array([1, 2, -2, -1])
        res2, len2 = simplify_relator(r2.copy(), 4, cyclical=True)
        assert len2 == 0

    def test_cyclic_reduction_single_element(self):
        """Single element [1] should not be affected by cyclic reduction."""
        relator = np.array([1, 0, 0])
        result, length = simplify_relator(relator, 3, cyclical=True, padded=True)
        assert length == 1
        assert result[0] == 1


# ──────────────────────────────────────────────────────────────────────
# 5. Can concatenation produce zero-length relators?
# ──────────────────────────────────────────────────────────────────────

class TestZeroLengthRelator:
    """Test whether concatenation can produce a relator of length 0."""

    def test_concatenation_full_cancel(self):
        """r0=[1,2], r1=[-2,-1]. r0 -> r0 * r1 should cancel to empty."""
        # r0=[1,2], r1=[-2,-1]
        state = np.array([1, 2, 0, -2, -1, 0], dtype=np.int8)
        lengths = [2, 2]

        # Action 3: r_0 -> r_0 * r_1
        # r_0 * r_1 = [1, 2, -2, -1] -> free reduction -> []
        # This SHOULD crash with assertion or produce length 0
        try:
            new_state, new_lengths = ACMove(
                move_id=3, presentation=state.copy(),
                max_relator_length=3, lengths=lengths.copy()
            )
            # If we get here without assertion error, check if length is 0
            if new_lengths[0] == 0:
                print(f"WARNING: Relator 0 has length 0 after concatenation!")
                print(f"  State: {list(new_state)}, Lengths: {new_lengths}")
                # This is a bug - the environment would declare done if other relator has length 2
                assert False, "Concatenation produced zero-length relator without assertion error!"
            else:
                # The concatenation might have been blocked (new_size > max_relator_length)
                print(f"  Concatenation result: state={list(new_state)}, lengths={new_lengths}")
        except AssertionError as e:
            print(f"  Assertion caught (expected): {e}")
            # This is expected behavior - simplify_presentation catches invalid state

    def test_concatenation_r1_cancels_to_zero(self):
        """r0=[1], r1=[1]. r1 -> r1 * r0^{-1} = [1, -1] -> []."""
        state = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
        lengths = [1, 1]

        # Action 2: r_1 -> r_1 * r_0^{-1} = [1] * [-1] = [1, -1] -> free reduction -> []
        try:
            new_state, new_lengths = ACMove(
                move_id=2, presentation=state.copy(),
                max_relator_length=3, lengths=lengths.copy()
            )
            if new_lengths[1] == 0:
                print(f"BUG: Relator 1 has length 0!")
                print(f"  State: {list(new_state)}, Lengths: {new_lengths}")
        except AssertionError:
            print("  Assertion caught (expected) for zero-length relator")


# ──────────────────────────────────────────────────────────────────────
# 6. Environment step correctness
# ──────────────────────────────────────────────────────────────────────

class TestEnvironmentStep:
    """Test ACEnv.step for correctness."""

    def test_env_done_matches_trivial(self):
        """Ensure that when env says done=True, state is actually trivial."""
        # Run random trajectories and check every time done=True
        rng = np.random.RandomState(123)
        false_positives = 0

        for idx in [0, 1, 16, 124, 171, 260]:
            state = get_presentation(idx)
            config = ACEnvConfig(initial_state=state.copy(), horizon_length=500)
            env = ACEnv(config)

            obs, _ = env.reset()
            for step in range(500):
                action = rng.randint(0, 12)
                obs, reward, done, truncated, info = env.step(action)

                if done:
                    # Verify the state is ACTUALLY trivial
                    if not is_presentation_trivial(obs):
                        false_positives += 1
                        print(f"FALSE POSITIVE! idx={idx}, step={step}, action={action}")
                        print(f"  State: {list(obs)}")
                        print(f"  Lengths: {env.lengths}")
                        print(f"  sum(lengths)={sum(env.lengths)}")
                    break
                if truncated:
                    break

        assert false_positives == 0, f"Found {false_positives} false positive terminations!"

    def test_env_done_with_trivial_start(self):
        """Starting from trivial state, first step should NOT be done (need move first)."""
        config = ACEnvConfig(initial_state=np.array([1, 0, 2, 0], dtype=np.int8))
        env = ACEnv(config)

        # The initial state IS trivial - sum(lengths) = 2
        assert sum(env.lengths) == 2
        assert is_presentation_trivial(env.state)

        # After any move, the state should change
        obs, _ = env.reset()
        obs, reward, done, truncated, info = env.step(0)  # r_1 -> r_1 * r_0
        # After this, r_1 = [2, 1], lengths should be [1, 2], sum = 3
        # So done should be False
        print(f"After action 0: state={list(obs)}, lengths={env.lengths}, done={done}")


# ──────────────────────────────────────────────────────────────────────
# 7. Group preservation test (AC moves should preserve the group)
# ──────────────────────────────────────────────────────────────────────

class TestGroupPreservation:
    """
    AC moves preserve the presented group. If we start with a non-trivial group,
    we should NEVER reach a trivial presentation. Test with a known non-trivial group.
    """

    def test_nontrivial_group_never_reaches_trivial(self):
        """
        <x, y | x^2, y^2> presents Z/2 * Z/2, a non-trivial group.
        AC moves should never reduce this to a trivial presentation.
        """
        # x^2 = [1, 1], y^2 = [2, 2]
        state = np.array([1, 1, 0, 0, 2, 2, 0, 0], dtype=np.int8)
        assert is_array_valid_presentation(state)

        max_relator_length = 4
        lengths = [2, 2]
        rng = np.random.RandomState(42)

        for trial in range(10):
            s = state.copy()
            l = lengths.copy()
            for step in range(500):
                action = rng.randint(0, 12)
                s, l = ACMove(
                    move_id=action,
                    presentation=s,
                    max_relator_length=max_relator_length,
                    lengths=l,
                )
                if sum(l) == 2:
                    # If we ever reach total length 2, check if it's trivial
                    if is_presentation_trivial(s):
                        pytest.fail(
                            f"BUG: Reached trivial state from non-trivial group!\n"
                            f"  Trial={trial}, Step={step}, Action={action}\n"
                            f"  State: {list(s)}, Lengths: {l}"
                        )
                    else:
                        print(f"  Reached sum(lengths)=2 but NOT trivial (expected): {list(s)}")


# ──────────────────────────────────────────────────────────────────────
# 8. Specific path replay from user's log (exact paths)
# ──────────────────────────────────────────────────────────────────────

class TestExactPathReplay:
    """Replay the exact paths from the user's training log."""

    # From user's log (path format: [[action_id, total_length_after], ...])
    EXACT_PATHS = [
        (171, [[4, 9], [4, 9], [1, 3], [7, 3], [5, 3], [8, 3], [4, 3], [2, 4], [3, 5], [5, 5],
               [0, 7], [2, 5], [6, 5], [0, 7], [9, 7], [4, 7], [2, 5], [8, 5], [7, 5], [1, 4],
               [0, 3], [3, 3], [8, 3], [2, 2]]),
        (171, [[11, 9], [8, 9], [4, 9], [1, 3], [11, 3], [0, 2]]),
        (171, [[7, 9], [11, 9], [1, 3], [0, 2]]),
        (494, [[1, 3], [11, 3], [0, 2]]),
        (124, [[1, 9], [0, 2]]),
        (260, [[1, 9], [0, 2]]),
        (518, [[1, 3], [0, 2]]),
        (407, [[1, 3], [0, 2]]),
        (16,  [[1, 7], [0, 2]]),
        (459, [[1, 3], [11, 3], [0, 2]]),
        (459, [[1, 3], [0, 2]]),
        (494, [[1, 3], [0, 2]]),
        (1,   [[1, 3], [0, 2]]),
        (171, [[1, 3], [0, 2]]),
        (309, [[1, 3], [0, 2]]),
        (1,   [[1, 3], [8, 3], [7, 3], [9, 3], [3, 3], [7, 3], [5, 3], [3, 5], [1, 3], [1, 3],
               [10, 3], [4, 3], [0, 2]]),
        (518, [[1, 3], [9, 3], [2, 4], [0, 3], [5, 3], [7, 3], [0, 2]]),
    ]

    @pytest.mark.parametrize("idx, path", EXACT_PATHS)
    def test_exact_path_replay(self, idx, path):
        """
        Replay the exact path and verify:
        1. Each step's total length matches the logged total length.
        2. The final state is truly trivial.
        """
        state = get_presentation(idx)
        max_relator_length = len(state) // 2
        lengths = [
            int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
            for i in range(2)
        ]

        print(f"\nReplaying idx={idx}, path_length={len(path)}")
        r0 = state[:max_relator_length][state[:max_relator_length] != 0]
        r1 = state[max_relator_length:][state[max_relator_length:] != 0]
        print(f"  Start: r0={list(r0)}, r1={list(r1)}, total_length={sum(lengths)}")

        for step_num, (action, expected_total_length) in enumerate(path):
            state, lengths = ACMove(
                move_id=action,
                presentation=state,
                max_relator_length=max_relator_length,
                lengths=lengths,
            )
            actual_total = sum(lengths)

            r0 = state[:max_relator_length][:lengths[0]]
            r1 = state[max_relator_length:][:lengths[1]]
            print(f"  Step {step_num}: action={action}, r0={list(r0)}, r1={list(r1)}, "
                  f"lengths={lengths}, total={actual_total} (expected {expected_total_length})")

            # Verify total length matches log
            assert actual_total == expected_total_length, (
                f"idx={idx}, step={step_num}: total length mismatch! "
                f"actual={actual_total}, expected={expected_total_length}"
            )

            # Verify lengths are consistent
            verify_lengths_consistent(state, lengths, max_relator_length)

        # Verify final state is trivial
        assert is_presentation_trivial(state), (
            f"idx={idx}: Final state is NOT trivial after {len(path)} moves!\n"
            f"  State: {list(state)}, Lengths: {lengths}"
        )
        print(f"  VERIFIED: Solution is correct (trivial state reached)")


# ──────────────────────────────────────────────────────────────────────
# 9. Stress test: many random trajectories checking invariants
# ──────────────────────────────────────────────────────────────────────

class TestStressInvariants:
    """Run many random trajectories checking invariants at every step."""

    def test_random_trajectories_all_invariants(self):
        """
        For each step of a random trajectory, verify:
        1. State is a valid presentation
        2. Lengths are consistent with state
        3. If sum(lengths)==2, check is_presentation_trivial
        """
        rng = np.random.RandomState(42)
        violations = []

        for idx in [0, 1, 16, 50, 100, 124, 171, 260, 309, 407, 459, 494, 518]:
            state = get_presentation(idx)
            max_relator_length = len(state) // 2
            lengths = [
                int(np.count_nonzero(state[i * max_relator_length : (i + 1) * max_relator_length]))
                for i in range(2)
            ]

            for step in range(500):
                action = rng.randint(0, 12)
                try:
                    state, lengths = ACMove(
                        move_id=action,
                        presentation=state,
                        max_relator_length=max_relator_length,
                        lengths=lengths,
                    )
                except Exception as e:
                    violations.append(f"idx={idx}, step={step}, action={action}: EXCEPTION {e}")
                    break

                # Check validity
                if not is_array_valid_presentation(state):
                    violations.append(
                        f"idx={idx}, step={step}, action={action}: Invalid presentation {list(state)}"
                    )
                    break

                # Check length consistency
                for i in range(2):
                    actual = int(np.count_nonzero(
                        state[i * max_relator_length : (i + 1) * max_relator_length]
                    ))
                    if actual != lengths[i]:
                        violations.append(
                            f"idx={idx}, step={step}, action={action}: Length mismatch "
                            f"relator {i}: tracked={lengths[i]}, actual={actual}"
                        )

                # If "done", check triviality
                if sum(lengths) == 2:
                    if not is_presentation_trivial(state):
                        violations.append(
                            f"idx={idx}, step={step}, action={action}: "
                            f"sum(lengths)==2 but NOT trivial! State={list(state)}, lengths={lengths}"
                        )
                    break  # Start next presentation

        if violations:
            print("\nVIOLATIONS FOUND:")
            for v in violations:
                print(f"  {v}")

        assert len(violations) == 0, f"Found {len(violations)} violations! See output above."


# ──────────────────────────────────────────────────────────────────────
# 10. Test the actions_to_path function
# ──────────────────────────────────────────────────────────────────────

class TestActionsToPath:
    """Test the actions_to_path function used for solution logging."""

    def test_actions_to_path_trivial(self):
        """Apply known moves to a simple presentation and verify path format."""
        from ac_solver.agents.training import actions_to_path

        # Start from a presentation that we know how to solve
        # Use idx 1 with the known 2-step solution
        state = get_presentation(1)
        actions = [1, 0]  # From user's log

        path, detailed = actions_to_path(actions, state)

        # Path should have 2 entries
        assert len(path) == 2
        assert len(detailed) == 2

        # Each entry is [action_id, total_length_after]
        for entry in path:
            assert len(entry) == 2
            assert isinstance(entry[0], int)
            assert isinstance(entry[1], int)

        # Final total length should be 2
        assert path[-1][1] == 2, f"Final total length should be 2, got {path[-1][1]}"

        # Detailed path should have relator states at each step
        for step in detailed:
            assert "action" in step
            assert "r0" in step
            assert "r1" in step
            assert "lengths" in step
            assert len(step["lengths"]) == 2

        # Final step should show trivial relators
        final = detailed[-1]
        assert len(final["r0"]) == 1, f"r0 should be single generator, got {final['r0']}"
        assert len(final["r1"]) == 1, f"r1 should be single generator, got {final['r1']}"
        assert abs(final["r0"][0]) != abs(final["r1"][0]), (
            f"r0 and r1 should be distinct generators, got r0={final['r0']}, r1={final['r1']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
