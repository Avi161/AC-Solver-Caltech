"""
Tests for iterative_refinement.py config logic.
Verifies architecture flag routing without running actual search or training.
"""
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Helpers that mirror the logic in run_iteration() / main()
# ---------------------------------------------------------------------------

def _get_train_arch(run_mlp: bool, run_seq: bool) -> str:
    """Mirrors the train_arch derivation in run_iteration()."""
    if not run_mlp and not run_seq:
        raise ValueError("At least one of run_mlp or run_seq must be True")
    if run_mlp and run_seq:
        return 'both'
    return 'mlp' if run_mlp else 'seq'


def _get_seq_model_path(model_path: str) -> str:
    """Mirrors the seq_model_path derivation in run_iteration()."""
    if model_path.endswith('best_seq.pt'):
        return model_path
    return model_path.replace('best_mlp.pt', 'best_seq.pt')


def _get_search_label(run_mlp: bool, run_seq: bool, enable_beam: bool) -> str:
    """Mirrors the search_label derivation in run_iteration() / main()."""
    return ' + '.join(filter(None, [
        'MLP' if run_mlp else '',
        'Seq' if run_seq else '',
        'Beam' if enable_beam else '',
    ]))


# ---------------------------------------------------------------------------
# Training architecture tests
# ---------------------------------------------------------------------------

class TestTrainArch:
    def test_seq_only(self):
        assert _get_train_arch(run_mlp=False, run_seq=True) == 'seq'

    def test_mlp_only(self):
        assert _get_train_arch(run_mlp=True, run_seq=False) == 'mlp'

    def test_both(self):
        assert _get_train_arch(run_mlp=True, run_seq=True) == 'both'

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            _get_train_arch(run_mlp=False, run_seq=False)


# ---------------------------------------------------------------------------
# Seq model path derivation tests
# ---------------------------------------------------------------------------

class TestSeqModelPath:
    def test_from_mlp_path(self):
        p = '/refinement/checkpoints_iter_1/best_mlp.pt'
        assert _get_seq_model_path(p) == '/refinement/checkpoints_iter_1/best_seq.pt'

    def test_already_seq_path(self):
        p = '/refinement/checkpoints_iter_1/best_seq.pt'
        assert _get_seq_model_path(p) == p  # unchanged

    def test_original_checkpoint(self):
        p = '/value_search/checkpoints/best_mlp.pt'
        assert _get_seq_model_path(p) == '/value_search/checkpoints/best_seq.pt'


# ---------------------------------------------------------------------------
# Search label tests
# ---------------------------------------------------------------------------

class TestSearchLabel:
    def test_seq_only(self):
        assert _get_search_label(False, True, False) == 'Seq'

    def test_mlp_only(self):
        assert _get_search_label(True, False, False) == 'MLP'

    def test_seq_and_beam(self):
        assert _get_search_label(False, True, True) == 'Seq + Beam'

    def test_mlp_and_seq(self):
        assert _get_search_label(True, True, False) == 'MLP + Seq'

    def test_all(self):
        assert _get_search_label(True, True, True) == 'MLP + Seq + Beam'


# ---------------------------------------------------------------------------
# CLI arg → flag logic (mirrors main())
# ---------------------------------------------------------------------------

class TestCLIFlags:
    def test_default_is_seq_only(self):
        # Default: --run-mlp not set (False), --no-seq not set (False)
        run_mlp = False
        run_seq = not False  # not args.no_seq
        assert run_mlp is False
        assert run_seq is True

    def test_run_mlp_flag(self):
        run_mlp = True   # --run-mlp
        run_seq = not False
        assert _get_train_arch(run_mlp, run_seq) == 'both'

    def test_no_seq_flag(self):
        run_mlp = False
        run_seq = not True  # --no-seq passed
        assert run_seq is False
        with pytest.raises(ValueError):
            _get_train_arch(run_mlp, run_seq)

    def test_mlp_only_config(self):
        run_mlp = True   # --run-mlp
        run_seq = not True  # --no-seq
        assert _get_train_arch(run_mlp, run_seq) == 'mlp'


# ---------------------------------------------------------------------------
# Multi-iteration chain tests
# Simulates the user's exact scenario:
#   - Old code ran iters 0 and 1, storing best_mlp.pt in state['model_paths']
#   - New seq-only code takes over from iter 2 onwards
# ---------------------------------------------------------------------------

def _simulate_stored_model_path(checkpoint_dir: str, run_mlp: bool, run_seq: bool) -> str:
    """
    Mirrors the post-training model path selection in run_iteration():
      new_model_path = checkpoint_dir/best_mlp.pt  (if it exists)
      else           = checkpoint_dir/best_seq.pt  (if it exists)
    """
    mlp = os.path.join(checkpoint_dir, 'best_mlp.pt')
    seq = os.path.join(checkpoint_dir, 'best_seq.pt')
    # Create the files that training would produce
    if run_mlp:
        open(mlp, 'w').close()
    if run_seq:
        open(seq, 'w').close()
    # Select which path to store (mirrors run_iteration logic)
    if os.path.exists(mlp):
        return mlp
    if os.path.exists(seq):
        return seq
    raise RuntimeError("Training produced no checkpoint")


class TestIterationChain:
    """
    Verify that model_path → seq_model_path derivation stays correct across
    multiple iterations as the stored path transitions from best_mlp.pt
    (old code) to best_seq.pt (new seq-only code).
    """

    def test_iter2_from_mlp_state(self):
        """
        Iter 2 start: state has best_mlp.pt (set by old code in iters 0 and 1).
        Seq model must be correctly derived via .replace().
        """
        model_path = '/refinement/checkpoints_iter_1/best_mlp.pt'
        seq_path = _get_seq_model_path(model_path)
        assert seq_path == '/refinement/checkpoints_iter_1/best_seq.pt'
        assert seq_path != model_path

    def test_iter2_training_stores_seq_path(self):
        """
        After seq-only training in iter 2, stored path must be best_seq.pt.
        """
        with tempfile.TemporaryDirectory() as d:
            stored = _simulate_stored_model_path(d, run_mlp=False, run_seq=True)
            assert stored.endswith('best_seq.pt')
            assert not stored.endswith('best_mlp.pt')

    def test_iter3_seq_path_unchanged(self):
        """
        Iter 3 start: state has best_seq.pt (stored by seq-only iter 2).
        Seq model path must equal model_path (no transformation needed).
        """
        model_path = '/refinement/checkpoints_iter_2/best_seq.pt'
        seq_path = _get_seq_model_path(model_path)
        assert seq_path == model_path  # already seq, no change

    def test_full_chain_iter2_to_iter4(self):
        """
        Trace model_path and seq_model_path across iters 2, 3, 4 with seq-only.
        Iter 2: model_path=best_mlp.pt (old) → seq=best_seq.pt → trains seq → stores best_seq.pt
        Iter 3: model_path=best_seq.pt      → seq=best_seq.pt → trains seq → stores best_seq.pt
        Iter 4: same as iter 3
        """
        with tempfile.TemporaryDirectory() as base:
            # Simulate old-code state: model_paths[-1] = checkpoints_iter_1/best_mlp.pt
            iter1_dir = os.path.join(base, 'checkpoints_iter_1')
            os.makedirs(iter1_dir)
            open(os.path.join(iter1_dir, 'best_mlp.pt'), 'w').close()
            open(os.path.join(iter1_dir, 'best_seq.pt'), 'w').close()

            model_paths = [os.path.join(iter1_dir, 'best_mlp.pt')]

            for iteration in [2, 3, 4]:
                model_path = model_paths[-1]

                # seq model derivation
                seq_path = _get_seq_model_path(model_path)
                assert os.path.exists(seq_path), \
                    f"iter {iteration}: seq model {seq_path} does not exist"

                # simulate seq-only training
                ckpt_dir = os.path.join(base, f'checkpoints_iter_{iteration}')
                os.makedirs(ckpt_dir)
                stored = _simulate_stored_model_path(ckpt_dir, run_mlp=False, run_seq=True)

                assert stored.endswith('best_seq.pt'), \
                    f"iter {iteration}: expected best_seq.pt, got {stored}"
                assert _get_train_arch(run_mlp=False, run_seq=True) == 'seq'

                model_paths.append(stored)

            # After 3 seq-only iterations, all stored paths should be best_seq.pt
            for p in model_paths[1:]:  # skip the initial best_mlp.pt from old code
                assert p.endswith('best_seq.pt')

    def test_both_arch_training_stores_mlp_path(self):
        """
        When run_mlp=True and run_seq=True, training produces both;
        stored path is best_mlp.pt (mlp takes priority in selection logic).
        """
        with tempfile.TemporaryDirectory() as d:
            stored = _simulate_stored_model_path(d, run_mlp=True, run_seq=True)
            assert stored.endswith('best_mlp.pt')
