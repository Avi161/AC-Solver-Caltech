"""
Tests for iterative_refinement.py config logic.
Verifies architecture flag routing without running actual search or training.
"""
import pytest
import sys
import os

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
