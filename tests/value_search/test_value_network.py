"""Unit tests for value_search.value_network (FeatureMLP, SequenceValueNet)."""

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.value_network import FeatureMLP, SequenceValueNet


class TestFeatureMLP:
    def test_default_init_shapes(self):
        model = FeatureMLP()
        x = torch.zeros(4, 14)
        out = model(x)
        assert out.shape == (4, 1)
        assert out.dtype == torch.float32

    def test_custom_input_dim(self):
        model = FeatureMLP(input_dim=8, hidden_dims=[32, 16])
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 1)

    def test_custom_hidden_dims(self):
        model = FeatureMLP(input_dim=14, hidden_dims=[64, 32])
        x = torch.randn(2, 14)
        out = model(x)
        assert out.shape == (2, 1)
        # Linear layers: 14->64, 64->32, 32->1 (3 Linear modules)
        n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        assert n_linear == 3

    def test_forward_no_nans(self):
        model = FeatureMLP()
        x = torch.randn(32, 14) * 10
        out = model(x)
        assert torch.isfinite(out).all()

    def test_backward_computes_gradients(self):
        model = FeatureMLP()
        x = torch.randn(8, 14, requires_grad=False)
        target = torch.randn(8, 1)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        # Every Linear weight should have a non-None gradient
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                assert m.weight.grad is not None
                assert m.weight.grad.shape == m.weight.shape

    def test_eval_mode_disables_dropout(self):
        # With dropout=0.5, outputs should differ in train mode and match in eval
        torch.manual_seed(0)
        model = FeatureMLP(dropout=0.5)
        x = torch.randn(16, 14)
        model.eval()
        out1 = model(x)
        out2 = model(x)
        # Eval mode is deterministic
        torch.testing.assert_close(out1, out2)

    def test_final_layer_small_init(self):
        # Final layer uses orthogonal init with gain=0.01, so its weight norm
        # should be much smaller than earlier layers.
        model = FeatureMLP()
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert linears[-1].weight.abs().max().item() < 0.5


class TestSequenceValueNet:
    def test_default_init_shapes(self):
        model = SequenceValueNet(max_seq_len=72)
        x = torch.zeros(4, 72, dtype=torch.long)
        out = model(x)
        assert out.shape == (4, 1)

    def test_custom_max_seq_len(self):
        model = SequenceValueNet(max_seq_len=36)
        x = torch.zeros(2, 36, dtype=torch.long)
        out = model(x)
        assert out.shape == (2, 1)

    def test_input_must_be_long_tensor(self):
        model = SequenceValueNet(max_seq_len=72)
        # Embedding requires long/int tensor
        x = torch.zeros(2, 72, dtype=torch.long)
        out = model(x)
        assert out.shape == (2, 1)

    def test_token_id_range_accepted(self):
        model = SequenceValueNet(vocab_size=5, max_seq_len=10)
        x = torch.tensor([[0, 1, 2, 3, 4, 2, 2, 2, 2, 2],
                          [4, 3, 2, 1, 0, 2, 2, 2, 2, 2]], dtype=torch.long)
        out = model(x)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_backward_computes_gradients(self):
        model = SequenceValueNet(max_seq_len=20)
        x = torch.randint(0, 5, (4, 20), dtype=torch.long)
        target = torch.randn(4, 1)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        # Embedding has gradients
        assert model.embedding.weight.grad is not None
        # Conv layers have gradients
        for m in model.modules():
            if isinstance(m, torch.nn.Conv1d):
                assert m.weight.grad is not None

    def test_eval_mode_deterministic(self):
        torch.manual_seed(0)
        model = SequenceValueNet(max_seq_len=20, dropout=0.5)
        x = torch.randint(0, 5, (8, 20), dtype=torch.long)
        model.eval()
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_presentation_to_token_ids_static(self):
        # Encoding: int8 values {-2,-1,0,1,2} → token IDs {0,1,2,3,4}
        pres = np.array([1, 2, 0, -1, -2], dtype=np.int8)
        token_ids = SequenceValueNet.presentation_to_token_ids(pres)
        np.testing.assert_array_equal(token_ids, [3, 4, 2, 1, 0])
        assert token_ids.dtype == np.int64

    def test_presentation_to_token_ids_round_trip_through_model(self):
        model = SequenceValueNet(max_seq_len=8)
        pres = np.array([1, 2, -1, -2, 0, 0, 0, 0], dtype=np.int8)
        token_ids = SequenceValueNet.presentation_to_token_ids(pres)
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        out = model(x)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all()
