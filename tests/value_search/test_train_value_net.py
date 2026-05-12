"""Unit tests for value_search.train_value_net.

Covers ACDataset, create_stratified_split, compute_feature_stats,
train_model (one short pass), and evaluate_model.
"""

import os
import pickle
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from value_search.train_value_net import (
    ACDataset,
    compute_feature_stats,
    create_stratified_split,
    evaluate_model,
    train_model,
)
from value_search.value_network import FeatureMLP, SequenceValueNet


def _make_synthetic_dataset_pickle(path, n=40, max_state_dim=20, feat_dim=14, seed=0):
    """Generate a tiny synthetic dataset compatible with ACDataset."""
    rng = np.random.RandomState(seed)
    states = rng.randint(-2, 3, size=(n, max_state_dim)).astype(np.int8)
    features = rng.randn(n, feat_dim).astype(np.float32)
    labels = rng.uniform(0, 20, size=n).astype(np.float32)
    # Mix positive (>=0) and negative (-1, -2, ...) source indices for split logic
    source_idx = np.concatenate([
        np.arange(n // 2, dtype=np.int32),
        -1 - np.arange(n - n // 2, dtype=np.int32),
    ])
    state_lengths = np.full(n, max_state_dim, dtype=np.int32)
    metadata = {"max_state_dim": max_state_dim, "feature_dim": feat_dim}

    data = {
        "states": states,
        "features": features,
        "labels": labels,
        "source_idx": source_idx,
        "state_lengths": state_lengths,
        "metadata": metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


@pytest.fixture
def synth_dataset_path(tmp_path):
    p = tmp_path / "synth.pkl"
    _make_synthetic_dataset_pickle(str(p))
    return str(p)


class TestACDataset:
    def test_load_and_length(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        assert len(ds) == 40

    def test_item_schema(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        item = ds[0]
        assert set(item.keys()) >= {
            "features", "token_ids", "label", "raw_label", "source_idx"
        }
        assert item["features"].shape == (14,)
        assert item["features"].dtype == torch.float32
        # Token IDs should map int8 {-2..2} → {0..4}
        assert item["token_ids"].dtype == torch.long
        assert int(item["token_ids"].min()) >= 0
        assert int(item["token_ids"].max()) <= 4
        # label is log1p-transformed
        assert item["label"].dtype == torch.float32

    def test_log_label_transform(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        # log_labels = log(1 + labels)
        np.testing.assert_allclose(
            ds.log_labels, np.log1p(ds.labels).astype(np.float32)
        )


class TestStratifiedSplit:
    def test_split_disjoint(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        train_idx, val_idx = create_stratified_split(ds, val_fraction=0.2, seed=42)
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) + len(val_idx) == len(ds)

    def test_no_source_leakage(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        train_idx, val_idx = create_stratified_split(ds, val_fraction=0.3, seed=7)
        train_sources = set(int(ds.source_idx[i]) for i in train_idx)
        val_sources = set(int(ds.source_idx[i]) for i in val_idx)
        assert train_sources.isdisjoint(val_sources)

    def test_seed_reproducible(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        t1, v1 = create_stratified_split(ds, val_fraction=0.2, seed=42)
        t2, v2 = create_stratified_split(ds, val_fraction=0.2, seed=42)
        assert t1 == t2 and v1 == v2


class TestComputeFeatureStats:
    def test_returns_lists_of_correct_length(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        stats = compute_feature_stats(loader)
        assert "mean" in stats and "std" in stats
        assert len(stats["mean"]) == 14
        assert len(stats["std"]) == 14
        # std should be strictly positive (floor of 1e-8 is added)
        assert all(s > 0 for s in stats["std"])

    def test_mean_matches_numpy(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        stats = compute_feature_stats(loader)
        np.testing.assert_allclose(
            np.array(stats["mean"], dtype=np.float32),
            ds.features.mean(axis=0),
            atol=1e-5,
        )


class TestTrainModelSmoke:
    """Run a tiny training loop and confirm checkpoints + history are written."""

    def test_train_mlp_short_loop(self, synth_dataset_path, tmp_path):
        ds = ACDataset(synth_dataset_path)
        train_idx, val_idx = create_stratified_split(ds, val_fraction=0.3, seed=0)
        train_sub = torch.utils.data.Subset(ds, train_idx)
        val_sub = torch.utils.data.Subset(ds, val_idx)
        train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)

        model = FeatureMLP(input_dim=14, hidden_dims=[16, 8])
        save_dir = tmp_path / "ckpt"
        history = train_model(
            model, train_loader, val_loader,
            model_type="mlp", epochs=3, lr=1e-3, patience=10,
            device="cpu", save_dir=str(save_dir), log_every=10,
        )
        assert "train_loss" in history and len(history["train_loss"]) >= 1
        assert "val_loss" in history and len(history["val_loss"]) >= 1
        # Best checkpoint should exist
        assert (save_dir / "best_mlp.pt").exists()
        # History JSON should exist
        assert (save_dir / "history_mlp.json").exists()
        # Best val loss should be a finite float
        assert np.isfinite(history["best_val_loss"])

    def test_train_seq_short_loop(self, synth_dataset_path, tmp_path):
        ds = ACDataset(synth_dataset_path)
        train_idx, val_idx = create_stratified_split(ds, val_fraction=0.3, seed=0)
        train_sub = torch.utils.data.Subset(ds, train_idx)
        val_sub = torch.utils.data.Subset(ds, val_idx)
        train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)

        model = SequenceValueNet(
            max_seq_len=ds.metadata["max_state_dim"],
            embed_dim=8, num_conv_layers=1, conv_channels=16, mlp_hidden=16,
        )
        save_dir = tmp_path / "ckpt_seq"
        history = train_model(
            model, train_loader, val_loader,
            model_type="seq", epochs=2, lr=1e-3, patience=10,
            device="cpu", save_dir=str(save_dir), log_every=10,
        )
        assert (save_dir / "best_seq.pt").exists()
        assert np.isfinite(history["best_val_loss"])


class TestEvaluateModel:
    def test_metrics_keys_and_types(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        model = FeatureMLP(input_dim=14, hidden_dims=[16])
        metrics = evaluate_model(model, loader, model_type="mlp", device="cpu")
        for key in ("mse", "mae", "correlation", "r2", "predictions", "labels"):
            assert key in metrics
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["correlation"], float)
        assert isinstance(metrics["r2"], float)
        assert len(metrics["predictions"]) == len(ds)
        assert len(metrics["labels"]) == len(ds)

    def test_seq_evaluate_runs(self, synth_dataset_path):
        ds = ACDataset(synth_dataset_path)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        model = SequenceValueNet(
            max_seq_len=ds.metadata["max_state_dim"],
            embed_dim=8, num_conv_layers=1, conv_channels=16, mlp_hidden=16,
        )
        metrics = evaluate_model(model, loader, model_type="seq", device="cpu")
        assert np.isfinite(metrics["mse"])
        assert np.isfinite(metrics["mae"])
