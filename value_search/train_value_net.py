"""
Training pipeline for value networks.

Loads extracted dataset, trains FeatureMLP and/or SequenceValueNet,
saves best checkpoints, and reports evaluation metrics.
"""

import os
import sys
import json
import pickle
import argparse

# Ensure project root is on path when run as subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from value_search.value_network import FeatureMLP, SequenceValueNet


class ACDataset(Dataset):
    """PyTorch Dataset for AC presentations."""

    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.states = data['states']           # (N, max_state_dim) int8
        self.features = data['features']       # (N, 14) float32
        self.labels = data['labels']           # (N,) float32
        self.source_idx = data['source_idx']   # (N,) int32
        self.state_lengths = data['state_lengths']  # (N,) int32
        self.metadata = data['metadata']

        # Log-transform labels for better MSE on long-tail distribution
        self.log_labels = np.log1p(self.labels).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        state = self.states[idx]
        # Convert to token IDs for sequence model
        token_ids = (state.astype(np.int64) + 2)  # {-2..2} -> {0..4}
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(self.log_labels[idx], dtype=torch.float32),
            'raw_label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'source_idx': int(self.source_idx[idx]),
        }


def create_stratified_split(dataset, val_fraction=0.2, seed=42):
    """
    Create train/val split stratified by source presentation.
    All states from the same source presentation go to the same split
    to prevent data leakage.
    """
    rng = np.random.RandomState(seed)

    # Get unique source indices
    unique_sources = np.unique(dataset.source_idx)
    rng.shuffle(unique_sources)

    # Split sources into train/val
    n_val = max(1, int(len(unique_sources) * val_fraction))
    val_sources = set(unique_sources[:n_val])

    train_indices = []
    val_indices = []
    for i in range(len(dataset)):
        if dataset.source_idx[i] in val_sources:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices


def train_model(model, train_loader, val_loader, model_type='mlp',
                epochs=100, lr=1e-3, weight_decay=1e-4, patience=15,
                device='cpu', save_dir='value_search/checkpoints', log_every=5):
    """
    Training loop with MSE loss and early stopping.

    Returns dict with training history.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            if model_type == 'mlp':
                inputs = batch['features'].to(device)
            else:
                inputs = batch['token_ids'].to(device)
            targets = batch['label'].to(device)

            preds = model(inputs).squeeze(-1)
            loss = nn.functional.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(targets)
            train_count += len(targets)

        train_loss = train_loss_sum / train_count

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'mlp':
                    inputs = batch['features'].to(device)
                else:
                    inputs = batch['token_ids'].to(device)
                targets = batch['label'].to(device)
                raw_targets = batch['raw_label'].to(device)

                preds = model(inputs).squeeze(-1)
                loss = nn.functional.mse_loss(preds, targets)
                # MAE in original scale
                preds_orig = torch.expm1(preds)
                mae = torch.mean(torch.abs(preds_orig - raw_targets))

                val_loss_sum += loss.item() * len(targets)
                val_mae_sum += mae.item() * len(targets)
                val_count += len(targets)

        val_loss = val_loss_sum / val_count
        val_mae = val_mae_sum / val_count

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mae={val_mae:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_path = os.path.join(save_dir, f'best_{model_type}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best: epoch {best_epoch})")
                break

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    print(f"\nBest model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")

    # Save history
    history_path = os.path.join(save_dir, f'history_{model_type}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    return history


def evaluate_model(model, data_loader, model_type='mlp', device='cpu'):
    """
    Evaluate model. Returns dict with metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_raw_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'mlp':
                inputs = batch['features'].to(device)
            else:
                inputs = batch['token_ids'].to(device)

            preds = model(inputs).squeeze(-1)
            preds_orig = torch.expm1(preds)

            all_preds.extend(preds_orig.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            all_raw_labels.extend(batch['raw_label'].numpy())

    preds = np.array(all_preds)
    raw_labels = np.array(all_raw_labels)

    mse = float(np.mean((preds - raw_labels) ** 2))
    mae = float(np.mean(np.abs(preds - raw_labels)))

    # Correlation
    if np.std(preds) > 0 and np.std(raw_labels) > 0:
        correlation = float(np.corrcoef(preds, raw_labels)[0, 1])
    else:
        correlation = 0.0

    # R² score
    ss_res = np.sum((raw_labels - preds) ** 2)
    ss_tot = np.sum((raw_labels - np.mean(raw_labels)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'r2': r2,
        'predictions': preds,
        'labels': raw_labels,
    }


def compute_feature_stats(train_loader):
    """Compute mean and std of features from training set for normalization."""
    all_features = []
    for batch in train_loader:
        all_features.append(batch['features'].numpy())
    all_features = np.concatenate(all_features, axis=0)
    return {
        'mean': all_features.mean(axis=0).tolist(),
        'std': (all_features.std(axis=0) + 1e-8).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train value network for AC presentations')
    parser.add_argument('--data-path', type=str,
                        default='value_search/data/training_data.pkl')
    parser.add_argument('--architecture', type=str, default='both',
                        choices=['mlp', 'seq', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='value_search/checkpoints')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = ACDataset(args.data_path)
    print(f"  Total examples: {len(dataset)}")
    print(f"  Metadata: {dataset.metadata}")

    # Split
    train_idx, val_idx = create_stratified_split(dataset, val_fraction=0.2, seed=args.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Compute feature normalization stats
    feat_stats = compute_feature_stats(train_loader)
    feat_stats_path = os.path.join(args.save_dir, 'feature_stats.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(feat_stats_path, 'w') as f:
        json.dump(feat_stats, f)
    print(f"  Feature stats saved to {feat_stats_path}")

    # Normalize features in-place for MLP
    feat_mean = np.array(feat_stats['mean'], dtype=np.float32)
    feat_std = np.array(feat_stats['std'], dtype=np.float32)
    dataset.features = (dataset.features - feat_mean) / feat_std

    # Re-create loaders after normalization
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    architectures = []
    if args.architecture in ('mlp', 'both'):
        architectures.append('mlp')
    if args.architecture in ('seq', 'both'):
        architectures.append('seq')

    for arch in architectures:
        print(f"\n{'='*50}")
        print(f"Training {arch.upper()} model")
        print(f"{'='*50}")

        if arch == 'mlp':
            model = FeatureMLP(input_dim=14, hidden_dims=[256, 256, 128], dropout=0.1)
        else:
            max_state_dim = dataset.metadata['max_state_dim']
            model = SequenceValueNet(max_seq_len=max_state_dim, dropout=0.1)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        history = train_model(
            model, train_loader, val_loader,
            model_type=arch, epochs=args.epochs, lr=args.lr,
            patience=args.patience, device=device, save_dir=args.save_dir,
        )

        # Reload best model for evaluation
        ckpt = torch.load(os.path.join(args.save_dir, f'best_{arch}.pt'),
                          map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])

        print(f"\nEvaluation on validation set:")
        metrics = evaluate_model(model, val_loader, model_type=arch, device=device)
        print(f"  MSE: {metrics['mse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")


if __name__ == '__main__':
    main()
