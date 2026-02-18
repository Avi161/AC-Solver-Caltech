"""
Value network architectures for predicting distance-to-trivial.

Two architectures:
  - FeatureMLP: MLP on 14 handcrafted features
  - SequenceValueNet: 1D CNN on raw token sequences
"""

import numpy as np
import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """
    Architecture A: MLP on handcrafted features.
    14 features -> hidden layers with ReLU -> 1 scalar (predicted distance).
    """

    def __init__(self, input_dim=14, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Final layer uses smaller initialization
        final = list(self.net.children())[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)

    def forward(self, features):
        """
        Parameters:
            features: (batch, 14) float tensor (normalized features)
        Returns:
            (batch, 1) predicted distance to trivial
        """
        return self.net(features)


class SequenceValueNet(nn.Module):
    """
    Architecture B: 1D CNN on raw presentation token sequences.
    Vocabulary: {-2, -1, 0, 1, 2} mapped to token IDs {0, 1, 2, 3, 4}.
    """

    def __init__(self, vocab_size=5, embed_dim=32, num_conv_layers=3,
                 conv_channels=64, kernel_size=3, mlp_hidden=128,
                 max_seq_len=72, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=2)  # 0 maps to token 2

        # 1D conv stack
        conv_layers = []
        in_channels = embed_dim
        for _ in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = conv_channels
        self.conv = nn.Sequential(*conv_layers)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        # Final layer uses smaller initialization
        final = list(self.head.children())[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)

    def forward(self, token_ids):
        """
        Parameters:
            token_ids: (batch, seq_len) long tensor with values in [0, 4]
        Returns:
            (batch, 1) predicted distance
        """
        x = self.embedding(token_ids)     # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)            # (batch, embed_dim, seq_len)
        x = self.conv(x)                 # (batch, conv_channels, seq_len)
        x = x.mean(dim=2)               # (batch, conv_channels) global avg pool
        return self.head(x)              # (batch, 1)

    @staticmethod
    def presentation_to_token_ids(presentation):
        """Convert int8 presentation (values in {-2,-1,0,1,2}) to token IDs (values in {0,1,2,3,4})."""
        return (np.asarray(presentation, dtype=np.int64) + 2)
