"""A small one-dimensional residual network for single-beat ECG classification."""
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, stride, 3, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 7, 1, 3, bias=False), nn.BatchNorm1d(c_out))
        self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class BeatNet(nn.Module):
    """Convolutional branch reads the beat (and, as a second channel, its deviation
    from the patient's median beat); a small dense branch reads the RR-interval
    ratios; the two are joined before the classifier."""

    def __init__(self, n_classes=4, n_rr=4, use_rr=True, in_channels=2, width=16, dropout=0.5):
        super().__init__()
        self.use_rr = use_rr
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, 1, 3, bias=False), nn.BatchNorm1d(width), nn.ReLU(),
            ResBlock(width, width, 2), ResBlock(width, 2 * width, 2), ResBlock(2 * width, 4 * width, 2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.rr = nn.Sequential(nn.Linear(n_rr, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width + (16 if use_rr else 0), n_classes))

    def forward(self, beat, rr):
        h = self.conv(beat)                          # beat: (batch, channels, samples)
        if self.use_rr:
            h = torch.cat([h, self.rr(rr)], dim=1)
        return self.head(h)
