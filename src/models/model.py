import torch.nn as nn


def build_model():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 3, padding=1),
        nn.Linear(32 * 3, 128),
        nn.Linear(128, 300)
    )
