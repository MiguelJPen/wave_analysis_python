import torch.nn as nn


def build_model():
    return nn.Sequential(
        nn.Linear(3, 200),
        nn.ReLU(),
        nn.Conv1d(1, 1, 3, padding="same"),
        nn.MaxPool1d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 300)
    )
