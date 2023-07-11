import torch.nn as nn


def build_600in_net():
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=5, padding="same"),
        nn.MaxPool1d(2),  # Capa de pooling
        nn.ReLU(),
        nn.Conv1d(1, 1, kernel_size=5, padding="same"),
        nn.MaxPool1d(2),  # Capa de pooling
        nn.ReLU(),
        nn.Linear(150, 150),  # Capa totalmente conectada
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(150, 66),
        nn.ReLU(),
        nn.Linear(66, 3)
    )
