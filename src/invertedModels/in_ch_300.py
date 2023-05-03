import torch.nn as nn


def build_300in_net():
    return nn.Sequential(
        nn.Linear(300, 300),
        nn.ReLU(),
        nn.Conv1d(1, 1, kernel_size=3, padding="same"),
        nn.AvgPool1d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(150, 150),
        nn.ReLU(),
        nn.Linear(150, 66),
        nn.ReLU(),
        nn.Linear(66, 3)
    )


def build_300in_2out_net():
    return nn.Sequential(
        nn.Linear(300, 100),
        nn.Conv1d(1, 1, kernel_size=3, padding="same"),
        nn.MaxPool1d(2),  # Capa de pooling
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(50, 66),
        nn.Linear(66, 22),
        nn.Linear(22, 2)
    )