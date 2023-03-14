import torch.nn as nn


def create_net_100():
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=3, padding = "same"),
        nn.MaxPool1d(2), # Capa de pooling
        nn.ReLU(),
        nn.Conv1d(1, 1, kernel_size=3, padding = "same"),
        nn.MaxPool1d(2), # Capa de pooling
        nn.ReLU(),
        nn.Flatten(), # Aplanar la salida de la capa de convolución
        nn.Linear(25, 30), # Capa totalmente conectada
        nn.Linear(30, 3)
    )


def create_net_300():
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=3, padding = "same"),
        nn.MaxPool1d(2), # Capa de pooling
        nn.ReLU(),
        nn.Linear(150, 80),
        nn.Linear(80, 40),
        nn.Linear(40, 20), # Capa totalmente conectada
        nn.Linear(20, 3)
    )