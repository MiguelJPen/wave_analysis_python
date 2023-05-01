import torch.nn as nn


def build_600in_net():
    return nn.Sequential(
        nn.Linear(600, 500),
        nn.Conv1d(1, 4, kernel_size=5, padding="same"),
        nn.MaxPool1d(5),  # Capa de pooling
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.Conv1d(4, 2, kernel_size=5, padding="same"),
        nn.MaxPool1d(2),  # Capa de pooling
        nn.ReLU(),
        nn.Linear(100, 50),  # Capa totalmente conectada
        nn.Flatten(),
        nn.Linear(100, 66),
        nn.Linear(66, 3)
    )

# def create_net_600():
#    return nn.Sequential(
#        nn.Conv1d(1, 64, kernel_size=3),
#        nn.ReLU(),
#        nn.MaxPool1d(kernel_size=2),
#        nn.Conv1d(64, 128, kernel_size=3),
#        nn.ReLU(),
#        nn.MaxPool1d(kernel_size=2),
#        nn.Flatten(),
#        nn.Linear(18944, 64),
#        nn.ReLU(),
#        nn.Linear(64, 3),
#        nn.Softmax(dim=1)
#    )
