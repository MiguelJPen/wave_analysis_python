import torch
from sklearn.model_selection import train_test_split


def load_data(data, targets):
    # Dividir los datos en conjuntos de entrenamiento, validación y prueba (60%, 28%, 12%)

    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.4, random_state=42)
    val_data, test_data, val_targets, test_targets = train_test_split(test_data, test_targets, test_size=0.3, random_state=42)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    return train_data, val_data, test_data, train_targets, val_targets, test_targets
