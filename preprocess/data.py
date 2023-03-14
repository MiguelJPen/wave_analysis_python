import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    lhs_10K_100 = "preprocess/results_lhs_10K_100.csv"
    lhs_10K_300 = "preprocess/results_lhs_10K_300.csv"
    lhs_50K_300 = "preprocess/results_lhs_50K_300.csv"
    lhs_100K_100 = "preprocess/results_lhs_100K_100.csv"
    equispaced_26K_300 = "preprocess/results_26K_300.csv"
    equispaced_33K_100 = "preprocess/results_33K_100.csv"
    equispaced_33K_300 = "preprocess/results_33K_300.csv"
    equispaced_75K_100 = "preprocess/results_75K_100.csv"


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


def load_csv(csv_filename):
    df = pd.read_csv(csv_filename)

    target_columns = df.columns[:3]
    data_columns = df.columns[4:-1]

    targets = df[target_columns].values
    data = df[data_columns].values
    return data, targets


def normalize(v):
    return (v - np.mean(v)) / np.std(v), np.mean(v), np.std(v)


def get_data(dataset):
    data, target = load_csv(dataset)

    target, target_mean, target_dev = normalize(target)
    data, data_mean, data_dev = normalize(data)

    print(target_mean, target_dev)
    print(data_mean, data_dev)
    print(target)
    print(data)

    return load_data(data, target), target_dev
