import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    eq_300_30K = "preprocess/rawData/eq_300_30K.csv"
    eq_300_64K = "preprocess/rawData/eq_300_64K.csv"
    eq_600_30K = "preprocess/rawData/eq_600_30K.csv"

    lhs_300_30K = "preprocess/rawData/lhs_300_30K.csv"
    lhs_300_64K = "preprocess/rawData/lhs_300_64K.csv"

    lhs_300_t20_25K = "preprocess/rawData/lhs_300_t20_25K.csv"
    lhs_600_t20_50K = "preprocess/rawData/lhs_600_t20_50K.csv"

    lhs_2out_300_x0_26 = "preprocess/rawData/lhs_2out_300_x0_2.6.csv"
    lhs_2out_300_x0_5 = "preprocess/rawData/lhs_2out_300_x0_5.csv"
    lhs_2out_300_x0_74 = "preprocess/rawData/lhs_2out_300_x0_7.4.csv"


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

    return load_data(data, target), target_dev


def load_csv_2param(csv_filename):
    df = pd.read_csv(csv_filename)

    target_columns = df.columns[:2]
    data_columns = df.columns[3:-1]

    targets = df[target_columns].values
    data = df[data_columns].values
    return data, targets


def get_data_2param(dataset):
    data, target = load_csv_2param(dataset)

    target, target_mean, target_dev = normalize(target)
    data, data_mean, data_dev = normalize(data)

    return load_data(data, target), target_dev
