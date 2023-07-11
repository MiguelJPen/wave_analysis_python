import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PATH = "rawData/"


class Dataset:
    eq_300_1K = PATH + "eq_300_1K.csv", 3, "eq_300_1K"
    eq_300_3K = PATH + "eq_300_3K.csv", 3, "eq_300_3K"
    eq_300_10K = PATH + "eq_300_10K.csv", 3, "eq_300_10K"
    eq_300_30K = PATH + "eq_300_30K.csv", 3, "eq_300_30K"

    lhs_300_1K = PATH + "lhs_300_1K.csv", 3, "lhs_300_1K"
    lhs_300_3K = PATH + "lhs_300_3K.csv", 3, "lhs_300_3K"
    lhs_300_10K = PATH + "lhs_300_10K.csv", 3, "lhs_300_10K"
    lhs_300_30K = PATH + "lhs_300_30K.csv", 3, "lhs_300_30K"
    lhs_300_100K = PATH + "lhs_300_100K.csv", 3, "lhs_300_100K"

    lhs_600_t20_25K = PATH + "lhs_600_t20_25K.csv", 3, "lhs_600_t20_25K"
    lhs_600_t20_50K = PATH + "lhs_600_t20_50K.csv", 3, "lhs_600_t20_50K"

    # Not used
    all_fixed = PATH + "all_fixed.csv", 3, "all_fixed"

    lhs_2out_300_x0_26 = PATH + "lhs_2out_300_x0_2.6.csv", 2, "lhs_2out_300_x0_2.6"
    lhs_2out_300_x0_5 = PATH + "lhs_2out_300_x0_5.csv", 2, "lhs_2out_300_x0_5"
    lhs_2out_300_x0_74 = PATH + "lhs_2out_300_x0_7.4.csv", 2, "lhs_2out_300_x0_7.4"


def data_from_file(dataset, inverted=False, full=False, norm=True):
    data, target = load_csv(dataset, inverted)

    return transform_data(data, target, inverted, full, norm), compute_range(target, inverted)


def transform_data(data, target, inverted, full, norm):
    if not norm:
        return split_data(data, target, full)
    if inverted:
        target, target_dev = normalize_by_column(target)
        data, _ = normalize(data)
    else:
        target, target_dev = normalize(target)
        data, _ = normalize_by_column(data)

    return split_data(data, target, full), target_dev


def compute_range(dataset, inverted=False):
    max_arr = dataset.max(axis=0)
    min_arr = dataset.min(axis=0)

    return (max_arr - min_arr) if inverted else (max(max_arr) - min(min_arr))


def split_data(data, targets, full):
    # Dividir los datos en conjuntos de entrenamiento, validación y prueba (60%, 28%, 12%)

    if not full:
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.4, random_state=42)
        val_data, test_data, val_targets, test_targets = train_test_split(test_data, test_targets, test_size=0.3, random_state=42)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        val_data = torch.tensor(val_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        val_targets = torch.tensor(val_targets, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)

        return (train_data.cuda(), train_targets.cuda()), \
            (val_data.cuda(), val_targets.cuda()), \
            (test_data.cuda(), test_targets.cuda())

    test_data = torch.tensor(data, dtype=torch.float32)
    test_targets = torch.tensor(targets, dtype=torch.float32)

    return test_data.cuda(), test_targets.cuda()


def load_csv(dataset, inverted):
    df = pd.read_csv(dataset[0])

    data_columns = df.columns[:dataset[1]]
    target_columns = df.columns[dataset[1] + 1:-1]

    if inverted:
        data_columns, target_columns = target_columns, data_columns

    targets = df[target_columns].values
    data = df[data_columns].values
    return data, targets


def normalize(v):
    # Using mean and std deviation from 100K set
    return (v - 0.007696894933845342) / 0.03540097691806109, 0.03540097691806109


def normalize_by_column(v):
    # Using mean and std deviation from 100K set
    return (v - [4.99999997, 2.6, 5.00000004]) / [1.15470053, 1.38564068, 2.30940101], [1.15470053, 1.38564068, 2.30940101]
