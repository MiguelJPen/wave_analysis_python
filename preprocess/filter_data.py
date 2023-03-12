import pandas as pd
import numpy as np

class Dataset:
    lhs_10K = "preprocess/results_lhs_10K.csv"
    lhs_100K = "preprocess/results_lhs_100K.csv"
    equispaced = "preprocess/results.csv"


def load_csv(csv_filename):
    df = pd.read_csv(csv_filename)

    target_columns = df.columns[:3]
    data_columns = df.columns[4:-1]

    targets = df[target_columns].values
    data = df[data_columns].values
    return data, targets


def normalize(v):
    return (v - np.mean(v)) / np.std(v), np.mean(v), np.std(v)