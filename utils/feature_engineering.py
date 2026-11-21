import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.preprocessing import SIGNALS

WINDOW_SECONDS = 1.0
MIN_SAMPLES = 75


def build_windows(clean, seconds=WINDOW_SECONDS, min_samples=MIN_SAMPLES):
    labeled = clean.loc[clean.activity.ne("mixed")].copy()
    labeled["window"] = np.floor(labeled.time_s / seconds).astype(int)
    rows = []
    for (recording, number), batch in labeled.groupby(["recording", "window"]):
        if len(batch) < min_samples or batch.time_s.max() - batch.time_s.min() < .75 * seconds:
            continue
        row = dict(recording=recording, window=int(number), activity=batch.activity.iloc[0],
                   participant=batch.participant.iloc[0], samples=len(batch),
                   start_s=number * seconds)
        for name in SIGNALS:
            values = batch[f"{name}_filtered"].to_numpy()
            squares = np.square(values)
            for metric, value in {
                "mean": values.mean(), "std": values.std(), "min": values.min(),
                "max": values.max(), "rms": np.sqrt(squares.mean()), "energy": squares.sum(),
            }.items():
                row[f"{name}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def feature_names(windows):
    return [name for name in windows if any(name.startswith(f"{signal}_") for signal in SIGNALS)]


def transform_for_export(windows, method):
    frame = windows.copy()
    features = feature_names(frame)
    scaler = {"Standardized": StandardScaler, "Min-max": MinMaxScaler}.get(method)
    if scaler:
        frame[features] = scaler().fit_transform(frame[features])
    return pd.concat([frame, pd.get_dummies(frame.activity, prefix="activity", dtype=int)], axis=1)
