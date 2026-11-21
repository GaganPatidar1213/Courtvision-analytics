import numpy as np
import pandas as pd
from scipy.stats import zscore

from utils.data_loader import CHANNELS, RAW_COLUMNS

SIGNALS = [*CHANNELS, "magnitude"]


def clean(raw):
    data = raw.reset_index(drop=True).copy()
    for name in RAW_COLUMNS:
        data[name] = pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data.loc[data.time_s < 0, "time_s"] = np.nan
    missing = data[RAW_COLUMNS].isna().sum()
    duplicate_recordings = data["duplicate_of"].notna() if "duplicate_of" in data else pd.Series(False, index=data.index)
    data = data.loc[~duplicate_recordings].copy()
    duplicates = data.duplicated(["recording", *RAW_COLUMNS])
    invalid_required = data[["time_s", *CHANNELS]].isna().any(axis=1)
    data = data.loc[~duplicates & ~invalid_required].copy()
    data = data.sort_values(["recording", "time_s"], kind="stable")
    collisions = data.duplicated(["recording", "time_s"], keep=False)
    # Conflicting packets with the same timestamp cannot be reliably ordered.
    data = data.loc[~collisions].copy()
    in_g = data.participant.eq("X")
    # The mixed files have no participant code. Their resting scale is undocumented.
    data["acceleration_unit"] = np.where(data.activity.eq("mixed"), "unverified", "m/s²")
    data.loc[in_g, [*CHANNELS, "reported_magnitude"]] *= 9.80665
    data["magnitude"] = np.linalg.norm(data[CHANNELS], axis=1)
    grouped = data.groupby("recording", sort=False)
    filtered = grouped[CHANNELS].transform(
        lambda signal: signal.rolling(5, center=True, min_periods=1).median()
    )
    residual = data[CHANNELS] - filtered
    data["noise_residual"] = np.linalg.norm(residual, axis=1)
    residual_group = data.groupby("recording").noise_residual
    center = residual_group.transform("median")
    mad = (data.noise_residual - center).abs().groupby(data.recording).transform("median")
    data["noise_flag"] = data.noise_residual > center + 6 * 1.4826 * mad
    data[[f"{name}_filtered" for name in CHANNELS]] = filtered
    data["magnitude_filtered"] = np.linalg.norm(filtered, axis=1)
    audit = {
        "input rows": len(raw), "clean rows": len(data),
        "duplicate recording rows": int(duplicate_recordings.sum()),
        "exact duplicates": int(duplicates.sum()),
        "invalid required rows": int((~duplicates & invalid_required).sum()),
        "conflicting timestamp rows": int(collisions.sum()),
        "noise flags": int(data.noise_flag.sum()),
    }
    return data.reset_index(drop=True), audit, missing


def outliers(frame):
    signals = frame[SIGNALS]
    lower, upper = signals.quantile(0.25), signals.quantile(0.75)
    iqr = upper - lower
    return pd.DataFrame({
        "IQR flags": ((signals < lower - 1.5 * iqr) | (signals > upper + 1.5 * iqr)).sum(),
        "Z-score flags": (np.abs(zscore(signals, nan_policy="omit")) > 3).sum(axis=0),
    })
