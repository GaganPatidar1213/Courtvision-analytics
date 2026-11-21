import numpy as np
import pandas as pd

from utils.feature_engineering import feature_names, transform_for_export
from utils.preprocessing import clean, outliers
from utils.statistics import describe


def test_original_archive_and_window_metadata(basketball):
    raw, processed, audit, missing, windows = basketball
    assert len(raw) == 63231
    assert raw.recording.nunique() == 81
    assert raw.loc[raw.activity.ne("mixed")].recording.nunique() == 79
    assert len(windows) > 300 and windows.samples.min() >= 75
    assert "mixed" not in windows.activity.unique()
    assert len(feature_names(windows)) == 24
    assert not processed.duplicated(["recording", "time_s"]).any()
    assert audit["input rows"] - audit["clean rows"] == sum(
        audit[name] for name in ["exact duplicates", "duplicate recording rows",
                                 "invalid required rows", "conflicting timestamp rows"]
    )
    assert processed.loc[processed.activity.eq("mixed")].recording.nunique() == 1
    original = raw.loc[raw.recording.eq("X_hold3.txt")].iloc[0]
    corrected = processed.loc[processed.recording.eq("X_hold3.txt")].iloc[0]
    assert np.isclose(corrected.x, original.x * 9.80665)
    assert np.isclose(corrected.magnitude, np.linalg.norm(corrected[["x", "y", "z"]].astype(float)))


def test_invalid_duplicate_and_timestamp_collisions(basketball):
    raw = basketball[0]
    trial = raw.loc[raw.recording.eq("D_hold1.txt")].iloc[:8].copy()
    extra = trial.iloc[[0]].copy()
    collision = trial.iloc[[1]].assign(x=1000)
    invalid = trial.iloc[[2]].assign(x=np.nan)
    batch = pd.concat([trial, extra, collision, invalid], ignore_index=True)
    processed, audit, missing = clean(batch)
    assert len(processed) == 7
    assert not processed.time_s.eq(collision.time_s.iat[0]).any()
    assert processed.time_s.eq(invalid.time_s.iat[0]).sum() == 1
    assert audit["exact duplicates"] == 1
    assert audit["conflicting timestamp rows"] == 2
    assert audit["invalid required rows"] == missing.x == 1


def test_statistics_outliers_and_scaling(basketball):
    _, processed, _, _, windows = basketball
    summary = describe(pd.DataFrame({"x": [1, 2, 2, 5, np.nan]})).loc["x"]
    assert summary["count"] == 4
    assert summary["mode"] == summary["median"] == 2
    assert summary["range"] == 4 and summary["missing %"] == 20
    assert outliers(processed)["IQR flags"].sum() > 0
    features = feature_names(windows)
    scaled = transform_for_export(windows, "Standardized")
    assert np.allclose(scaled[features].mean(), 0)
    scaled = transform_for_export(windows, "Min-max")
    assert np.allclose(scaled[features].min(), 0)
    assert np.allclose(scaled[features].max(), 1)
