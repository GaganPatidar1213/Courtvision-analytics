import numpy as np

from utils.decision_tree import evaluate
from utils.feature_engineering import feature_names


def test_tree_disjoint_split_and_predictions(basketball):
    windows = basketball[-1]
    features = feature_names(windows)
    result = evaluate(windows, features)
    train = windows.iloc[result["train"]]
    test = windows.iloc[result["test"]]
    assert set(train.recording).isdisjoint(test.recording)
    assert set(train.activity) == set(test.activity)
    assert len(test) == result["matrix"].sum()
    assert np.array_equal(result["model"].predict(test[features]), result["predictions"])
    assert np.isclose(result["accuracy"], np.trace(result["matrix"]) / len(test))
    assert 0 < result["report"]["macro avg"]["f1-score"] <= 1
    assert all(line.startswith("IF ") and " THEN " in line for line in result["rules"])
    assert any("<=" in line for line in result["rules"])
