from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text


def evaluate(windows, features, depth=5, leaf=4, selected=12):
    target = windows.activity
    groups = windows.recording
    features = [feature for feature in features if windows[feature].notna().all()]
    if not features:
        raise ValueError("No complete features are available.")
    values = windows[features]
    classes = sorted(target.unique())
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=587)
    for train, test in splitter.split(values, target, groups):
        if set(target.iloc[test]) == set(target.iloc[train]) == set(classes):
            break
    else:
        raise ValueError("No recording-disjoint split contains all classes.")
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    model = make_pipeline(
        SelectKBest(f_classif, k=min(selected, len(features))),
        DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                               class_weight="balanced", random_state=587),
    )
    model.fit(values.iloc[train], target.iloc[train])
    predictions = model.predict(values.iloc[test])
    majority = target.iloc[train].value_counts().idxmax()
    tree = model[-1]
    names = list(model[0].get_feature_names_out())
    rules = []

    def walk(node, conditions):
        left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
        if left == right:
            label = tree.classes_[tree.tree_.value[node][0].argmax()]
            rules.append(f"IF {' AND '.join(conditions) or 'TRUE'} THEN {label}")
            return
        name = names[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        walk(left, [*conditions, f"{name} <= {threshold:.12g}"])
        walk(right, [*conditions, f"{name} > {threshold:.12g}"])

    walk(0, [])
    return {
        "model": model, "names": names, "classes": classes, "rules": rules,
        "structure": export_text(tree, feature_names=names, decimals=5),
        "accuracy": accuracy_score(target.iloc[test], predictions),
        "majority_class": majority,
        "baseline_accuracy": float(target.iloc[test].eq(majority).mean()),
        "report": classification_report(target.iloc[test], predictions,
                                         labels=classes, output_dict=True, zero_division=0),
        "matrix": confusion_matrix(target.iloc[test], predictions, labels=classes),
        "test": test, "train": train, "predictions": predictions,
        "train_recordings": groups.iloc[train].nunique(),
        "test_recordings": groups.iloc[test].nunique(),
    }
