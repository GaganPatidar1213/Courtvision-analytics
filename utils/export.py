"""Generate the default local data, model, and report deliverables."""

from pathlib import Path

import joblib
import pandas as pd

from utils.aclose import mine_closed
from utils.apriori_analysis import label, mine, transactions
from utils.data_loader import load_archive
from utils.decision_tree import evaluate
from utils.feature_engineering import build_windows, feature_names, transform_for_export
from utils.preprocessing import clean
from utils.reporting import create_report
from utils.statistics import describe
from utils.visualization import MEASURES


def export_defaults(root):
    raw = load_archive()
    processed, audit, missing = clean(raw)
    data = processed.loc[processed.activity.ne("mixed")]
    windows = build_windows(processed)
    transactions_table, _ = transactions(windows)
    apriori = mine(transactions_table, .15, .55, 1.)
    closed = mine_closed(transactions_table, .15, .55, 1.)
    model = evaluate(windows, feature_names(windows))
    for folder in ("data/processed", "models", "reports/tables", "output/pdf"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    processed.to_csv(root / "data/processed/cleaned_samples.csv", index=False)
    transform_for_export(windows, "Original features").to_csv(
        root / "data/processed/activity_windows.csv", index=False
    )
    stats = describe(data[MEASURES]).rename_axis("sensor")
    stats.to_csv(root / "reports/tables/descriptive_statistics.csv")
    apriori[1].to_csv(root / "reports/tables/apriori_rules.csv", index=False)
    closed[0].assign(itemsets=closed[0].itemsets.map(label)).to_csv(
        root / "reports/tables/closed_itemsets.csv", index=False
    )
    closed[2].to_csv(root / "reports/tables/approximate_closed_rules.csv", index=False)
    closed[3].to_csv(root / "reports/tables/exact_implications.csv", index=False)
    predictions = windows.iloc[model["test"]][["recording", "window", "activity"]]
    predictions.assign(prediction=model["predictions"]).to_csv(
        root / "reports/tables/heldout_predictions.csv", index=False
    )
    report_args = (raw, data, windows, audit, missing, model, apriori, closed)
    for format_name, folder in (("HTML", "reports"), ("PDF", "output/pdf")):
        path = root / folder / f"basketball_report.{format_name.lower()}"
        path.write_bytes(create_report(format_name, *report_args))
    joblib.dump(model["model"], root / "models/decision_tree.joblib")
    print(pd.Series(audit).to_string())
    print(f"Labeled windows: {len(windows)}")
    print(f"Held-out accuracy: {model['accuracy']:.4f}")
    print(f"Held-out macro F1: {model['report']['macro avg']['f1-score']:.4f}")
    print(f"Training majority baseline accuracy: {model['baseline_accuracy']:.4f}")
    print(f"Apriori: {len(apriori[0])} itemsets and {len(apriori[1])} rules")
    print(f"A-CLOSE: {closed[0].itemsets.map(bool).sum()} nonempty closed itemsets, "
          f"{len(closed[1])} minimal generators, {len(closed[2])} approximate rules, {len(closed[3])} exact implications")


if __name__ == "__main__":
    export_defaults(Path(__file__).resolve().parents[1])
