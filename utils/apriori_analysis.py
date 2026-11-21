from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

MINING_FEATURES = ["magnitude_mean", "magnitude_std", "x_rms", "y_rms", "z_rms"]
RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift",
                "leverage", "conviction", "rule"]


def transactions(windows):
    items = {"Activity: " + label: windows.activity.eq(label)
             for label in sorted(windows.activity.unique())}
    boundaries = []
    for name in MINING_FEATURES:
        low, high = windows[name].quantile([1/3, 2/3])
        bins = np.select([windows[name] <= low, windows[name] <= high],
                         ["low", "mid"], default="high")
        boundaries.append((name, low, high))
        for label in ("low", "mid", "high"):
            items[f"{name}: {label}"] = bins == label
    table = pd.DataFrame(items, index=windows.index)
    return table.loc[:, table.any()], pd.DataFrame(boundaries, columns=["Feature", "Low to", "Mid to"])


def label(items):
    return "{" + ", ".join(sorted(items)) + "}"


def format_rules(rules):
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    result = rules.copy()
    result["antecedents"] = result.antecedents.map(label)
    result["consequents"] = result.consequents.map(label)
    result["rule"] = "IF " + result.antecedents + " THEN " + result.consequents
    return result[RULE_COLUMNS].sort_values(["lift", "confidence"], ascending=False)


def mine(table, support, confidence, lift):
    start = perf_counter()
    itemsets = apriori(table, min_support=support, use_colnames=True, low_memory=True)
    rules = pd.DataFrame()
    if any(itemsets.itemsets.map(len) > 1):
        rules = association_rules(itemsets, metric="confidence", min_threshold=confidence)
        rules = rules.loc[rules.lift >= lift]
    return itemsets, format_rules(rules), perf_counter() - start
