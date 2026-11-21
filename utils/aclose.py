from collections import defaultdict
from itertools import combinations
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from utils.apriori_analysis import format_rules


def mine_closed(table, support, confidence, lift):
    """Find frequent minimal generators levelwise, then compute their closures."""
    if table.empty or not 0 < support <= 1:
        raise ValueError("A nonempty transaction table and valid support are required.")
    started = perf_counter()
    size = len(table)
    minimum = ceil(np.nextafter(support * size, -np.inf))
    columns = sorted(table.columns)
    masks = {name: table[name].to_numpy(dtype=bool) for name in columns}
    all_rows = np.ones(size, dtype=bool)
    generators = {}
    closures = defaultdict(list)

    def register(items, mask):
        closure = frozenset(name for name in columns if masks[name][mask].all())
        count = int(mask.sum())
        generators[items] = count
        closures[closure].append((items, count))

    register(frozenset(), all_rows)
    level = {frozenset({column}) for column in columns}
    while level:
        retained = set()
        for items in level:
            mask = np.logical_and.reduce([masks[name] for name in items])
            count = int(mask.sum())
            if count < minimum or any(generators[items - {item}] == count for item in items):
                continue
            register(items, mask)
            retained.add(items)
        unions = {a | b for a, b in combinations(retained, 2) if len(a - b) == 1}
        level = {items for items in unions if all(items - {item} in retained for item in items)}

    counts = {closed: entries[0][1] for closed, entries in closures.items()}
    for closed, count in counts.items():
        assert all(int(np.logical_and.reduce([masks[item], *[masks[x] for x in closed]]).sum()) < count
                   for item in set(columns) - closed)
    mapping = pd.DataFrame(
        [(gen, closed, count / size) for closed, entries in closures.items()
         for gen, count in entries],
        columns=["generator", "closure", "support"],
    )
    closed_table = pd.DataFrame(
        [(closed, count / size) for closed, count in counts.items()],
        columns=["itemsets", "support"],
    ).sort_values("support", ascending=False)
    approximate = []
    exact = []

    def rule(left, right, union_count, left_count):
        right_count = int(np.logical_and.reduce([masks[name] for name in right]).sum())
        conf = union_count / left_count
        base = right_count / size
        ratio = conf / base
        if conf >= confidence and ratio >= lift:
            return dict(antecedents=left, consequents=right, support=union_count / size,
                        confidence=conf, lift=ratio, leverage=union_count / size - left_count / size * base,
                        conviction=(1 - base) / (1 - conf) if conf < 1 else float("inf"))
        return None

    for upper, count in counts.items():
        for generator, _ in closures[upper]:
            if generator and upper - generator:
                result = rule(generator, upper - generator, count, count)
                if result:
                    exact.append(result)
        for lower, lower_count in counts.items():
            if lower and lower < upper and count < lower_count:
                result = rule(lower, upper - lower, count, lower_count)
                if result:
                    approximate.append(result)
    return closed_table, mapping, format_rules(pd.DataFrame(approximate)), format_rules(pd.DataFrame(exact)), perf_counter() - started
