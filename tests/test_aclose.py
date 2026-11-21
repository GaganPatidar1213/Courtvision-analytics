from itertools import combinations

import numpy as np
import pandas as pd

from utils.aclose import mine_closed
from utils.apriori_analysis import mine, transactions


def test_known_closures_and_generators():
    table = pd.DataFrame({"A": [1, 1, 1, 0, 0], "B": [1, 1, 0, 0, 0],
                          "C": [1, 1, 0, 0, 1]}, dtype=bool)
    closed, mapping, approximate, exact, _ = mine_closed(table, .4, .5, 1)
    expected = {frozenset(): 1., frozenset("A"): .6, frozenset("C"): .6, frozenset("ABC"): .4}
    assert dict(zip(closed.itemsets, closed.support)) == expected
    assert set(mapping.loc[mapping.closure.eq(frozenset("ABC")), "generator"]) == {
        frozenset("B"), frozenset("AC"),
    }
    assert exact.confidence.eq(1).all() and len(exact) == 2
    assert len(approximate) == 2
    assert np.allclose(approximate.confidence, 2 / 3)


def test_universal_item_and_brute_force():
    rows = [[1, 1, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1]]
    table = pd.DataFrame(rows, columns=list("ABCU"), dtype=bool)
    closed, mapping, _, _, _ = mine_closed(table, .2, 0, 0)
    supports = {}
    for size in range(len(table.columns) + 1):
        for items in combinations(table.columns, size):
            supports[frozenset(items)] = table[list(items)].all(axis=1).mean()
    expected = {a for a, value in supports.items() if value >= .2
                and not any(a < b and value == other for b, other in supports.items())}
    assert set(closed.itemsets) == expected
    assert mapping.loc[mapping.generator.eq(frozenset()), "closure"].iat[0] == frozenset("U")
    assert all(not any(b < a and value == supports[b] for b in supports)
               for a, _, value in mapping.itertuples(index=False, name=None))


def test_real_closed_support_matches_apriori(basketball):
    table, _ = transactions(basketball[-1])
    frequent, _, _ = mine(table, .15, .55, 1.)
    closed, _, _, _, _ = mine_closed(table, .15, .55, 1.)
    supports = dict(zip(frequent.itemsets, frequent.support))
    expected = {a for a, s in supports.items() if not any(a < b and s == other for b, other in supports.items())}
    assert {a for a in closed.itemsets if a} == expected
