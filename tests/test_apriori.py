import pandas as pd
import pytest

from utils.apriori_analysis import mine, transactions


def test_apriori_exact_support_confidence():
    table = pd.DataFrame({"A": [1, 1, 1, 0], "B": [1, 1, 0, 0]}, dtype=bool)
    frequent, rules, _ = mine(table, .5, .5, 1.)
    pair = frequent.loc[frequent.itemsets == frozenset({"A", "B"})].iloc[0]
    assert pair.support == .5
    a_to_b = rules.loc[rules.antecedents.eq("{A}")].iloc[0]
    assert a_to_b.confidence == pytest.approx(2 / 3)
    assert a_to_b.lift == pytest.approx(4 / 3)
    assert a_to_b.leverage == pytest.approx(.125)


def test_real_transactions_and_empty_rules(basketball):
    table, bounds = transactions(basketball[-1])
    assert table.sum(axis=1).eq(6).all()
    assert len(bounds) == 5
    frequent, rules, _ = mine(table, .95, .95, 1.)
    assert frequent.empty and rules.empty
