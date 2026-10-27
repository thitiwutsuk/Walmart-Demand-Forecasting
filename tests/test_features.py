import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import FeatureBuilder


def make_builder(later_price):
    n = 100
    data = pd.DataFrame({
        'item_id': ['A'] * n, 'dept_id': ['D'] * n, 'cat_id': ['C'] * n,
        'store_id': ['S'] * n, 'state_id': ['CA'] * n,
        'd': np.arange(1, n + 1), 'sales': np.ones(n),
        'sell_price': [np.nan] + [10.] * 89 + [later_price] * 10,
        'snap': np.zeros(n),
    })
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        data[col] = data[col].astype('category')
    calendar = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n),
        'wday': [1] * n, 'month': [1] * n,
        'event_name_1': [None] * n, 'event_type_1': [None] * n,
    })
    return FeatureBuilder(data, calendar)


def test_future_price_cannot_change_past_regular_price_feature():
    before, _ = make_builder(10).build(84, 89)
    after, _ = make_builder(20).build(84, 89)
    np.testing.assert_array_equal(before['price_rel_regular'], after['price_rel_regular'])
    assert (after['price_rel_regular'] == 1).all()
