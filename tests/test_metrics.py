import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
import pytest

from metrics import WRMSSE, aggregation_matrix


def make_data(n_days=10, h=3):
    ids = pd.DataFrame({
        'item_id': ['A', 'B', 'A', 'B'],
        'dept_id': ['D1', 'D1', 'D1', 'D1'],
        'cat_id': ['C', 'C', 'C', 'C'],
        'store_id': ['S1', 'S1', 'S2', 'S2'],
        'state_id': ['CA', 'CA', 'TX', 'TX'],
    })
    rng = np.random.default_rng(0)
    sales = rng.poisson(3, size=(4, n_days))
    sales[3, :4] = 0  # สินค้าที่เริ่มขายทีหลัง
    train = pd.concat([ids, pd.DataFrame(sales, columns=[f'd_{i}' for i in range(1, n_days + 1)])], axis=1)
    calendar = pd.DataFrame({'d': [f'd_{i}' for i in range(1, n_days + 1)], 'wm_yr_wk': [1] * 7 + [2] * (n_days - 7)})
    prices = pd.DataFrame([
        (s, i, w, p) for s in ['S1', 'S2'] for (i, p) in [('A', 1.0), ('B', 5.0)] for w in [1, 2]
    ], columns=['store_id', 'item_id', 'wm_yr_wk', 'sell_price'])
    return WRMSSE(train, calendar, prices, h=h), sales


def test_aggregation_has_one_row_per_series_per_level():
    agg, level_of = aggregation_matrix(pd.DataFrame({
        'item_id': ['A', 'B', 'A', 'B'], 'dept_id': ['D'] * 4, 'cat_id': ['C'] * 4,
        'store_id': ['S1', 'S1', 'S2', 'S2'], 'state_id': ['CA', 'CA', 'TX', 'TX'],
    }))
    assert np.bincount(level_of)[1:].tolist() == [1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 4, 4]
    assert np.allclose(agg.sum(axis=1).A1[level_of == 1], 4)


def test_weights_sum_to_one_per_level():
    ev, _ = make_data()
    assert np.allclose(np.bincount(ev.level_of, weights=ev.weights)[1:], 1.0)


def test_perfect_forecast_scores_zero():
    ev, _ = make_data()
    y = np.ones((4, 3))
    assert ev.score(y, y) == pytest.approx(0.0)


def test_single_series_matches_rmsse_formula():
    ev, sales = make_data()
    y_true, y_pred = np.full((4, 3), 2.0), np.full((4, 3), 3.0)
    _, per_level = ev.score(y_true, y_pred, by_level=True)
    # ระดับ 12 คือสินค้า×สาขา: คำนวณ RMSSE ตรงจากสูตร แล้วถ่วงน้ำหนัก
    expected = 0.0
    w12 = ev.weights[ev.level_of == 12]
    for i in range(4):
        s = sales[i, np.argmax(sales[i] != 0):]
        scale = np.mean(np.diff(s) ** 2)
        expected += w12[i] * np.sqrt(1.0 / scale)
    assert per_level[12] == pytest.approx(expected)
