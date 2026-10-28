"""
เทียบ WRMSSE ใน src/metrics.py กับ implementation แยกอีกชุดที่เขียนตามนิยามของ Kaggle M5 โดยตรง
(groupby/merge ของ pandas แทน sparse matrix) บนข้อมูลจริงและรอบ backtest จริง
ข้ามอัตโนมัติถ้าไม่มีข้อมูลดิบ ตั้ง M5_RAW_DIR ถ้าไฟล์ไม่ได้อยู่ใน data/raw
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import WRMSSE

# 12 ระดับตามกติกา M5 เขียนซ้ำที่นี่ ไม่ import จาก metrics เพื่อให้เป็นการตรวจอิสระ
LEVEL_KEYS = [
    [], ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
    ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'], ['store_id', 'dept_id'],
    ['item_id'], ['item_id', 'state_id'], ['item_id', 'store_id'],
]
ID_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
FOLD_END, H = 1913, 28


def reference_wrmsse(train, calendar, prices, y_true, preds, h=H):
    """คืน {ชื่อ forecast: per-level Series (12 ระดับ)} คำนวณทีละระดับด้วย pandas groupby"""
    d_cols = [c for c in train.columns if c.startswith('d_')]

    # น้ำหนัก: ยอดขายเป็นเงิน h วันสุดท้ายของ train (ยอดขาย × ราคาของสัปดาห์นั้น, ไม่มีราคา = 0)
    recent = train[ID_COLS + d_cols[-h:]].melt(id_vars=ID_COLS, var_name='d', value_name='units')
    recent = recent.merge(calendar[['d', 'wm_yr_wk']], on='d').merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    recent['dollars'] = recent['units'] * recent['sell_price'].fillna(0)
    dollars = recent.groupby(ID_COLS, sort=False)['dollars'].sum().reindex(pd.MultiIndex.from_frame(train[ID_COLS]))
    dollars = pd.Series(dollars.to_numpy(), index=train.index)

    results = {name: [] for name in preds}
    for keys in LEVEL_KEYS:
        by = [train[k] for k in keys] if keys else np.zeros(len(train), int)
        history = train[d_cols].groupby(by).sum().to_numpy(float)
        w = dollars.groupby(by).sum().to_numpy()
        w = w / w.sum()

        # scale = MSE ของ naive 1 วัน นับจากวันแรกที่อนุกรมนั้นมียอดขาย
        scale = np.array([np.mean(np.diff(row[np.argmax(row != 0):]) ** 2) if row.any() else 0.0 for row in history])
        w = np.where(scale > 0, w, 0.0)
        scale = np.where(scale > 0, scale, 1.0)

        actual = pd.DataFrame(y_true).groupby(by).sum().to_numpy(float)
        for name, pred in preds.items():
            forecast = pd.DataFrame(pred).groupby(by).sum().to_numpy(float)
            rmsse = np.sqrt(((actual - forecast) ** 2).mean(axis=1) / scale)
            results[name].append((w * rmsse).sum())
    return {name: pd.Series(v, index=range(1, 13)) for name, v in results.items()}


def test_reference_agrees_with_metrics_on_a_tiny_example():
    ids = pd.DataFrame({
        'item_id': ['A', 'B', 'A', 'B'], 'dept_id': ['D'] * 4, 'cat_id': ['C'] * 4,
        'store_id': ['S1', 'S1', 'S2', 'S2'], 'state_id': ['CA', 'CA', 'TX', 'TX'],
    })
    sales = np.random.default_rng(0).poisson(3, size=(4, 12))
    sales[3, :4] = 0
    train = pd.concat([ids, pd.DataFrame(sales, columns=[f'd_{i}' for i in range(1, 13)])], axis=1)
    calendar = pd.DataFrame({'d': [f'd_{i}' for i in range(1, 13)], 'wm_yr_wk': [1] * 7 + [2] * 5})
    prices = pd.DataFrame(
        [(s, i, w, p) for s in ['S1', 'S2'] for i, p in [('A', 1.0), ('B', 5.0)] for w in [1, 2]],
        columns=['store_id', 'item_id', 'wm_yr_wk', 'sell_price'],
    )
    y_true, y_pred = np.random.default_rng(1).poisson(3, size=(4, 3)), np.random.default_rng(2).poisson(3, size=(4, 3))
    expected = reference_wrmsse(train, calendar, prices, y_true, {'f': y_pred}, h=3)['f']
    _, actual = WRMSSE(train, calendar, prices, h=3).score(y_true, y_pred, by_level=True)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_wrmsse_matches_reference_on_real_backtest_fold(raw_data):
    sales, calendar, prices = raw_data
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    train = sales[ID_COLS + d_cols[:FOLD_END]]
    y_true = sales[d_cols[FOLD_END:FOLD_END + H]].to_numpy(float)

    rng = np.random.default_rng(0)
    last_week = train[d_cols[FOLD_END - 7:FOLD_END]].to_numpy(float)
    mean_28 = train[d_cols[FOLD_END - 28:FOLD_END]].to_numpy(float).mean(axis=1, keepdims=True)
    preds = {
        'seasonal_naive': np.tile(last_week, H // 7),
        'noisy_mean': np.repeat(mean_28, H, axis=1) * rng.lognormal(0, 0.3, size=(len(train), H)),
        'perfect': y_true,
    }

    scorer = WRMSSE(train, calendar, prices)
    assert np.bincount(scorer.level_of)[1:].sum() == 42_840
    expected = reference_wrmsse(train, calendar, prices, y_true, preds)
    for name, pred in preds.items():
        total, per_level = scorer.score(y_true, pred, by_level=True)
        np.testing.assert_allclose(per_level.to_numpy(), expected[name].to_numpy(), rtol=1e-9, err_msg=name)
        assert total == pytest.approx(expected[name].mean(), rel=1e-9)
    assert scorer.score(y_true, y_true) == pytest.approx(0.0)
