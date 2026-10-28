import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import FIRST_USABLE_DAY, HORIZON, ID_COLS, LAGS, WINDOWS, FeatureBuilder

SALES_FEATURES = [f'lag_{lag}' for lag in LAGS] + [f'rmean_{w}' for w in WINDOWS]


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


START = '2020-01-30'  # เลือกวันที่ให้ข้ามเดือน เพื่อให้ month และ mday เปลี่ยนในช่วงทดสอบ


def make_panel(sales, price=None, snap=None, events=None):
    """
    sales, price, snap: array (สินค้า×สาขา, วัน) ทุกแถวเป็นสินค้าคนละตัวใน store/state เดียวกัน
    events: {วัน d: (event_name, event_type)}
    """
    sales = np.asarray(sales, float)
    n_series, n_days = sales.shape
    price = np.full(sales.shape, 5.) if price is None else np.asarray(price, float)
    snap = np.zeros(sales.shape) if snap is None else np.asarray(snap, float)
    data = pd.DataFrame({
        'item_id': np.repeat([f'I{i}' for i in range(n_series)], n_days),
        'dept_id': 'D', 'cat_id': 'C', 'store_id': 'S', 'state_id': 'CA',
        'd': np.tile(np.arange(1, n_days + 1), n_series),
        'sales': sales.ravel(), 'sell_price': price.ravel(), 'snap': snap.ravel(),
    })
    for col in ID_COLS:
        data[col] = data[col].astype('category')
    dates = pd.date_range(START, periods=n_days)
    event_name, event_type = [None] * n_days, [None] * n_days
    for d, (name, kind) in (events or {}).items():
        event_name[d - 1], event_type[d - 1] = name, kind
    calendar = pd.DataFrame({
        'date': dates, 'wday': dates.day_of_week + 1, 'month': dates.month,
        'event_name_1': event_name, 'event_type_1': event_type,
    })
    return FeatureBuilder(data, calendar)


def random_sales(n_series=3, n_days=120, seed=0):
    return np.random.default_rng(seed).integers(0, 50, size=(n_series, n_days)).astype(float)


def test_lag_features_look_back_the_right_number_of_days():
    sales = random_sales()
    X, y = make_panel(sales).build(90, 110)
    days = np.arange(90, 111)
    for lag in LAGS:
        # วัน d_k อยู่ที่ column k-1
        np.testing.assert_array_equal(X[f'lag_{lag}'], sales[:, days - lag - 1].ravel())
    np.testing.assert_array_equal(y, sales[:, days - 1].ravel())


def test_rolling_means_cover_the_window_ending_horizon_days_before():
    sales = random_sales()
    X, _ = make_panel(sales).build(90, 110)
    for w in WINDOWS:
        expected = pd.DataFrame(sales.T).rolling(w).mean().shift(HORIZON).to_numpy().T[:, 89:110]
        np.testing.assert_allclose(X[f'rmean_{w}'].to_numpy().reshape(3, -1), expected, rtol=1e-5)


def test_first_usable_day_is_the_earliest_day_with_full_history():
    sales = random_sales(n_series=2, n_days=100)
    builder = make_panel(sales)
    # ถ้าไม่มี guard rmean_56 จะอ่าน cumsum[-1] (ยอดรวมทั้งหมด) แบบเงียบๆ
    with pytest.raises(ValueError):
        builder.build(FIRST_USABLE_DAY - 1, 90)
    X, _ = builder.build(FIRST_USABLE_DAY, FIRST_USABLE_DAY)
    assert X[SALES_FEATURES].notna().all().all()
    np.testing.assert_allclose(X['rmean_56'], sales[:, :max(WINDOWS)].mean(axis=1), rtol=1e-5)


def test_sales_features_ignore_sales_inside_the_forecast_horizon():
    base = random_sales(n_days=120)
    changed = base.copy()
    changed[:, 61:] += 100  # เปลี่ยนยอดขายตั้งแต่ d_62 เป็นต้นไป
    before, _ = make_panel(base).build(84, 89)
    after, _ = make_panel(changed).build(84, 89)
    np.testing.assert_array_equal(before[SALES_FEATURES], after[SALES_FEATURES])
    # d_90 เริ่มมองเห็นยอด d_62 ถ้า feature ไม่เปลี่ยน แปลว่าเทสต์ข้างบนไม่ได้พิสูจน์อะไร
    next_before, _ = make_panel(base).build(90, 90)
    next_after, _ = make_panel(changed).build(90, 90)
    assert not np.array_equal(next_before['lag_28'], next_after['lag_28'])


def test_price_features_are_nan_until_the_item_launches():
    price = np.full((1, 120), np.nan)
    price[0, 89:] = 10.  # วางขายตั้งแต่ d_90
    X, _ = make_panel(random_sales(n_series=1), price=price).build(84, 100)
    days = np.arange(84, 101)
    before_launch = days < 90
    assert X['sell_price'][before_launch].isna().all()
    assert X['price_rel_regular'][before_launch].isna().all()
    assert (X['price_rel_regular'][~before_launch] == 1).all()
    # price_change_7d ต้องมีราคาเมื่อ 7 วันก่อนด้วย จึงเริ่มมีค่าที่ d_97
    change = X['price_change_7d'].to_numpy()
    assert np.isnan(change[days < 97]).all()
    assert (change[days >= 97] == 0).all()


def test_price_features_track_discounts_and_price_increases():
    price = np.full((1, 120), 10.)
    price[0, 89:95] = 8.   # ลดราคา d_90–d_95
    price[0, 99:] = 12.    # ขึ้นราคาตั้งแต่ d_100
    X, _ = make_panel(random_sales(n_series=1), price=price).build(84, 105)
    days = np.arange(84, 106)
    rel, change = X['price_rel_regular'].to_numpy(), X['price_change_7d'].to_numpy()

    # ตอนลดราคา ราคาปกติยังเป็น 10 (ถ้าอนาคตรั่วเข้ามาจะได้ 8/12)
    np.testing.assert_allclose(rel[days == 92], 0.8)
    # หลังขึ้นราคา ราคาปกติใหม่คือ 12
    np.testing.assert_allclose(rel[days == 100], 1.0)
    np.testing.assert_allclose(change[days == 90], 8 / 10 - 1)    # เริ่มลดราคา
    np.testing.assert_allclose(change[days == 96], 0.0)           # กลับราคาเดิม เทียบ d_89 (10)
    np.testing.assert_allclose(change[days == 100], 12 / 8 - 1)   # ขึ้นราคา เทียบ d_93 (8)


def test_rows_are_ordered_by_series_then_day_with_matching_ids_and_calendar():
    sales = random_sales(n_series=2, n_days=120)
    snap = np.zeros((2, 120))
    snap[1, 90] = 1  # สินค้าตัวที่ 2 วัน d_91 เป็นวัน SNAP
    X, y = make_panel(sales, snap=snap, events={90: ('Easter', 'Religious')}).build(88, 92)
    dates = pd.date_range(START, periods=120)[87:92]

    assert len(X) == len(y) == 10
    np.testing.assert_array_equal(y, sales[:, 87:92].ravel())
    assert X['item_id'].tolist() == [0] * 5 + [1] * 5
    assert X['mday'].tolist() == list(dates.day) * 2
    assert X['month'].tolist() == list(dates.month) * 2
    assert X['wday'].tolist() == list(dates.day_of_week + 1) * 2
    assert X['event_name_1'].tolist() == [0, 0, 1, 0, 0] * 2  # 0 = ไม่มี event
    assert X['event_type_1'].tolist() == [0, 0, 1, 0, 0] * 2
    assert X['snap'].tolist() == [0] * 5 + [0, 0, 0, 1, 0]
