"""
ตรวจ build_long() ใน src/data.py กับข้อมูลดิบจริง (ข้ามอัตโนมัติถ้าไม่มีข้อมูล, ตั้ง M5_RAW_DIR ถ้าไฟล์ไม่ได้อยู่ใน data/raw)
ค่าเริ่มต้นตรวจทั้ง 30,490 อนุกรม × 1,969 วัน ≈ 60 ล้านแถว ใช้ RAM สูงสุดราว 3.8 GB
ถ้า RAM ไม่พอ ตั้ง M5_SERIES_STRIDE=7 เพื่อตรวจทุกอนุกรมที่ 7 (ครอบคลุมทุกสาขาและทุกรัฐ) ใช้ราว 2.1 GB
(ลดได้ไม่ถึง 7 เท่า เพราะยังต้องโหลดไฟล์ CSV ทั้งไฟล์)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import ID_COLS, build_long

STATES = ['CA', 'TX', 'WI']
STRIDE = int(os.environ.get('M5_SERIES_STRIDE', 1))


@pytest.fixture(scope='module')
def m5(raw_data):
    """(sales, calendar, prices) ที่ใช้ตรวจ: ทั้งหมด หรือเฉพาะทุกอนุกรมที่ STRIDE"""
    sales, calendar, prices = raw_data
    if STRIDE > 1:
        sales = sales.iloc[::STRIDE].reset_index(drop=True)
        prices = prices.merge(sales[['store_id', 'item_id']], on=['store_id', 'item_id'])
    return sales, calendar, prices


@pytest.fixture(scope='module')
def long_df(m5):
    return build_long(*m5)


def as_grid(long_df, col, n_series, n_days):
    return long_df[col].to_numpy().reshape(n_series, n_days)


def test_long_table_has_one_row_per_series_per_day_in_series_order(m5, long_df):
    sales, calendar, _ = m5
    n_series, n_days = len(sales), len(calendar)
    assert n_days == 1_969
    if STRIDE == 1:
        assert n_series == 30_490
    assert len(long_df) == n_series * n_days

    assert (as_grid(long_df, 'd', n_series, n_days) == np.arange(1, n_days + 1)).all()
    for col in ID_COLS:
        codes = long_df[col].cat.codes.to_numpy().reshape(n_series, n_days)
        assert (codes == codes[:, :1]).all(), f'{col} changes within a series'
        assert (long_df[col].cat.categories[codes[:, 0]].to_numpy() == sales[col].to_numpy()).all()


def test_sales_match_wide_file_and_are_nan_only_after_the_last_known_day(m5, long_df):
    sales, calendar, _ = m5
    d_cols = [c for c in sales.columns if c.startswith('d_')]
    y = as_grid(long_df, 'sales', len(sales), len(calendar))

    assert len(d_cols) == 1_941
    np.testing.assert_array_equal(y[:, :len(d_cols)], sales[d_cols].to_numpy(np.float32))
    assert np.isnan(y[:, len(d_cols):]).all()


def test_snap_flag_is_the_calendar_flag_of_the_stores_state(m5, long_df):
    sales, calendar, _ = m5
    snap = as_grid(long_df, 'snap', len(sales), len(calendar))
    for state in STATES:
        expected = calendar[f'snap_{state}'].to_numpy()
        rows = (sales['state_id'] == state).to_numpy()
        assert expected.any() and rows.any()
        assert (snap[rows] == expected).all()


def test_prices_match_sell_prices_and_are_nan_before_launch(m5, long_df):
    sales, calendar, prices = m5
    n_series, n_days = len(sales), len(calendar)
    price = as_grid(long_df, 'sell_price', n_series, n_days)

    # ทุกแถวของ sell_prices ต้องกลายเป็นราคาในทุกวันของสัปดาห์นั้น และไม่มีราคาที่อื่น
    days_per_week = calendar['wm_yr_wk'].value_counts()
    assert (~np.isnan(price)).sum() == int(prices['wm_yr_wk'].map(days_per_week).fillna(0).sum())

    # สุ่มตรวจรายเซลล์เทียบกับ sell_prices (ต้องมีทั้งวันที่มีและไม่มีราคา)
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, n_series, 50_000), rng.integers(0, n_days, 50_000)
    keys = pd.MultiIndex.from_arrays([
        sales['store_id'].to_numpy()[rows], sales['item_id'].to_numpy()[rows], calendar['wm_yr_wk'].to_numpy()[cols],
    ])
    expected = prices.set_index(['store_id', 'item_id', 'wm_yr_wk'])['sell_price'].reindex(keys).to_numpy(np.float32)
    assert np.isnan(expected).any() and (~np.isnan(expected)).any()
    np.testing.assert_array_equal(price[rows, cols], expected)


def test_cached_parquet_is_identical_to_a_fresh_build(raw_dir, raw_data, long_df):
    cached = raw_dir.parent / 'processed' / 'm5_long.parquet'
    if not cached.exists():
        pytest.skip(f'{cached} not found')
    n_all, n_days = len(raw_data[0]), len(raw_data[1])
    rows = (np.arange(0, n_all, STRIDE)[:, None] * n_days + np.arange(n_days)).ravel()
    for col in long_df.columns:  # อ่านทีละคอลัมน์เพื่อประหยัด RAM
        old = pd.read_parquet(cached, columns=[col])[col].iloc[rows]
        new = long_df[col]
        if isinstance(new.dtype, pd.CategoricalDtype) and STRIDE == 1:
            assert list(old.cat.categories) == list(new.cat.categories), col
            np.testing.assert_array_equal(old.cat.codes.to_numpy(), new.cat.codes.to_numpy(), err_msg=col)
        elif isinstance(new.dtype, pd.CategoricalDtype):  # category ถูกนับเลขใหม่ตามชุดย่อย เทียบค่าที่ถอดรหัสแล้ว
            assert (old.astype(object).to_numpy() == new.astype(object).to_numpy()).all(), col
        else:
            np.testing.assert_array_equal(old.to_numpy(), new.to_numpy(), err_msg=col)
