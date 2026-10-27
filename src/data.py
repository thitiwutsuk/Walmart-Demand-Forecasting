"""โหลดข้อมูลดิบ และสร้าง data/processed/m5_long.parquet (สินค้า×สาขา × วัน d_1..d_1969)"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
LONG_PATH = PROCESSED / 'm5_long.parquet'

ID_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def load_raw():
    """sales (wide, ถึง d_1941), calendar, prices"""
    sales = pd.read_csv(RAW / 'sales_train_evaluation.csv')
    calendar = pd.read_csv(RAW / 'calendar.csv')
    prices = pd.read_csv(RAW / 'sell_prices.csv')
    return sales, calendar, prices


def build_long(sales, calendar, prices):
    """
    ตาราง long เรียงตามสินค้า×สาขา แล้วตามวัน d = 1..1969
    sales เป็น NaN ในวันที่ยังไม่มีข้อมูล (d_1942 ขึ้นไป), sell_price เป็น NaN ก่อนสินค้าวางขาย
    snap คือวัน SNAP ของรัฐที่สาขานั้นตั้งอยู่
    """
    n_series, n_days = len(sales), len(calendar)
    d_cols = [c for c in sales.columns if c.startswith('d_')]

    y = np.full((n_series, n_days), np.nan, np.float32)
    y[:, :len(d_cols)] = sales[d_cols].to_numpy(np.float32)

    price_by_week = prices.pivot_table(index=['store_id', 'item_id'], columns='wm_yr_wk', values='sell_price')
    price_by_week = price_by_week.reindex(pd.MultiIndex.from_frame(sales[['store_id', 'item_id']]))
    price = price_by_week[calendar['wm_yr_wk']].to_numpy(np.float32)

    states = ['CA', 'TX', 'WI']
    snap_by_state = calendar[[f'snap_{s}' for s in states]].to_numpy(np.int8).T
    snap = snap_by_state[sales['state_id'].map({s: i for i, s in enumerate(states)}).to_numpy()]

    long = {}
    for c in ID_COLS:
        cat = sales[c].astype('category')
        long[c] = pd.Categorical.from_codes(np.repeat(cat.cat.codes.to_numpy(), n_days), cat.cat.categories)
    long['d'] = np.tile(np.arange(1, n_days + 1, dtype=np.int16), n_series)
    long['sales'] = y.ravel()
    long['sell_price'] = price.ravel()
    long['snap'] = snap.ravel()
    return pd.DataFrame(long)


def load_long():
    """อ่าน m5_long.parquet ถ้ายังไม่มีให้สร้างจากข้อมูลดิบก่อน"""
    if not LONG_PATH.exists():
        PROCESSED.mkdir(parents=True, exist_ok=True)
        build_long(*load_raw()).to_parquet(LONG_PATH, index=False)
    return pd.read_parquet(LONG_PATH)
