import os
from pathlib import Path

import pandas as pd
import pytest

RAW_FILES = ['calendar.csv', 'sales_train_evaluation.csv', 'sell_prices.csv']


@pytest.fixture(scope='session')
def raw_dir():
    """โฟลเดอร์ข้อมูลดิบ M5 (ตั้งได้ด้วย M5_RAW_DIR) ถ้าไม่พบจะข้ามเทสต์ที่ต้องใช้ข้อมูลจริง"""
    path = Path(os.environ.get('M5_RAW_DIR', Path(__file__).resolve().parents[1] / 'data' / 'raw'))
    missing = [f for f in RAW_FILES if not (path / f).exists()]
    if missing:
        pytest.skip(f'M5 raw data not found in {path} (missing: {", ".join(missing)})')
    return path


@pytest.fixture(scope='session')
def raw_data(raw_dir):
    """(sales, calendar, prices) เหมือน src/data.py load_raw() โหลดครั้งเดียวต่อการรัน"""
    return tuple(pd.read_csv(raw_dir / f) for f in ['sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv'])
