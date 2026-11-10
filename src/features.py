"""
Feature สำหรับพยากรณ์แบบ direct: feature ยอดขายทุกตัวมองย้อนหลังอย่างน้อย 28 วัน
จึงทายวันที่ 1–28 ข้างหน้าได้ในครั้งเดียวโดยไม่ต้องใช้ค่าที่ทายเองเป็น input
"""
import numpy as np
import pandas as pd

ID_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
CAT_FEATURES = ID_COLS + ['event_name_1', 'event_type_1']
HORIZON = 28
LAGS = [28, 35, 42, 49, 56]
WINDOWS = [7, 28, 56]
FIRST_USABLE_DAY = HORIZON + max(WINDOWS)  # วันแรกที่มีประวัติพอคำนวณทุก feature


class FeatureBuilder:
    """
    long_df: data/processed/m5_long.parquet (เรียงตามสินค้า×สาขา แล้วตามวัน d = 1..N)
    calendar: calendar.csv
    """

    def __init__(self, long_df, calendar):
        n_days = int(long_df['d'].max())
        n_series = len(long_df) // n_days
        self.ids = long_df.iloc[::n_days][ID_COLS].reset_index(drop=True)
        self.id_codes = {c: self.ids[c].cat.codes.to_numpy(np.int16) for c in ID_COLS}

        self.sales = long_df['sales'].to_numpy(np.float32).reshape(n_series, n_days)
        self.price = long_df['sell_price'].to_numpy(np.float32).reshape(n_series, n_days)
        self.snap = long_df['snap'].to_numpy(np.int8).reshape(n_series, n_days)
        # cumsum[:, k] = ยอดรวมของวัน index 0..k-1 ใช้คำนวณ rolling mean ได้ในครั้งเดียว
        self.cumsum = np.zeros((n_series, n_days + 1), np.float64)
        np.cumsum(np.nan_to_num(self.sales), axis=1, out=self.cumsum[:, 1:])
        # Running maximum uses only prices known up to each day.
        self.regular_price = np.fmax.accumulate(self.price, axis=1)

        cal = calendar.iloc[:n_days]
        self.day = {
            'wday': cal['wday'].to_numpy(np.int8),
            'month': cal['month'].to_numpy(np.int8),
            'mday': pd.to_datetime(cal['date']).dt.day.to_numpy(np.int8),
            'event_name_1': (cal['event_name_1'].astype('category').cat.codes + 1).to_numpy(np.int8),  # 0 = ไม่มี event
            'event_type_1': (cal['event_type_1'].astype('category').cat.codes + 1).to_numpy(np.int8),
        }

    def build(self, d_from, d_to, rows=None):
        """
        Feature และยอดขายจริงของวัน d_from..d_to (นับวันแบบ d_1 = 1) เรียงตามสินค้า×สาขา แล้วตามวัน
        rows: index ของสินค้า×สาขาที่ต้องการ (เช่น เฉพาะสาขาเดียว) ค่าเริ่มต้นคือทุกรายการ
        """
        if d_from < FIRST_USABLE_DAY:
            raise ValueError(f'd_from must be >= {FIRST_USABLE_DAY}')
        rows = np.arange(len(self.ids)) if rows is None else np.asarray(rows)
        j = np.arange(d_from, d_to + 1) - 1
        n, m = len(rows), len(j)
        grid = np.ix_(rows, j)

        X = {c: np.repeat(codes[rows], m) for c, codes in self.id_codes.items()}
        X.update({c: np.tile(v[j], n) for c, v in self.day.items()})
        X['snap'] = self.snap[grid].ravel()

        price = self.price[grid]
        X['sell_price'] = price.ravel()
        X['price_rel_regular'] = (price / self.regular_price[grid]).ravel()
        X['price_change_7d'] = (price / self.price[np.ix_(rows, j - 7)] - 1).ravel()

        for lag in LAGS:
            X[f'lag_{lag}'] = self.sales[np.ix_(rows, j - lag)].ravel()
        end = j - HORIZON + 1
        for w in WINDOWS:
            total = self.cumsum[np.ix_(rows, end)] - self.cumsum[np.ix_(rows, end - w)]
            X[f'rmean_{w}'] = (total / w).astype(np.float32).ravel()

        return pd.DataFrame(X), self.sales[grid].ravel()
