"""WRMSSE ตามกติกา Kaggle M5: รวมผลสินค้า×สาขาขึ้นเป็น 12 ระดับ (42,840 อนุกรม) แล้วถ่วงน้ำหนักด้วยยอดขายเป็นเงิน"""
import numpy as np
import pandas as pd
from scipy import sparse

ID_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

LEVELS = {
    1: [],
    2: ['state_id'],
    3: ['store_id'],
    4: ['cat_id'],
    5: ['dept_id'],
    6: ['state_id', 'cat_id'],
    7: ['state_id', 'dept_id'],
    8: ['store_id', 'cat_id'],
    9: ['store_id', 'dept_id'],
    10: ['item_id'],
    11: ['item_id', 'state_id'],
    12: ['item_id', 'store_id'],
}


def aggregation_matrix(ids):
    """Sparse matrix (จำนวนอนุกรมทุกระดับ × จำนวนสินค้า×สาขา) และเลขระดับของแต่ละแถว"""
    n = len(ids)
    blocks, level_of = [], []
    for level, cols in LEVELS.items():
        codes = np.zeros(n, dtype=np.int64) if not cols else ids.groupby(cols, observed=True, sort=False).ngroup().to_numpy()
        n_groups = codes.max() + 1
        blocks.append(sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n)))
        level_of.append(np.full(n_groups, level))
    return sparse.vstack(blocks).tocsr(), np.concatenate(level_of)


class WRMSSE:
    """
    train_wide: ตาราง sales_train_* ที่ตัดคอลัมน์ d_ ให้จบที่วันสุดท้ายของช่วง train
    calendar, prices: calendar.csv และ sell_prices.csv
    """

    def __init__(self, train_wide, calendar, prices, h=28):
        d_cols = [c for c in train_wide.columns if c.startswith('d_')]
        ids = train_wide[ID_COLS].astype(str).reset_index(drop=True)
        y = train_wide[d_cols].to_numpy(np.float64)

        self.agg, self.level_of = aggregation_matrix(ids)
        Y = self.agg @ y

        # scale = MSE ของ naive 1 วัน นับตั้งแต่วันแรกที่อนุกรมนั้นมียอดขาย
        started = np.maximum.accumulate(Y != 0, axis=1)
        valid = started[:, :-1]
        sq_diff = np.diff(Y, axis=1) ** 2
        n_valid = valid.sum(axis=1)
        scale = np.where(valid, sq_diff, 0).sum(axis=1) / np.maximum(n_valid, 1)
        self.scale = np.where(scale > 0, scale, 1.0)

        # weight = ยอดขายเป็นเงิน h วันสุดท้าย ทุกระดับรวมกันได้ระดับละ 1
        last_days = d_cols[-h:]
        weeks = calendar.set_index('d').loc[last_days, 'wm_yr_wk'].to_numpy()
        price_wide = (
            prices[prices.wm_yr_wk.isin(set(weeks))]
            .pivot_table(index=['store_id', 'item_id'], columns='wm_yr_wk', values='sell_price')
        )
        price = price_wide.reindex(pd.MultiIndex.from_frame(ids[['store_id', 'item_id']]))[weeks].to_numpy()
        dollars = np.nansum(y[:, -h:] * price, axis=1)
        self.weights = (self.agg @ dollars) / dollars.sum()
        self.weights[scale == 0] = 0.0

    def score(self, y_true, y_pred, by_level=False):
        """y_true, y_pred: array (สินค้า×สาขา, h) เรียงแถวเหมือน train_wide"""
        err = self.agg @ (np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64))
        rmsse = np.sqrt((err ** 2).mean(axis=1) / self.scale)
        per_level = pd.Series(
            np.bincount(self.level_of, weights=self.weights * rmsse)[1:], index=pd.Index(LEVELS, name='level')
        )
        return (per_level.mean(), per_level) if by_level else per_level.mean()
