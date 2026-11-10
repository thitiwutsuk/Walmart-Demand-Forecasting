"""
Backtest แบบแบ่ง batch: เทรนทีละโมเดล และเก็บผลของแต่ละ fold ลงไฟล์ทันทีที่เสร็จ
หยุดกลางทางแล้วรันใหม่ได้ fold ที่มีไฟล์ผลแล้วจะถูกข้าม notebook 02 อ่านผลจากไฟล์เหล่านี้

    python -m src.backtest --list                  # ดูสถานะของทุกโมเดล
    python -m src.backtest "LightGBM"              # backtest ทุก fold ของโมเดลนี้
    python -m src.backtest "LightGBM" --final      # เทรนถึง d_1941 แล้วพยากรณ์ d_1942–d_1969
"""
import argparse
import pickle
import time

import numpy as np

from src.data import PROCESSED, load_long, load_raw
from src.features import CAT_FEATURES, HORIZON, FeatureBuilder

FOLDS = [1857, 1885, 1913]   # วันสุดท้ายของข้อมูลที่ใช้เทรนในแต่ละรอบ backtest
FINAL_END = 1941             # ข้อมูลจริงวันสุดท้าย ใช้เทรนตัวจริงเพื่อพยากรณ์ d_1942–d_1969
TRAIN_DAYS = 365             # โมเดล global: เทรนด้วยประวัติ 1 ปีล่าสุด (ลดเวลาและ RAM)
STORE_TRAIN_DAYS = 730       # โมเดลรายสาขา: ข้อมูลต่อโมเดลน้อยกว่า 10 เท่า จึงใช้ประวัติ 2 ปีได้
N_ROUNDS = 800               # จำนวนต้นไม้ของโมเดล boosting
SEED = 42
CACHE = PROCESSED / 'backtest'
MODEL_NAMES = ['LightGBM', 'LightGBM per store', 'Random Forest', 'XGBoost', 'CatBoost']


def _predict_test(model, fb, end, rows=None):
    X_test, _ = fb.build(end + 1, end + HORIZON, rows=rows)
    pred = model.predict(X_test).reshape(-1, HORIZON)
    pred[X_test['sell_price'].isna().to_numpy().reshape(-1, HORIZON)] = 0  # สินค้ายังไม่วางขาย
    return np.clip(pred, 0, None).astype(np.float32)


def _global(make_model, fb, end, fit_kwargs=None, sample=None):
    """โมเดลเดียวเรียนจากทุกสินค้า เทรนด้วยข้อมูลถึงวัน end แล้วทาย end+1..end+28"""
    X, y = fb.build(end - TRAIN_DAYS + 1, end)
    on_sale = X['sell_price'].notna().to_numpy()  # เทรนเฉพาะวันที่สินค้าวางขายแล้ว
    X, y = X[on_sale], y[on_sale]
    if sample and sample < len(y):
        idx = np.random.default_rng(SEED + end).choice(len(y), sample, replace=False)
        X, y = X.iloc[idx], y[idx]
    model = make_model()
    model.fit(X, y, **(fit_kwargs or {}))
    del X, y
    return _predict_test(model, fb, end), model


def _make_lgb():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.1, n_estimators=N_ROUNDS, learning_rate=0.05,
        num_leaves=127, min_child_samples=100, subsample=0.5, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )


def _make_xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective='reg:tweedie', tweedie_variance_power=1.1, tree_method='hist', n_estimators=N_ROUNDS,
        learning_rate=0.05, max_depth=8, subsample=0.5, colsample_bytree=0.8, max_bin=128,
    )


def _make_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        loss_function='Tweedie:variance_power=1.1', iterations=N_ROUNDS, learning_rate=0.1, depth=8,
        verbose=0, allow_writing_files=False,
    )


def _make_rf():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=50, max_features=0.5, n_jobs=-1)


def _lgb_by_store(fb, end):
    """1 โมเดลต่อสาขา คืนค่า (pred, {รหัสสาขา: model})"""
    pred = np.zeros((len(fb.ids), HORIZON), np.float32)
    models = {}
    store_codes = fb.id_codes['store_id']
    for code in np.unique(store_codes):
        rows = np.flatnonzero(store_codes == code)
        X, y = fb.build(end - STORE_TRAIN_DAYS + 1, end, rows=rows)
        on_sale = X['sell_price'].notna().to_numpy()
        model = _make_lgb()
        model.fit(X[on_sale], y[on_sale], categorical_feature=CAT_FEATURES)
        del X, y
        pred[rows] = _predict_test(model, fb, end, rows=rows)
        models[int(code)] = model
    return pred, models


def forecast(name, fb, end):
    """เทรนโมเดล name ด้วยข้อมูลถึงวัน end คืนค่า (pred [สินค้า×สาขา, 28 วัน], model)"""
    if name == 'LightGBM':
        return _global(_make_lgb, fb, end, fit_kwargs={'categorical_feature': CAT_FEATURES})
    if name == 'XGBoost':
        return _global(_make_xgb, fb, end)
    if name == 'CatBoost':
        return _global(_make_catboost, fb, end, fit_kwargs={'cat_features': CAT_FEATURES}, sample=3_000_000)
    if name == 'Random Forest':
        return _global(_make_rf, fb, end, sample=1_000_000)
    if name == 'LightGBM per store':
        return _lgb_by_store(fb, end)
    raise ValueError(f'unknown model: {name}')


def _slug(name):
    return name.lower().replace(' ', '_')


def fold_path(name, end):
    return CACHE / f'{_slug(name)}_d{end}.npz'


def final_path(name):
    return CACHE / f'final_{_slug(name)}.pkl'


def run_fold(name, fb, end):
    """backtest 1 fold แล้วบันทึกผล (เขียนไฟล์ชั่วคราวก่อน ถ้าถูกหยุดกลางทางจะไม่มีไฟล์เสีย)"""
    start = time.time()
    pred, _ = forecast(name, fb, end)
    train_min = (time.time() - start) / 60
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = fold_path(name, end).with_suffix('.tmp.npz')
    np.savez_compressed(tmp, pred=pred, train_min=train_min)
    tmp.replace(fold_path(name, end))
    return pred, train_min


def run_final(name, fb):
    """เทรนถึง FINAL_END แล้วเก็บทั้งค่าพยากรณ์และโมเดล (ใช้ทำ SHAP)"""
    start = time.time()
    pred, model = forecast(name, fb, FINAL_END)
    out = {'pred': pred, 'model': model, 'train_min': (time.time() - start) / 60}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = final_path(name).with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(out, f)
    tmp.replace(final_path(name))
    return out


def load_fold(name, end):
    """(pred, train_min) จากไฟล์ หรือ None ถ้ายังไม่ได้รัน"""
    path = fold_path(name, end)
    if not path.exists():
        return None
    with np.load(path) as f:
        return f['pred'], float(f['train_min'])


def load_final(name):
    path = final_path(name)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def status():
    for name in MODEL_NAMES:
        done = [f'd_{end}' for end in FOLDS if fold_path(name, end).exists()]
        final = 'มี' if final_path(name).exists() else '-'
        print(f'{name:20s} fold เสร็จ {len(done)}/{len(FOLDS)} {done}  final: {final}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='*', help=f'ชื่อโมเดล: {MODEL_NAMES}')
    parser.add_argument('--final', action='store_true', help='เทรนถึง d_1941 เพื่อพยากรณ์จริง แทนการ backtest')
    parser.add_argument('--list', action='store_true', help='แสดงสถานะ')
    args = parser.parse_args()
    if args.list or not args.models:
        status()
        return
    unknown = set(args.models) - set(MODEL_NAMES)
    if unknown:
        parser.error(f'unknown model(s): {sorted(unknown)}; choose from {MODEL_NAMES}')

    _, calendar, _ = load_raw()
    fb = FeatureBuilder(load_long(), calendar)
    for name in args.models:
        if args.final:
            print(f'{name}: final (train to d_{FINAL_END}) ...', flush=True)
            print(f"  done in {run_final(name, fb)['train_min']:.1f} min", flush=True)
            continue
        for end in FOLDS:
            if fold_path(name, end).exists():
                print(f'{name} d_{end}: มีผลแล้ว ข้าม', flush=True)
                continue
            print(f'{name} d_{end}: training ...', flush=True)
            _, train_min = run_fold(name, fb, end)
            print(f'  done in {train_min:.1f} min', flush=True)


if __name__ == '__main__':
    main()
