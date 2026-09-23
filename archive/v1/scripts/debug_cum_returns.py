from pathlib import Path
import pandas as pd
import numpy as np

results_dir = Path('results')
timestamp = '20260730_003330'
pred_dir = results_dir / 'predictions' / timestamp

print('Pred dir:', pred_dir)

# helper

def load_series(file_name):
    p = pred_dir / file_name
    if not p.exists():
        print(f'Missing {p}')
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(float)
    detected = False
    if np.nanmax(np.abs(s.values)) > 1.5:
        print(f'Detected percent-scale in {file_name}; converting to decimals')
        s = s / 100.0
        detected = True
    return s

moe_series = load_series('moe_portfolio_returns.csv')
rolling_series = load_series('rolling_avg_portfolio_returns.csv')

# load actuals for equal weight
act_p = pred_dir / 'moe_actuals.csv'
if act_p.exists():
    act = pd.read_csv(act_p, index_col=0, parse_dates=True)
    equal = pd.Series(np.nanmean(act.values, axis=1), index=act.index)
    if np.nanmax(np.abs(equal.values)) > 1.5:
        print('Detected percent-scale in actuals; converting to decimals')
        equal = equal / 100.0
else:
    equal = None

print('\n--- Series heads ---')
for name, s in [('moe', moe_series), ('rolling', rolling_series), ('equal', equal)]:
    if s is None:
        print(name, 'missing')
        continue
    print(f"\n{name} shape: {s.shape}")
    print(s.head())
    print('...')
    print(s.tail())
    print('max, min, last:', s.max(), s.min(), s.iloc[-1])

# reconstruct from preds if needed
preds_p = pred_dir / 'moe_predictions.csv'
act_p = pred_dir / 'moe_actuals.csv'
if preds_p.exists() and act_p.exists():
    preds = pd.read_csv(preds_p, index_col=0, parse_dates=True)
    acts = pd.read_csv(act_p, index_col=0, parse_dates=True)
    print('\nPreds shape, acts shape:', preds.shape, acts.shape)
    preds_arr = preds.values
    acts_arr = acts.values
    if np.nanmax(np.abs(preds_arr)) > 1.5 or np.nanmax(np.abs(acts_arr)) > 1.5:
        print('Detected percent-scale in preds/acts; converting to decimals')
        preds_arr = preds_arr / 100.0
        acts_arr = acts_arr / 100.0
    # simple reconstruction same as viz
    weights = np.where(preds_arr > 0, preds_arr, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums != 0)
    returns = np.sum(weights * acts_arr, axis=1)
    print('reconstructed returns shape:', returns.shape)
    print('first 6:', returns[:6])
    print('last 6:', returns[-6:])
    # cumulative
    n_predictions = 83
    start_date = pd.Timestamp('2019-08-31')
    full_dates = pd.date_range(start=start_date, periods=n_predictions, freq='ME')
    # align
    def to_aligned(arr_or_series):
        if arr_or_series is None:
            return pd.Series(0.0, index=full_dates)
        if isinstance(arr_or_series, (np.ndarray, list)):
            arr = np.asarray(arr_or_series)
            if np.nanmax(np.abs(arr)) > 1.5:
                arr = arr / 100.0
            pad = max(0, len(full_dates) - arr.size)
            padded = np.concatenate([np.zeros(pad), arr])
            return pd.Series(padded, index=full_dates)
        s = arr_or_series.copy()
        s.index = pd.to_datetime(s.index)
        return s.reindex(full_dates).fillna(0.0)

    moe_al = to_aligned(moe_series if moe_series is not None else returns)
    rolling_al = to_aligned(rolling_series)
    equal_al = to_aligned(equal)

    print('\nAligned moe head:', moe_al.head())
    print('Aligned moe tail:', moe_al.tail())
    print('moe_al max/min/last:', moe_al.max(), moe_al.min(), moe_al.iloc[-1])
    print('\nCumulative last values:')
    print('moe cum last:', (1+moe_al).cumprod().iloc[-1])
    print('rolling cum last:', (1+rolling_al).cumprod().iloc[-1])
    print('equal cum last:', (1+equal_al).cumprod().iloc[-1])
else:
    print('Preds/actuals not available for reconstruction')

print('\nDone')
