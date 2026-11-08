"""Is there a ceiling in TENDENCY space? (0.25 deg, 2020)
Forecast increment  zf = (f - x0_model - mu_inc)/sd_inc,  true increment zy = (y - x0_era5 - mu_inc)/sd_inc,
sd_inc/mu_inc = 1979-2017 monthly stats of L-h ERA5 increments (increment_climatology.py). x0_model = the model's own initial
state (ERA5 for AI, HRES-fc0 for HRES). Leads 48/120 h (same UTC hour as init, no diurnal term).
Reports tail-frequency ratio P(zf>q)/P(zy>q) and forward reliability E[zy | zf] in the tail.
usage: python src/analysis/increment_tail_calibration.py VAR KIND models"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
VAR, KIND = sys.argv[1], sys.argv[2]; MODELS = sys.argv[3].split(','); s = 1 if KIND == 'max' else -1
Y = 2020
I = np.load(f'{C.D}/derived/incsd_{VAR}.npz')
vt, y = C.truth_year(VAR, Y); T = len(vt)
mon = vt.astype('datetime64[M]').astype(int) % 12
def lagged(src, k):
    _, a = C.truth_year(VAR, Y, src=src); _, b = C.truth_year(VAR, Y - 1, src=src)
    return np.concatenate([b, a])[np.arange(T) + len(b) - k]
Q = [2, 3, 4, 5]
for L in (48, 120):
    k = L // 12
    x0e = lagged('era5', k)
    sdi = I[f'sd{L}'][mon]; mui = I[f'mu{L}'][mon]
    zy = s * (y - x0e - mui) / sdi
    print(f'== {VAR} {KIND} lead {L}: tail freq P(z>q) for q={Q} (truth row first), then E[zy|zf] for zf bins')
    print(f' truth       ' + ' '.join(f'{np.mean(zy > q):.2e}' for q in Q))
    for name in MODELS:
        _, F = C.forecast_valid(name, VAR, Y, L)
        x0 = lagged('hres_t0', k) if name == 'hres' else x0e
        zf = s * (F - x0 - mui) / sdi
        ok = np.isfinite(zf)
        fr = ' '.join(f'{np.mean(zf[ok] > q):.2e}' for q in Q)
        bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 99)]
        rel = ' '.join(f'[{a},{b}):{np.mean(zy[ok & (zf >= a) & (zf < b)]):+.2f}/{(ok & (zf >= a) & (zf < b)).sum()}' for a, b in bins)
        print(f' {name:11s} {fr} | {rel}', flush=True)
