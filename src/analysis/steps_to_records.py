"""Are record STATES and the TENDENCIES leading to them out of the training distribution -- locally or globally?
0.25 deg ERA5 t2m, training years 1979-2017.  Step sizes: 6 h (GraphCast step) and 24 h (Pangu's main step).
For each 2020 record event (heat: ERA5 > 1979-2017 monthly record) and lead L, take the true path x(v-L) ... x(v) and its
step increments d_k. Report the fraction of events whose largest path increment exceeds
  (a) the cell's own 1979-2017 maximum increment of that step size (local OOD tendency)
  (b) the global land 1979-2017 maximum / 99.99th percentile increment (network-level OOD tendency),
next to the state itself (all events exceed the local record by definition; 0 exceed the global land max).
usage: python src/analysis/steps_to_records.py KIND (max|min)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
KIND = sys.argv[1]; s = 1 if KIND == 'max' else -1
fn = f'{C.D}/derived/tend_{KIND}.npz'
STEPS = {'6h': 1, '24h': 4}
if not os.path.exists(fn):
    loc = {k: np.full(C.NC, -np.inf, np.float32) for k in STEPS}
    hist = {k: np.zeros(4000) for k in STEPS}        # global histogram of s*increment, 0.01 K bins from -20 to 20
    prev = None
    for y in range(1979, 2018):
        x = np.load(f'{C.D}/era5/2m_temperature_{y}.npy', mmap_mode='r')
        x = np.asarray(x) if prev is None else np.concatenate([prev, np.asarray(x)])
        for k, st in STEPS.items():
            d = s * (x[st:] - x[:-st])
            loc[k] = np.maximum(loc[k], d.max(0))
            hist[k] += np.histogram(d[::3].ravel(), bins=4000, range=(-20, 20))[0]
        prev = x[-4:]
        print('train', y, flush=True)
    np.savez(fn, **{f'loc_{k}': v for k, v in loc.items()}, **{f'hist_{k}': v for k, v in hist.items()})
T = np.load(fn)
edges = np.linspace(-20, 20, 4001)
def pct(h, q):
    c = np.cumsum(h) / h.sum(); return edges[1:][np.searchsorted(c, q)]
x20 = np.concatenate([np.load(f'{C.D}/era5/2m_temperature_2019.npy', mmap_mode='r')[-40:], np.load(f'{C.D}/era5/2m_temperature_2020.npy')])
t20 = np.arange(np.datetime64('2020-01-01T00'), np.datetime64('2021-01-01T00'), np.timedelta64(6, 'h'))
hr = (t20 - t20.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
mon = t20.astype('datetime64[M]').astype(int) % 12
R = np.load(f'{C.D}/derived/rec_t2m_{KIND}_1979_2017.npy')
glob_max_state = R.max() if KIND == 'max' else R.min()
v_idx = np.where(np.isin(hr, (0, 12)))[0]
xv = x20[40 + v_idx]; ev = s * (xv - R[mon[v_idx]]) > 0
ti, ci = np.where(ev)
print(f'2020 {KIND} records: {len(ti)} events; beyond global land 1979-2017 extreme state: {(s*(xv[ev]-glob_max_state) > 0).mean():.4f}')
for k, st in STEPS.items():
    gmax = pct(T[f'hist_{k}'], 1.0 - 1e-9); g9999 = pct(T[f'hist_{k}'], 0.9999); g999 = pct(T[f'hist_{k}'], 0.999)
    print(f' step {k}: global land training increment 99.9%={g999:.2f} K, 99.99%={g9999:.2f} K, max~{gmax:.2f} K')
    for L in (24, 48, 120):
        nst = L // 6
        if nst < st: continue
        vi = 40 + v_idx[ti]
        # path increments of size st covering (v-L, v]
        incs = np.stack([s * (x20[vi - j, ci] - x20[vi - j - st, ci]) for j in range(0, nst - st + 1, st)], 0)
        mx = incs.max(0)
        print(f'   lead {L:3d}: largest path step-increment median {np.median(mx):.2f} K | > local cell max: {(mx > T[f"loc_{k}"][ci]).mean():.4f}'
              f' | > global 99.9%: {(mx > g999).mean():.4f} | > global 99.99%: {(mx > g9999).mean():.4f} | > global max: {(mx > gmax).mean():.4f}', flush=True)
