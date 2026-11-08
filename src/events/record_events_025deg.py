"""0.25 deg: event tables (+expected error, per-cell window) and forward reliability bins.
usage: python src/events/record_events_025deg.py YEAR VAR KIND TRUTH models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
from common.ordinary_days import _win

Y, VAR, KIND, TRUTH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(','); LEADS = [int(x) for x in sys.argv[6].split(',')]
s = 1 if KIND == 'max' else -1
od = f'{C.D}/out'; os.makedirs(od, exist_ok=True)
vt, y = C.truth_year(VAR, Y, src=TRUTH)
mu, sd = C.clim_at(C.clim(VAR), vt)
m_ = vt.astype('datetime64[M]').astype(int) % 12
r17 = C.records(VAR, 1979, 2017, KIND)[m_]; r19 = C.records(VAR, 1979, 2019, KIND)[m_]
g = s * (r19 - r17) / sd
ev = s * (y - r17) > 0
ti, ci = np.where(ev)
print(Y, VAR, KIND, TRUTH, 'events', len(ti), flush=True)
ya = (y - mu).astype(np.float64)
EDGES = np.arange(-6, 4.0001, 0.1); nb = len(EDGES) - 1
subsets = {'all': np.ones_like(g, bool), 'unseen': g == 0, 'seen': g >= 0.3}
for name in MODELS:
    fn = f'{od}/{Y}_{VAR}_{KIND}_{TRUTH}_{name}.npz'
    if os.path.exists(fn): continue
    res = {k: [] for k in ('f', 'bhat', 'B', 'n')}
    fw = np.zeros((len(LEADS), 2, len(subsets), nb, 4))
    for li, L in enumerate(LEADS):
        _, F = C.forecast_valid(name, VAR, Y, L)
        fa = (F - mu).astype(np.float64)
        M = np.isfinite(fa) & np.isfinite(ya)
        yy = np.where(M, ya, 0); ff = np.where(M, fa, 0)
        S = {k: _win(v) for k, v in dict(n=M.astype(np.float64), y=yy, f=ff, yy=yy * yy, fy=ff * yy).items()}
        n = np.maximum(S['n'], 1); my, mf = S['y'] / n, S['f'] / n
        B = (S['fy'] / n - mf * my) / (S['yy'] / n - my ** 2)
        bh = (mf - my) + (B - 1) * (ya - my)
        res['f'].append(F[ev]); res['bhat'].append(bh[ev].astype(np.float32)); res['B'].append(B[ev].astype(np.float32))
        res['n'].append(S['n'][ev].astype(np.float32))
        ok = np.isfinite(F)
        for ri, r in enumerate((r17, r19)):
            u = s * (F - r) / sd; w = s * (y - r) / sd
            b = np.floor((np.nan_to_num(u, nan=-99) - EDGES[0]) / 0.1).astype(np.int64)
            for si, msk in enumerate(subsets.values()):
                k = msk & ok & (b >= 0) & (b < nb)
                bb = b[k]
                fw[li, ri, si, :, 0] = np.bincount(bb, minlength=nb)
                fw[li, ri, si, :, 1] = np.bincount(bb, w[k], minlength=nb)
                fw[li, ri, si, :, 2] = np.bincount(bb, w[k] ** 2, minlength=nb)
                fw[li, ri, si, :, 3] = np.bincount(bb, u[k], minlength=nb)
        e = s * (F[ev] - y[ev])
        print(name, L, 'bias %.2f expected %.2f' % (np.nanmean(e), np.nanmean(s * bh[ev])), flush=True)
    np.savez(fn, t=ti, c=ci, y=y[ev], r=r17[ev], r19=r19[ev], sd=sd[ev], mu=mu[ev], leads=np.array(LEADS), vt=vt,
             fw=fw, edges=EDGES, subsets=list(subsets), **{k: np.array(v) for k, v in res.items()})
