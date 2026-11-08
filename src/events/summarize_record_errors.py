"""Cluster-bootstrap summary of observed vs expected record bias (1.5 deg event tables).
Cluster = ISO week x 30deg lon x 30deg lat. Signs: s*(f-y) so negative = too mild (underestimates the extreme).
usage: python src/events/summarize_record_errors.py YEAR VAR KIND TRUTH models leads  -> prints markdown rows"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15

NB = 400
rng = np.random.default_rng(0)


def clusters(d):
    wk = (d['vt'][d['t']].astype('datetime64[D]').astype(int) // 7)
    lo = (grid15.LON[d['i']] // 30).astype(int); la = ((grid15.LAT[d['j']] + 90) // 30).astype(int)
    _, c = np.unique(wk * 1000 + lo * 10 + la, return_inverse=True)
    return c


def boot(c, *cols, stat):
    """stat(*cols_subset) -> scalar; resample clusters."""
    K = c.max() + 1
    idx = [np.where(c == k)[0] for k in range(K)]
    est = stat(*cols)
    bs = []
    for _ in range(NB):
        pick = np.concatenate([idx[k] for k in rng.integers(0, K, K)])
        bs.append(stat(*[x[pick] for x in cols]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def slope(e, z):
    e = e - e.mean(); return (e * (z - z.mean())).sum() / (e * e).sum()


def rows(Y, VAR, KIND, TRUTH, models, leads):
    s = 1 if KIND == 'max' else -1
    out = []
    for name in models:
        fn = f'{grid15.D}/events/{Y}_{VAR}_{KIND}_{TRUTH}_{name}.npz'
        if not os.path.exists(fn): continue
        d = np.load(fn); c0 = clusters(d); e0 = s * (d['y'] - d['r'])
        for L in leads:
            li = int(np.where(d['leads'] == L)[0][0])
            err = s * (d['f'][li] - d['y']); bh = s * d['bhat'][li]
            ok = np.isfinite(err) & np.isfinite(bh) & (d['n'][li] > 50)
            err, bh, e, c = err[ok], bh[ok], e0[ok], c0[ok]
            _, c = np.unique(c, return_inverse=True)
            q = np.quantile(e, 0.8)
            b = boot(c, err, stat=np.mean)
            x = boot(c, err - bh, stat=np.mean)
            so = boot(c, e, err, stat=slope); sg = boot(c, e, bh, stat=slope)
            tx = boot(c, err - bh, e, stat=lambda r, ee: r[ee >= q].mean())
            rm = np.sqrt((err ** 2).mean()); rg = np.sqrt(d['mse_hat'][li][ok].mean())
            f = lambda t: f'{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]'
            out.append(f'| {Y} | {VAR} {KIND} | {TRUTH} | {name} | {L} | {ok.sum()} | {f(b)} | {bh.mean():+.2f} | {f(x)} | '
                       f'{f(so)} | {sg[0]:+.2f} | {f(tx)} | {rm:.2f} / {rg:.2f} |')
    return out


HDR = ('| year | var | truth | model | lead h | N | bias obs [95%] | bias expected | residual obs−expected | slope obs vs e | '
       'slope expected | tail excess (top20% e) | RMSE obs / expected |\n|' + '---|' * 13)

if __name__ == '__main__':
    Y, VAR, KIND, TRUTH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    for r in rows(Y, VAR, KIND, TRUTH, sys.argv[5].split(','), [int(x) for x in sys.argv[6].split(',')]):
        print(r, flush=True)
