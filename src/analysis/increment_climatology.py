"""Mechanism: are records extreme in STATE but ordinary in TENDENCY?
For each 0.25deg record event (ERA5 r2017, 2020) and lead L in {12,48,120} h:
  state anomaly   zs = (y - mu)/sd                       (climatological units)
  needed increment zi = (y - x0)/sd_inc(L)               x0 = ERA5 at init; sd_inc = std of L-h increments of the cell,
                                                         same calendar month, 1979-2017 (training period)
Report: distribution of zs and zi for events vs all cell-times, and the fraction of events whose state lies beyond the
training max while their increment lies within the training range of increments (|zi| < max |zi| seen in training is
approximated by the 99.9th pct).  usage: python src/analysis/increment_climatology.py VAR KIND"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
VAR, KIND = sys.argv[1], sys.argv[2]; s = 1 if KIND == 'max' else -1
Y = 2020
fn = f'{C.D}/derived/incsd_{VAR}.npz'
if not os.path.exists(fn):
    # monthly std of L-h increments and their 99.9th pct of s*increment, 1979-2017 (00/12 UTC)
    acc = {L: [np.zeros((12, C.NC)) for _ in range(2)] for L in (12, 48, 120)}
    q = {L: [[] for _ in range(12)] for L in (12, 48, 120)}
    rng = np.random.default_rng(0)
    for yr in range(1979, 2018):
        t, x = C.truth_year(VAR, yr)
        m = t.astype('datetime64[M]').astype(int) % 12
        for L in (12, 48, 120):
            k = L // 12; d = (x[k:] - x[:-k]).astype(np.float64); mm = m[k:]
            for mi in range(12):
                sel = d[mm == mi]
                acc[L][0][mi] += sel.sum(0); acc[L][1][mi] += (sel ** 2).sum(0)
    n = {L: np.array([((np.arange(1979, 2018)[:, None] * 0 + 1)).sum() for _ in range(12)]) for L in (12, 48, 120)}
    out = {}
    for L in (12, 48, 120):
        # counts per month: approximate by days in month*2*39
        cnt = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])[:, None] * 2 * 39
        mu = acc[L][0] / cnt; out[f'sd{L}'] = np.sqrt(acc[L][1] / cnt - mu ** 2).astype(np.float32); out[f'mu{L}'] = mu.astype(np.float32)
    np.savez(fn, **out)
I = np.load(fn)
vt, y = C.truth_year(VAR, Y); vp, yp = C.truth_year(VAR, Y - 1)
ally = np.concatenate([yp, y]); T = len(vt); off = len(vp)
mu, sd = C.clim_at(C.clim(VAR), vt)
mon = vt.astype('datetime64[M]').astype(int) % 12
r = C.records(VAR, 1979, 2017, KIND)[mon]
ev = s * (y - r) > 0
zs = s * (y - mu) / sd
print(f'{VAR} {KIND} 2020 events={ev.sum()}  state z (clim sd): events median {np.median(zs[ev]):.2f}, all-days 99.9pct {np.percentile(zs[::7], 99.9):.2f}')
for L in (12, 48, 120):
    k = L // 12
    x0 = ally[np.arange(T) + off - k]
    sdi = I[f'sd{L}'][mon]; mui = I[f'mu{L}'][mon]
    zi = s * (y - x0 - mui) / sdi
    zall = zi[::5].ravel(); zall = zall[np.isfinite(zall)]
    p = np.percentile(zall, [50, 90, 99, 99.9])
    ze = zi[ev]
    print(f' lead {L:3d}: needed increment z on events: median {np.median(ze):+.2f}, 90% {np.percentile(ze,90):+.2f} | '
          f'all cell-times 50/90/99/99.9%: {p.round(2)} | frac events with increment z > all-days 99.9pct: {(ze > p[3]).mean():.3f} '
          f'| share of record anomaly already present at init: {np.median(s*(x0[ev]-mu[ev])/(s*(y[ev]-mu[ev]))):.2f}')
