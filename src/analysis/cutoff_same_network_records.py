"""Same-architecture training-cutoff experiment (0.25 deg, 2020): GraphCast trained 1979-2019 (WB2 'graphcast') vs the
public 1979-2017 checkpoint run by us ('gc2017'), identical ERA5 inputs; only training years differ.
 (a) record events (ERA5 r2017): paired d = s(f19 - f17); DiD gap-cells (2018-19 new records, event below r2019) vs unseen
 (b) input-conditioned E[s(y-f) | u0] with u0 relative to r2017, in seen cells (OOD for gc2017 only) and unseen cells
 (c) slope of s(f-y) against the record exceedance, both models, same events.
Only init times present in gc2017 are used for both models. usage: python src/analysis/cutoff_same_network_records.py VAR KIND"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
VAR, KIND = sys.argv[1], sys.argv[2]; s = 1 if KIND == 'max' else -1; Y = 2020
rng = np.random.default_rng(0)
vt, y = C.truth_year(VAR, Y); vp, yp = C.truth_year(VAR, Y - 1); ally = np.concatenate([yp, y]); T = len(vt)
mu, sd = C.clim_at(C.clim(VAR), vt); mon = vt.astype('datetime64[M]').astype(int) % 12
r17 = np.load(f'{C.D}/derived/rec_{VAR}_{KIND}_1979_2017.npy')[mon]; r19 = np.load(f'{C.D}/derived/rec_{VAR}_{KIND}_1979_2019.npy')[mon]
g = s * (r19 - r17) / sd
ev = s * (y - r17) > 0
grp = np.full(y.shape, -1, np.int8); grp[ev & (g == 0)] = 0; grp[ev & (g >= 0.3) & (s * (y - r19) <= 0)] = 1; grp[ev & (g >= 0.3) & (s * (y - r19) > 0)] = 2
wk = (vt.astype('datetime64[D]').astype(int) // 7)[:, None]
cl = wk * 1000 + ((C.LON // 30).astype(int) * 10 + ((C.LAT + 90) // 30).astype(int))[None, :]


def boot_did(v, m1, m0, B=300):
    _, cid = np.unique(cl[m1 | m0], return_inverse=True)
    vv = v[m1 | m0]; a = m1[m1 | m0]; K = cid.max() + 1
    est = vv[a].mean() - vv[~a].mean(); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K)[cid]
        bs.append((w * vv * a).sum() / (w * a).sum() - (w * vv * ~a).sum() / (w * ~a).sum())
    return est, *np.percentile(bs, [2.5, 97.5])


ls = list(np.load(f'{C.D}/fc/gc2017/leads.npy'))
for L in ls:
    _, F17 = C.forecast_valid('gc2017', VAR, Y, L); _, F19 = C.forecast_valid('graphcast', VAR, Y, L)
    ok = np.isfinite(F17) & np.isfinite(F19)
    d = s * (F19 - F17)
    print(f'== {VAR} {KIND} lead {L}: valid cell-times {ok.sum()}, events {(ev & ok).sum()} (unseen {(ok & (grp == 0)).sum()}, gap {(ok & (grp == 1)).sum()}, beyond {(ok & (grp == 2)).sum()})')
    print(f'   overall mean d(19-17) all land {np.nanmean(d[ok]):+.3f}; rmse vs ERA5: gc17 {np.sqrt(np.nanmean(((F17 - y)[ok]) ** 2)):.3f} gc19 {np.sqrt(np.nanmean(((F19 - y)[ok]) ** 2)):.3f}')
    for gi, nm in ((1, 'gap'), (2, 'beyond')):
        e, lo, hi = boot_did(d, ok & (grp == gi), ok & (grp == 0))
        print(f'   events: mean d unseen {d[ok & (grp == 0)].mean():+.3f} | {nm} {d[ok & (grp == gi)].mean():+.3f} | DiD {e:+.3f} [{lo:+.3f},{hi:+.3f}]')
    # slope against record exceedance on the same events
    e_ = s * (y - r17)
    for nm, F in (('gc17', F17), ('gc19', F19)):
        m = ev & ok; er = s * (F - y)[m]
        print(f'   {nm}: record bias {er.mean():+.3f}, slope vs exceedance {np.polyfit(e_[m], er, 1)[0]:+.3f}')
    # input-conditioned (u0 wrt r2017) in seen (g>=0.3) and unseen cells
    x0 = ally[np.arange(T) + len(vp) - L // 12]; u0 = s * (x0 - r17) / sd
    for snm, sm in (('seen', g >= 0.3), ('unseen', g == 0)):
        row = []
        for a, b in ((-1, -0.5), (-0.5, 0), (0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2)):
            k = ok & sm & (u0 >= a) & (u0 < b)
            row.append(f'[{a},{b}) n={k.sum()} 17:{np.mean(s*(y-F17)[k]):+.2f} 19:{np.mean(s*(y-F19)[k]):+.2f}')
        print(f'   input-cond {snm}: ' + ' | '.join(row), flush=True)
