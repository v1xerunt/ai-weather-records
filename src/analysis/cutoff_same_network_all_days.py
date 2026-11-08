"""Same-architecture cutoff DiD (GC-2019 - GC-2017) on ordinary days vs record events vs warm non-record days,
2018-19 new-record cells (g>=0.3 sd) vs unseen cells (g==0). 0.25 deg, 2020, available gc2017 inits. Cluster bootstrap
(week x 30deg); ordinary days subsampled for speed.
usage: python src/analysis/cutoff_same_network_all_days.py"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
rng = np.random.default_rng(0)
vt, y = C.truth_year('t2m', 2020); mu, sd = C.clim_at(C.clim('t2m'), vt); mon = vt.astype('datetime64[M]').astype(int) % 12
r17 = np.load(f'{C.D}/derived/rec_t2m_max_1979_2017.npy')[mon]; r19 = np.load(f'{C.D}/derived/rec_t2m_max_1979_2019.npy')[mon]
g = (r19 - r17) / sd; ev = y > r17; seen = g >= 0.3; unseen = g == 0
wk = (vt.astype('datetime64[D]').astype(int) // 7)[:, None]
cl = wk * 1000 + ((C.LON // 30).astype(int) * 10 + ((C.LAT + 90) // 30).astype(int))[None, :]
sub = np.zeros_like(ev); sub[:, ::5] = True
def did(d, a, b, B=200):
    m = a | b; _, cid = np.unique(cl[m], return_inverse=True); K = cid.max() + 1; v = d[m]; aa = a[m]
    est = v[aa].mean() - v[~aa].mean(); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K)[cid]
        bs.append((w * v * aa).sum() / (w * aa).sum() - (w * v * ~aa).sum() / (w * ~aa).sum())
    return f'{est:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]'
for L in (12, 48, 120):
    _, F17 = C.forecast_valid('gc2017', 't2m', 2020, L); _, F19 = C.forecast_valid('graphcast', 't2m', 2020, L)
    ok = np.isfinite(F17) & np.isfinite(F19); d = F19 - F17; warm = (y - r17) / sd > -1
    print(f'lead {L}: ordinary {did(d, ok & seen & ~ev & sub, ok & unseen & ~ev & sub)} | warm non-record {did(d, ok & seen & ~ev & warm, ok & unseen & ~ev & warm)} | records {did(d, ok & seen & ev, ok & unseen & ev)}', flush=True)
