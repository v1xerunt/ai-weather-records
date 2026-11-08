"""Knowledge vs amplitude among record events: Spearman rank correlation between forecast exceedance s(f - r) and true
exceedance s(y - r) on the SAME events, per model and lead (rank-based: invariant to any monotone recalibration), with a
week x 30deg cluster bootstrap for the difference vs a reference model. Also the record-exceedance slope for context.
usage: python src/analysis/rank_within_records.py RES(15|025) YEAR VAR KIND TRUTH ref models"""
import sys, os, numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
RES, Y, VAR, KIND, TR, REF = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
MODELS = sys.argv[7].split(','); s = 1 if KIND == 'max' else -1
rng = np.random.default_rng(0)
if RES == '15':
    from common import grid15 as C; ED = f'{C.D}/events'
else:
    from common import grid025 as C; ED = f'{C.D}/out'
def load(m): return np.load(f'{ED}/{Y}_{VAR}_{KIND}_{TR}_{m}.npz')
dR = load(REF); vt = dR['vt']; t = dR['t']
lon = C.LON[dR['i']] if RES == '15' else C.LON[dR['c']]; lat = C.LAT[dR['j']] if RES == '15' else C.LAT[dR['c']]
wk = vt[t].astype('datetime64[D]').astype(int) // 7
_, cl = np.unique(wk * 1000 + (lon // 30).astype(int) * 10 + ((lat + 90) // 30).astype(int), return_inverse=True)
def spear(a, b):
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    return (ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum())
e = s * (dR['y'] - dR['r'])
for L in dR['leads']:
    if L not in (12, 24, 48, 120, 168): continue
    li = list(dR['leads']).index(L)
    F = {m: s * (load(m)['f'][li] - dR['r']) for m in MODELS + [REF]}
    ok = np.all([np.isfinite(v) for v in F.values()], 0)
    K = cl[ok].max() + 1; _, cc = np.unique(cl[ok], return_inverse=True); K = cc.max() + 1
    idx = [np.where(cc == k)[0] for k in range(K)]
    base = {m: spear(F[m][ok], e[ok]) for m in F}
    row = []
    for m in MODELS:
        bs = []
        for _ in range(200):
            p = np.concatenate([idx[k] for k in rng.integers(0, K, K)])
            bs.append(spear(F[m][ok][p], e[ok][p]) - spear(F[REF][ok][p], e[ok][p]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sl = np.polyfit(e[ok], (F[m] - e)[ok], 1)[0]
        row.append(f'{m} {base[m]:.3f} (vs {REF} {base[m]-base[REF]:+.3f} [{lo:+.3f},{hi:+.3f}]; slope {sl:+.2f})')
    print(f'{RES} {Y} {VAR} {KIND} {TR} lead {L} N={ok.sum()}: {REF} {base[REF]:.3f} (slope {np.polyfit(e[ok], (F[REF]-e)[ok], 1)[0]:+.2f}) | ' + ' | '.join(row), flush=True)
