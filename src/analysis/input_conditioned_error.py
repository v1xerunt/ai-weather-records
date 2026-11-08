"""Input-conditioned ceiling test (0.25 deg). Condition on the INITIAL state x0 (ERA5 at init, which the model sees),
not on the outcome: E[s(y-f) | u0], u0 = s(x0 - r_c)/sd. Because x0 is in the forecast's information set there is no
regression-to-the-mean artefact: an ideal forecast has E[y-f | x0] = const (local mean bias). A learned ceiling at r_c
predicts that s(y-f) jumps/rises for u0 > 0 (out-of-distribution inputs), at the model's own cutoff.
Accumulates by (lead, threshold[r2017,r2019], subset[all,unseen,seen], u0-bin): n, sum e, sum e^2, sum u0, sum s(y-x0).
usage: python src/analysis/input_conditioned_error.py YEAR VAR KIND models"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C

Y, VAR, KIND = int(sys.argv[1]), sys.argv[2], sys.argv[3]; MODELS = sys.argv[4].split(',')
s = 1 if KIND == 'max' else -1
od = f'{C.D}/input'; os.makedirs(od, exist_ok=True)
vt, y = C.truth_year(VAR, Y)
vprev, yprev = C.truth_year(VAR, Y - 1)
T = len(vt)
allt = np.concatenate([vprev, vt]); ally = np.concatenate([yprev, y])
mu, sd = C.clim_at(C.clim(VAR), allt)
mon = allt.astype('datetime64[M]').astype(int) % 12
R = {c: np.load(f'{C.D}/derived/rec_{VAR}_{KIND}_1979_{c}.npy') for c in (2017, 2019)}
EDGES = np.arange(-4, 3.0001, 0.1); nb = len(EDGES) - 1
for name in MODELS:
    fn = f'{od}/{Y}_{VAR}_{KIND}_{name}.npz'
    if os.path.exists(fn): continue
    leads = list(np.load(f'{C.D}/fc/{name}/leads.npy'))
    out = np.zeros((len(leads), 2, 3, nb, 5))
    for li, L in enumerate(leads):
        _, F = C.forecast_valid(name, VAR, Y, L)
        k = L // 12
        idx0 = np.arange(T) + len(vprev) - k          # init index into allt
        x0 = ally[idx0]; m0 = mon[idx0]; sd0 = sd[idx0]
        e = s * (y - F)
        g = s * (R[2019][m0] - R[2017][m0]) / sd0
        subs = [np.ones_like(g, bool), g == 0, g >= 0.3]
        dy = s * (y - x0)
        for ri, c in enumerate((2017, 2019)):
            u0 = s * (x0 - R[c][m0]) / sd0
            b = np.floor((u0 - EDGES[0]) / 0.1).astype(np.int64)
            ok = np.isfinite(e) & (b >= 0) & (b < nb)
            for si, sm in enumerate(subs):
                kk = ok & sm; bb = b[kk]
                out[li, ri, si, :, 0] = np.bincount(bb, minlength=nb)
                out[li, ri, si, :, 1] = np.bincount(bb, e[kk], minlength=nb)
                out[li, ri, si, :, 2] = np.bincount(bb, e[kk] ** 2, minlength=nb)
                out[li, ri, si, :, 3] = np.bincount(bb, u0[kk], minlength=nb)
                out[li, ri, si, :, 4] = np.bincount(bb, dy[kk], minlength=nb)
        print(name, L, 'done', flush=True)
    np.savez(fn, out=out, edges=EDGES, leads=np.array(leads))
