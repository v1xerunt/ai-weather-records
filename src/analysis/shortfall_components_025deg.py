"""0.25 deg version of the offset / info(rho) / amplitude(a) split (per-cell +-15 d window, excl +-3 d, no pooling).
usage: python src/analysis/shortfall_components_025deg.py YEAR KIND TRUTH models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
from common.ordinary_days import _win
Y, KIND, TR = int(sys.argv[1]), sys.argv[2], sys.argv[3]; MODELS = sys.argv[4].split(','); LS = [int(x) for x in sys.argv[5].split(',')]
s = 1 if KIND == 'max' else -1
vt, y = C.truth_year('t2m', Y, src=TR); mu, _ = C.clim_at(C.clim('t2m'), vt)
mon = vt.astype('datetime64[M]').astype(int) % 12
r = C.records('t2m', 1979, 2017, KIND)[mon]; ev = s * (y - r) > 0
ya = (y - mu).astype(np.float64)
for L in LS:
    for m in MODELS:
        _, F = C.forecast_valid(m, 't2m', Y, L); fa = (F - mu).astype(np.float64)
        M = np.isfinite(fa) & np.isfinite(ya); z = lambda a: np.where(M, a, 0.0)
        S = {k: _win(v) for k, v in dict(n=M.astype(float), y=z(ya), f=z(fa), yy=z(ya)**2, ff=z(fa)**2, fy=z(fa)*z(ya)).items()}
        n = np.maximum(S['n'], 1); my, mf = S['y']/n, S['f']/n
        vy, vf, cv = S['yy']/n - my**2, S['ff']/n - mf**2, S['fy']/n - mf*my
        rho = cv/np.sqrt(vy*vf); a = np.sqrt(vf/vy)
        ok = ev & M & (S['n'] > 20) & np.isfinite(rho)
        D = (ya - my)[ok]; off = s*(mf-my)[ok]; info = -s*(1-rho[ok])*D; amp = -s*rho[ok]*(1-a[ok])*D
        obs = s*(fa-ya)[ok]
        print(f'{Y} {KIND} {TR} lead {L:3d} {m:10s} obs {obs.mean():+.2f} | pred {off.mean()+info.mean()+amp.mean():+.2f} = off {off.mean():+.2f} info {info.mean():+.2f} amp {amp.mean():+.2f} | rho {np.median(rho[ok]):.3f} a {np.median(a[ok]):.3f}', flush=True)
