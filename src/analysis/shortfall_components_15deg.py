"""Split the expected record bias into three additive parts (per event, then averaged):
  bias_hat = offset + (rho*a - 1)*D,  D = y' - mu_y (event anomaly vs ordinary-day window mean)
  offset   = mu_f - mu_y                         (mean bias in the window)
  info     = -(1 - rho) * D                      (unpredictable part: even an a=1 forecast with this rho has it)
  amp      = -rho * (1 - a) * D                  (amplitude / smoothing: a<1 shrinks the forecast)
info + amp = (rho*a - 1) D exactly. Reference points: ideal conditional mean has a=rho; a single realisation a=1.
usage: python src/analysis/shortfall_components_15deg.py YEAR VAR KIND TRUTH models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
Y, VAR, KIND, TR = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(','); LS = [int(x) for x in sys.argv[6].split(',')]
s = 1 if KIND == 'max' else -1
print(f'{Y} {VAR} {KIND} truth={TR}: observed | predicted = offset + info(rho) + amp(a) | median rho, a | amp share of (info+amp)')
for L in LS:
    for m in MODELS:
        fn = f'{grid15.D}/events/{Y}_{VAR}_{KIND}_{TR}_{m}.npz'
        if not os.path.exists(fn): continue
        d = np.load(fn); li = list(d['leads']).index(L)
        rho, a = d['rho'][li], d['a'][li]
        ok = np.isfinite(d['f'][li]) & np.isfinite(rho) & np.isfinite(a) & (d['n'][li] > 50)
        D = (d['y'] - d['mu']) - d['mu_y'][li]
        off = s * (d['mu_f'][li] - d['mu_y'][li]); info = -s * (1 - rho) * D; amp = -s * rho * (1 - a) * D
        obs = s * (d['f'][li] - d['y'])
        o, i_, a_ = off[ok].mean(), info[ok].mean(), amp[ok].mean()
        print(f' lead {L:3d} {m:12s} {obs[ok].mean():+.2f} | {o+i_+a_:+.2f} = {o:+.2f} {i_:+.2f} {a_:+.2f} | rho {np.median(rho[ok]):.3f} a {np.median(a[ok]):.3f} | amp share {a_/(i_+a_):.0%}')
