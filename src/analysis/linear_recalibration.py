"""Local (cross-fitted window) linear recalibration to the conditional mean: f* = mu_y + (rho/a)(f'-mu_f).
After it every forecast has the conditional-mean amplitude (a=rho) -> record-bias differences reflect information (rho) only.
Also 'inflated' version f+ = mu_y + (1/a)(f'-mu_f) (a=1, the MOS-inflation point).
usage: python src/analysis/linear_recalibration.py YEAR VAR KIND TRUTH models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
Y, VAR, KIND, TRUTH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(','); LS = [int(x) for x in sys.argv[6].split(',')]
s = 1 if KIND == 'max' else -1
print(f'{Y} {VAR} {KIND} truth={TRUTH}: record bias raw -> cond-mean recal (median rho) | inflated (a=1)')
for L in LS:
    for name in MODELS:
        fn = f'{grid15.D}/events/{Y}_{VAR}_{KIND}_{TRUTH}_{name}.npz'
        if not os.path.exists(fn): continue
        d = np.load(fn); li = int(np.where(d['leads'] == L)[0][0])
        fa = d['f'][li] - d['mu']; ya = d['y'] - d['mu']
        rho, a = d['rho'][li], d['a'][li]
        ok = np.isfinite(fa) & np.isfinite(rho) & (d['n'][li] > 50) & (a > 0)
        fs = d['mu_y'][li] + rho / a * (fa - d['mu_f'][li])
        fx = d['mu_y'][li] + 1 / a * (fa - d['mu_f'][li])
        b = lambda g: np.mean(s * (g - ya)[ok])
        print(f' lead {L:3d} {name:12s} {b(fa):+.2f} -> {b(fs):+.2f} (rho {np.median(rho[ok]):.3f}) | {b(fx):+.2f}')
