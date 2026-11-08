"""Monotone forward recalibration in LOCAL standardized units:
z_f = (f'-mu_f)/sd_f, z_y = (y'-mu_y)/sd_y with cross-fitted window moments (ordinary_days.moments); fit one pooled isotonic
g = E[z_y | z_f] on NON-record land cell-times of 4 week-folds, apply to the 5th (cross-fitting); recalibrated
f* = mu + mu_y + sd_y * g(z_f). Linear analogue: g(z) = rho*z (local conditional-mean recalibration).
Monotone maps add no information: if a model's extra record-exceedance slope disappears under g, it was calibration
(state-dependent amplitude), not missing knowledge. Same treatment for every model.
usage: python src/analysis/monotone_recalibration.py YEAR VAR KIND TRUTH models leads"""
import sys, os, numpy as np
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
import grid15, ordinary_days
Y, VAR, KIND, TR = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(','); LS = [int(x) for x in sys.argv[6].split(',')]
s = 1 if KIND == 'max' else -1
vt, y = grid15.era5_year(VAR, Y, src=TR)
mu, _ = grid15.clim_at(grid15.clim(VAR), vt)
r = grid15.rec_at(grid15.records(VAR, 1979, 2017, KIND), vt)
L3 = np.broadcast_to(grid15.LAND[None], y.shape)
ev = L3 & (s * (y - r) > 0)
fold = np.broadcast_to(((vt.astype('datetime64[D]').astype(int) // 7) % 5)[:, None, None], y.shape)
ya = y - mu
exc = s * (y - r)
print(f'{Y} {VAR} {KIND} truth={TR} events={ev.sum()}: record bias & record-exceedance slope  raw | linear cond-mean | isotonic cond-mean')
for L in LS:
    for name in MODELS:
        base, _, mem = name.partition(':')
        _, F = grid15.forecast_valid(base, VAR, Y, leads=[L], member=int(mem) if mem else None)
        fa = F[0] - mu
        m = ordinary_days.moments(ya, fa)
        sf, sy = np.sqrt(np.maximum(m['vf'], 1e-6)), np.sqrt(np.maximum(m['vy'], 1e-6))
        zf = (fa - m['mu_f']) / sf; zy = (ya - m['mu_y']) / sy
        rho = m['cv'] / (sf * sy)
        ok = L3 & np.isfinite(zf) & np.isfinite(zy) & (m['n'] > 50)
        g = np.full_like(zf, np.nan)
        for k in range(5):
            tr = ok & ~ev & (fold != k); te = ok & (fold == k)
            ir = IsotonicRegression(out_of_bounds='clip').fit(zf[tr][::3], zy[tr][::3])
            g[te] = ir.predict(zf[te])
        f_iso = m['mu_y'] + sy * g
        f_lin = m['mu_y'] + sy * rho * zf
        e = ev & ok & np.isfinite(g)
        out = []
        for tag, ff in (('raw', fa), ('lin', f_lin), ('iso', f_iso)):
            err = s * (ff - ya)[e]
            out.append(f'{err.mean():+.2f} / {np.polyfit(exc[e], err, 1)[0]:+.2f}')
        print(f' lead {L:3d} {name:12s} ' + ' | '.join(out) + f'   (N={e.sum()}, median rho {np.median(rho[e]):.3f})', flush=True)
        if os.environ.get('BINS'):
            bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)]
            for tag, ff in (('raw', fa), ('iso', f_iso)):
                err = s * (ff - ya)
                print(f'      {tag} by exceedance bin (K): ' + ' '.join(f'[{a},{b}):{err[e & (exc >= a) & (exc < b)].mean():+.2f}(n={(e & (exc >= a) & (exc < b)).sum()})' for a, b in bins), flush=True)
