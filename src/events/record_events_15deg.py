"""Build record-event tables with the observed error and the expected error from ordinary days.
usage: python src/events/record_events_15deg.py YEAR VAR KIND TRUTH model1,model2,...   (TRUTH in era5|hres_t0)
Saves data/events/{YEAR}_{VAR}_{KIND}_{TRUTH}_{MODEL}.npz with per-event arrays (lead axis first)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15, ordinary_days

Y, VAR, KIND, TRUTH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(',')
REF1 = int(os.environ.get('REF1', 2017)); REF0 = int(os.environ.get('REF0', 1979))
TAG = os.environ.get('TAG', '')
od = f'{grid15.D}/events'; os.makedirs(od, exist_ok=True)
s = 1 if KIND == 'max' else -1

vt, y = grid15.era5_year(VAR, Y, src=TRUTH)
R = grid15.records(VAR, REF0, REF1, KIND)
r = grid15.rec_at(R, vt)
C = grid15.clim(VAR)
mu, sd = grid15.clim_at(C, vt)
ev = grid15.LAND[None] & ((y - r) * s > 0)
ti, ii, jj = np.where(ev)
print(Y, VAR, KIND, TRUTH, 'events', len(ti), flush=True)
ya = y - mu
R2019 = grid15.records(VAR, 1979, 2019, KIND)
base = dict(t=ti, i=ii, j=jj, y=y[ev], r=r[ev], mu=mu[ev], sd=sd[ev], r2019=grid15.rec_at(R2019, vt)[ev],
            leads=grid15.LEADS, vt=vt)

for name in MODELS:
    fn = f'{od}/{Y}_{VAR}_{KIND}_{TRUTH}_{name}{TAG}.npz'
    if os.path.exists(fn):
        continue
    base_name, _, mem = name.partition(':')
    _, F = grid15.forecast_valid(base_name, VAR, Y, member=int(mem) if mem else None)
    if np.isfinite(F).sum() == 0:
        print('no data', name); continue
    out = {k: [] for k in ('f', 'bhat', 'mse_hat', 'B', 'rho', 'a', 'n', 'mu_y', 'mu_f')}
    for li in range(len(grid15.LEADS)):
        fa = F[li] - mu
        m = ordinary_days.moments(ya, fa)
        bh, mh, B = ordinary_days.predict(m, ya)
        out['f'].append(F[li][ev]); out['bhat'].append(bh[ev]); out['mse_hat'].append(mh[ev]); out['B'].append(B[ev])
        out['rho'].append((m['cv'] / np.sqrt(m['vy'] * m['vf']))[ev]); out['a'].append(np.sqrt(m['vf'] / m['vy'])[ev])
        out['n'].append(m['n'][ev]); out['mu_y'].append(m['mu_y'][ev]); out['mu_f'].append(m['mu_f'][ev])
    np.savez(fn, **base, **{k: np.array(v, np.float32) for k, v in out.items()})
    f = np.array(out['f']); e = f - base['y'][None]
    print(name, 'lead24 bias %.2f rmse %.2f  ordinary_days bias %.2f rmse %.2f' % (
        np.nanmean(e[1]), np.sqrt(np.nanmean(e[1] ** 2)), np.nanmean(out['bhat'][1]), np.sqrt(np.nanmean(out['mse_hat'][1]))), flush=True)
