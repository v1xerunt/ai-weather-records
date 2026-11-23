"""Spread-skill ratio of the GenCast and IFS ensembles (1.5 deg, 2020, land, ERA5 truth).
SSR = sqrt((M+1)/M * mean ensemble variance) / RMSE of ensemble mean; 1 = reliable, <1 under-, >1 over-dispersive.
Reported for all valid times and for situations the ensemble mean forecasts as extreme (mean anomaly beyond 2 local s.d.),
which is conditioning on the forecast and therefore free of outcome selection.
usage: python src/analysis/spread_skill.py VAR leads   (e.g. t2m 12,24,48,120,240)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
VAR = sys.argv[1]; LEADS = [int(x) for x in sys.argv[2].split(',')]; Y = 2020
vt = np.arange(np.datetime64(f'{Y}-01-01T00'), np.datetime64(f'{Y+1}-01-01T00'), np.timedelta64(12, 'h'))
mu, sd = grid15.clim_at(grid15.clim(VAR), vt)
_, y = grid15.era5_year(VAR, Y)
for ENS in ('gencast10', 'ens10'):
    for L in LEADS:
        Ms = np.stack([grid15.forecast_valid(ENS, VAR, Y, leads=[L], member=m)[1][0] for m in range(10)])
        M = Ms.shape[0]; fm = Ms.mean(0); var = Ms.var(0, ddof=1)
        z = (fm - mu) / sd
        out = []
        for tag, sel in (('all', np.ones_like(z, bool)), ('mean>+2sd', z > 2), ('mean<-2sd', z < -2)):
            ok = grid15.LAND[None] & np.isfinite(fm) & np.isfinite(y) & sel
            rmse = np.sqrt(np.mean((fm - y)[ok] ** 2)); spr = np.sqrt((M + 1) / M * np.mean(var[ok]))
            out.append('%s n=%d RMSE %.2f spread %.2f SSR %.2f' % (tag, ok.sum(), rmse, spr, spr / rmse))
        print(f'{ENS:9s} {VAR} {L:3d}h | ' + ' | '.join(out), flush=True)
