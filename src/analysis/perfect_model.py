"""Perfect-model experiment (1.5 deg): truth = ensemble member m, forecast = mean of the other members of the same
ensemble (an honest forecast drawn from the exact same model climate, no AI/physics mismatch, no shared lineage between forecast and truth).
Records: member exceeds ERA5 1979-2017 monthly record. Reports observed record bias vs expected error, and the same
for a single other member used as forecast (member paradox in a perfect model).
usage: python src/analysis/perfect_model.py ENS(ens10|gencast10) VAR KIND leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15, ordinary_days
ENS, VAR, KIND = sys.argv[1], sys.argv[2], sys.argv[3]; LEADS = [int(x) for x in sys.argv[4].split(',')]
s = 1 if KIND == 'max' else -1; Y = 2020
vt = np.arange(np.datetime64(f'{Y}-01-01T00'), np.datetime64(f'{Y+1}-01-01T00'), np.timedelta64(12, 'h'))
mu, sd = grid15.clim_at(grid15.clim(VAR), vt)
r = grid15.rec_at(grid15.records(VAR, 1979, 2017, KIND), vt)
for L in LEADS:
    Ms = [grid15.forecast_valid(ENS, VAR, Y, leads=[L], member=m)[1][0] for m in range(10)]
    S = np.nansum(Ms, 0)
    rows = []
    for m in range(int(os.environ.get("NTRUTH", 4))):     # members as truth
        y = Ms[m]; fbar = (S - y) / 9; fone = Ms[(m + 1) % 10]
        ev = grid15.LAND[None] & (s * (y - r) > 0) & np.isfinite(fbar)
        ya = y - mu
        for tag, f in (('mean9', fbar), ('member', fone)):
            mm = ordinary_days.moments(ya, f - mu)
            bh, _, B = ordinary_days.predict(mm, ya)
            ok = ev & (mm['n'] > 50)
            rows.append((m, tag, ok.sum(), np.mean(s * (f - y)[ok]), np.mean(s * bh[ok]), np.median(B[ok]),
                         np.median((mm['cv'] / np.sqrt(mm['vy'] * mm['vf']))[ok])))
    print(f'== {ENS} {VAR} {KIND} lead {L}: truth=member m; forecast = mean of other 9 | single other member')
    for row in rows:
        print('  truth m%d %-6s N=%5d  obs bias %+.2f  expected %+.2f  (B %.2f rho %.2f)' % row, flush=True)
