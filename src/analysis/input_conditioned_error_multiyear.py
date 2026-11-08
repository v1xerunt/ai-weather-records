"""Input-conditioned OOD test pooled over years (1.5 deg). u0 = s(x0 - r_c)/sd with r_c = the model's own training-period
record (default 1979-2017). Bins up to +4 sigma; per-year counts so that single events (e.g. PNW 2021) can be traced.
usage: python src/analysis/input_conditioned_error_multiyear.py VAR KIND models years leads [cut]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
VAR, KIND = sys.argv[1], sys.argv[2]; MODELS = sys.argv[3].split(','); YEARS = [int(x) for x in sys.argv[4].split(',')]
LEADS = [int(x) for x in sys.argv[5].split(',')]; CUT = int(sys.argv[6]) if len(sys.argv) > 6 else 2017
s = 1 if KIND == 'max' else -1
R = grid15.records(VAR, 1979, CUT, KIND); C = grid15.clim(VAR)
ED = np.arange(-3, 4.51, 0.25); nb = len(ED) - 1
res = {}
for name in MODELS:
    acc = np.zeros((len(LEADS), len(YEARS), nb, 3))
    for yi, Y in enumerate(YEARS):
        vt, y = grid15.era5_year(VAR, Y); vp, yp = grid15.era5_year(VAR, Y - 1)
        allt = np.concatenate([vp, vt]); ally = np.concatenate([yp, y])
        _, sd = grid15.clim_at(C, allt); mon = allt.astype('datetime64[M]').astype(int) % 12
        _, F = grid15.forecast_valid(name, VAR, Y, leads=LEADS)
        for li, L in enumerate(LEADS):
            idx = np.arange(len(vt)) + len(vp) - L // 12
            u0 = s * (ally[idx] - R[mon[idx]]) / sd[idx]
            e = s * (y - F[li])
            b = np.floor((np.nan_to_num(u0, nan=-99) - ED[0]) / 0.25).astype(np.int64)
            k = np.isfinite(e) & (b >= 0) & (b < nb) & grid15.LAND[None]
            acc[li, yi, :, 0] = np.bincount(b[k], minlength=nb); acc[li, yi, :, 1] = np.bincount(b[k], e[k], minlength=nb)
            acc[li, yi, :, 2] = np.bincount(b[k], e[k] ** 2, minlength=nb)
        print(name, Y, flush=True)
    res[name] = acc
np.savez(f'{grid15.D}/derived/e7multi_{VAR}_{KIND}_{CUT}.npz', edges=ED, leads=np.array(LEADS), years=np.array(YEARS), **res)
c = 0.5 * (ED[:-1] + ED[1:])
for li, L in enumerate(LEADS):
    print(f'== {VAR} {KIND} lead {L}: mean s(y-f) by u0 (input relative to 1979-{CUT} record), years {YEARS}')
    sel = [i for i in range(nb) if c[i] >= -1.5]
    print('u0      ' + ' '.join(f'{c[i]:+6.2f}' for i in sel))
    for name in MODELS:
        a = res[name][li].sum(0)
        print(f'{name:7s} ' + ' '.join((f'{a[i,1]/a[i,0]:+6.2f}' if a[i, 0] >= 20 else '     .') for i in sel))
    a = res[MODELS[0]][li]
    print('n       ' + ' '.join(f'{int(a[:, i, 0].sum()):6d}' for i in sel))
