"""Information content (1.5 deg): rank-based discrimination of threshold exceedance, invariant to any common
monotone recalibration of the standardized score u=(f-r)/sd.  Thresholds: record r2017 and sub-record placebos r-1sd, r-2sd.
Reports AUC and TPR at FPR in {1e-3, 1e-2}.  Cluster bootstrap on AUC differences vs a reference model.
usage: python src/analysis/deep_tail_detection.py YEAR VAR KIND TRUTH models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15

Y, VAR, KIND, TRUTH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
MODELS = sys.argv[5].split(','); LEADS = [int(x) for x in sys.argv[6].split(',')]
s = 1 if KIND == 'max' else -1
vt, y = grid15.era5_year(VAR, Y, src=TRUTH)
_, sd = grid15.clim_at(grid15.clim(VAR), vt)
r = grid15.rec_at(grid15.records(VAR, 1979, 2017, KIND), vt)
L = np.broadcast_to(grid15.LAND[None], y.shape)


def auc_tpr(score, lab):
    o = np.argsort(-score, kind='stable'); l = lab[o]
    P = l.sum(); N = len(l) - P
    tp = np.cumsum(l); fp = np.cumsum(~l)
    auc = (tp[~l].sum()) / (P * N)
    out = [auc]
    for a in (1e-3, 1e-2):
        k = np.searchsorted(fp, a * N); out.append(tp[min(k, len(tp) - 1)] / P)
    return out, P


print(f'{Y} {VAR} {KIND} truth={TRUTH}: AUC / TPR@FPR1e-3 / TPR@1e-2   (P = positives)')
OFFS = [float(x) for x in os.environ.get('OFFS', '0,1,2').split(',')]
for off in OFFS:
    thr = r - s * off * sd
    lab_full = s * (y - thr) > 0
    for li, Ld in enumerate(LEADS):
        line = []
        for name in MODELS:
            base_name, _, mem = name.partition(':')
            _, F = grid15.forecast_valid(base_name, VAR, Y, leads=[Ld], member=int(mem) if mem else None)
            f = F[0]
            ok = L & np.isfinite(f)
            sc = (s * (f - thr) / sd)[ok]; lab = lab_full[ok]
            (a, t3, t2), P = auc_tpr(sc, lab)
            line.append(f'{name}={a:.4f}/{t3:.3f}/{t2:.3f}')
        print(f' thr=r-({off})sd lead={Ld:3d} P={P}: ' + '  '.join(line), flush=True)
