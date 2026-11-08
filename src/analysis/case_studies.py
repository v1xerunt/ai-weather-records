"""Case studies from event tables: events inside a lat/lon box and time window. Reports per model and lead the mean
record exceedance (truth), forecast exceedance, observed bias and expected error from ordinary days.
usage: python src/analysis/case_studies.py YEAR TRUTH models lat0 lat1 lon0 lon1 t0 t1"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
Y, TR = int(sys.argv[1]), sys.argv[2]; MODELS = sys.argv[3].split(',')
la0, la1, lo0, lo1 = map(float, sys.argv[4:8]); t0, t1 = np.datetime64(sys.argv[8]), np.datetime64(sys.argv[9])
for m in MODELS:
    fn = f'{grid15.D}/events/{Y}_t2m_max_{TR}_{m}.npz'
    if not os.path.exists(fn): continue
    d = np.load(fn); vt = d['vt'][d['t']]; lat = grid15.LAT[d['j']]; lon = grid15.LON[d['i']]
    lo = lon if lo0 <= lo1 else np.where(lon > 180, lon - 360, lon)
    a0, a1 = (lo0, lo1) if lo0 <= lo1 else (lo0 - 360, lo1)
    k = (lat >= la0) & (lat <= la1) & (lo >= a0) & (lo <= a1) & (vt >= t0) & (vt <= t1)
    if m == MODELS[0]:
        print(f'{Y} box events: {k.sum()}, mean truth exceedance {np.mean(d["y"][k]-d["r"][k]):.2f} K, max {np.max(d["y"][k]-d["r"][k]):.1f} K, '
              f'mean local z {np.mean((d["y"][k]-d["mu"][k])/d["sd"][k]):.1f}')
    row = []
    for L in (24, 48, 72, 120, 168):
        li = list(d['leads']).index(L); f = d['f'][li][k]; ok = np.isfinite(f)
        row.append(f'{L}h: fc-exc {np.mean((f-d["r"][k])[ok]):+.1f} bias {np.mean((f-d["y"][k])[ok]):+.2f} / expected {np.mean(d["bhat"][li][k][ok]):+.2f}')
    print(f'  {m:9s} ' + ' | '.join(row))
