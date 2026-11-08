"""Decompose expected record bias into mean-offset (mu_f-mu_y) and regression ((B-1)(y'-mu_y)) terms.
usage: python src/analysis/offset_regression_split.py YEAR VAR KIND models leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
Y, VAR, KIND = int(sys.argv[1]), sys.argv[2], sys.argv[3]; MODELS = sys.argv[4].split(','); LS = [int(x) for x in sys.argv[5].split(',')]
s = 1 if KIND == 'max' else -1
print('truth model lead | obs | expected = offset + regression | median B rho a | mean anomaly of event y-mu_win (sd units)')
for TR in ('era5', 'hres_t0'):
    for name in MODELS:
        fn = f'{grid15.D}/events/{Y}_{VAR}_{KIND}_{TR}_{name}.npz'
        if not os.path.exists(fn): continue
        d = np.load(fn)
        for L in LS:
            li = int(np.where(d['leads'] == L)[0][0])
            ok = np.isfinite(d['f'][li]) & np.isfinite(d['bhat'][li]) & (d['n'][li] > 50)
            ya = d['y'] - d['mu']
            off = s * (d['mu_f'][li] - d['mu_y'][li]); reg = s * (d['B'][li] - 1) * (ya - d['mu_y'][li])
            obs = s * (d['f'][li] - d['y'])
            sdw = np.sqrt(np.maximum(0, 1)) 
            print(f'{TR:7s} {name:12s} {L:3d} | {obs[ok].mean():+.2f} | {(off+reg)[ok].mean():+.2f} = {off[ok].mean():+.2f} + {reg[ok].mean():+.2f} | '
                  f'{np.median(d["B"][li][ok]):.2f} {np.median(d["rho"][li][ok]):.2f} {np.median(d["a"][li][ok]):.2f} | {np.mean(s*(ya-d["mu_y"][li])[ok]):.2f} K')
