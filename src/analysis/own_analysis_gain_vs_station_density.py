"""Where does verifying on a model's own analysis pay off? Per 1.5deg land cell (2020, t2m, 00/12 UTC, all days):
  gain_HRES = corr(HRES_L, HRES-fc0) - corr(HRES_L, ERA5)     (HRES scored on its own analysis vs on ERA5)
  gain_AI   = corr(AI_L, ERA5)      - corr(AI_L, HRES-fc0)    (ERA5-initialised AI scored on ERA5 vs on HRES-fc0)
  analysis disagreement = sd(HRES-fc0 - ERA5) anomalies
regressed on METAR station density (stations within 300 km, 2018-2020 reporting stations).
Hypothesis (cycled data assimilation): where observations are sparse the analysis ~ background ~ own short forecast,
so the gain from verifying on the model's own analysis concentrates in data-sparse regions. usage: python src/analysis/own_analysis_gain_vs_station_density.py leads"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid15
from scipy.spatial import cKDTree
LS = [int(x) for x in sys.argv[1].split(',')]
Y = 2020
vt, ye = grid15.era5_year('t2m', Y); _, yh = grid15.era5_year('t2m', Y, src='hres_t0')
mu, _ = grid15.clim_at(grid15.clim('t2m'), vt)
st = np.load(f'{grid15.ROOT}/data/metar/stations.npz', allow_pickle=True)
def xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
li, lj = np.where(grid15.LAND)
tree = cKDTree(xyz(st['lat'], st['lon']))
chord = 2 * np.sin(300 / 6371 / 2)
dens = np.array([len(x) for x in tree.query_ball_point(xyz(grid15.LAT[lj], grid15.LON[li]), chord)])
def corr(a, b):
    a = a - np.nanmean(a, 0); b = b - np.nanmean(b, 0)
    return np.nansum(a * b, 0) / np.sqrt(np.nansum(a * a, 0) * np.nansum(b * b, 0))
E = (ye - mu)[:, li, lj]; H = (yh - mu)[:, li, lj]
dis = np.nanstd(H - E, 0)
bins = [0, 1, 3, 10, 30, 1000]
g = np.digitize(dens, bins) - 1
print('stations within 300 km: bins', bins, 'cells per bin', np.bincount(g, minlength=len(bins) - 1))
print('analysis disagreement sd(HRES-fc0 - ERA5) by density bin:', ' '.join(f'{np.nanmedian(dis[g==k]):.2f}' for k in range(len(bins) - 1)))
for L in LS:
    for name in ('hres', 'graphcast', 'pangu', 'era5fc'):
        _, F = grid15.forecast_valid(name, 't2m', Y, leads=[L])
        Fa = (F[0] - mu)[:, li, lj]
        ok = np.isfinite(Fa) & np.isfinite(E) & np.isfinite(H)
        Fa = np.where(ok, Fa, np.nan); Ee = np.where(ok, E, np.nan); Hh = np.where(ok, H, np.nan)
        gain = corr(Fa, Hh) - corr(Fa, Ee)          # + = closer to HRES-fc0
        print(f' lead {L:3d} {name:9s} corr(f,HRES-fc0)-corr(f,ERA5) by density bin: ' +
              ' '.join(f'{np.nanmedian(gain[g==k]):+.3f}' for k in range(len(bins) - 1)), flush=True)
