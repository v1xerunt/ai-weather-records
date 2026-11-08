"""Where does HRES beat AI against stations? Stratify station record events by representativeness:
|z_station - z_cell| (0.25deg nearest land cell) and sub-grid orography roughness (sd of 0.25deg land-cell heights within
75 km). Report mean record bias difference HRES - AI (positive = HRES better) and residual beyond the expected error per tercile.
usage: python src/analysis/station_stratify.py YEAR KIND"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common import grid025 as C
from scipy.spatial import cKDTree
Y, KIND = int(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
st = np.load(f'{ROOT}/data/metar/stations.npz', allow_pickle=True)
ev = np.load(f'{ROOT}/data/metar/events_{Y}_{KIND}.npz')
zc = C._st['z'] / 9.80665
def xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
tree = cKDTree(xyz(C.LAT, C.LON))
nb = tree.query_ball_point(xyz(st['lat'], st['lon']), 2 * np.sin(75 / 6371 / 2))
rough = np.array([np.std(zc[n]) if len(n) > 2 else np.nan for n in nb])
dz = np.abs(st['elev'] - zc[ev['i025']])
coast = np.array([len(n) for n in nb])          # fewer land cells nearby = coastal
for L in (12, 48, 120):
    k_ai, k_h = f'graphcast_25_{L}', f'hres_25_{L}'
    if k_ai not in ev or k_h not in ev: continue
    A, H = ev[k_ai], ev[k_h]
    key = lambda a: list(zip(a[2].astype(int), a[3].astype(int)))
    da = dict(zip(key(A), zip(A[0], A[1]))); dh = dict(zip(key(H), zip(H[0], H[1])))
    common = sorted(set(da) & set(dh))
    sidx = np.array([c[0] for c in common])
    ea = np.array([da[c][0] for c in common]); ga = np.array([da[c][1] for c in common])
    eh = np.array([dh[c][0] for c in common]); gh = np.array([dh[c][1] for c in common])
    print(f'== {Y} {KIND} lead {L}: N={len(common)}  (bias difference HRES-GraphCast; + = HRES better) | excess AI / HRES')
    for nm, v in (('|dz| station-cell', dz[sidx]), ('orog roughness', rough[sidx]), ('land cells within 75km', coast[sidx])):
        q = np.nanquantile(v, [1 / 3, 2 / 3]); g = np.digitize(v, q)
        row = ' | '.join(f'{(eh-ea)[g==j].mean():+.2f} ({(ea-ga)[g==j].mean():+.2f}/{(eh-gh)[g==j].mean():+.2f})' for j in range(3))
        print(f'   {nm:24s} terciles [{q[0]:.0f},{q[1]:.0f}]: {row}')
