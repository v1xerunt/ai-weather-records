"""Training-cutoff natural experiment (difference-in-differences) on record events.
Model A has seen cutoff year cA, model B cB < cA. 'seen' cell-months: record at cA exceeds record at cB by >= G sd.
Ceiling hypothesis: in seen cells, B is capped at r_cB, A at r_cA > r_cB  ->  A gains relative to B in seen vs unseen cells.
Statistics (per lead): DiD of paired error  d = s(f_A - f_B)  and of residual beyond the expected error  x = s(f-y) - s*bhat.
usage: python src/analysis/cutoff_cross_architecture.py RES(15|025) YEAR VAR KIND TRUTH A:cA B:cB [G]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
RES, Y, VAR, KIND, TRUTH = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
(A, cA), (Bm, cB) = [x.split(':') for x in sys.argv[6:8]]
G = float(sys.argv[8]) if len(sys.argv) > 8 else 0.3
s = 1 if KIND == 'max' else -1
rng = np.random.default_rng(0)
if RES == '15':
    from common import grid15 as C; ED = f'{C.D}/events'
else:
    from common import grid025 as C; ED = f'{C.D}/out'


def load(name):
    return np.load(f'{ED}/{Y}_{VAR}_{KIND}_{TRUTH}_{name}.npz')


dA, dB = load(A), load(Bm)
vt = dA['vt']; t = dA['t']
mon = vt[t].astype('datetime64[M]').astype(int) % 12
if RES == '15':
    cell = (dA['i'], dA['j']); R = lambda c: C.records(VAR, 1979, int(c), KIND)[mon, cell[0], cell[1]]
    lon, lat = C.LON[dA['i']], C.LAT[dA['j']]
else:
    cell = dA['c']; R = lambda c: np.load(f'{C.D}/derived/rec_{VAR}_{KIND}_1979_{int(c)}.npy')[mon, cell]
    lon, lat = C.LON[cell], C.LAT[cell]
rA, rB = R(cA), R(cB)
gap = s * (rA - rB) / dA['sd']
y = dA['y']
grp = np.full(len(y), -1)
grp[gap == 0] = 0                                   # unseen by both
grp[(gap >= G) & (s * (y - rA) <= 0)] = 1           # seen: event lies inside the gap (A trained on such values, B not)
grp[(gap >= G) & (s * (y - rA) > 0)] = 2            # seen: event beyond A's record too
wk = vt[t].astype('datetime64[D]').astype(int) // 7
_, clu = np.unique(wk * 1000 + (lon // 30).astype(int) * 10 + ((lat + 90) // 30).astype(int), return_inverse=True)
print(f'{RES} {Y} {VAR} {KIND} truth={TRUTH}  A={A}(<= {cA}) B={Bm}(<= {cB})  gap>={G}sd  n unseen/gap/beyond =',
      [(grp == g).sum() for g in (0, 1, 2)])


def did(v, g1):
    m0, m1 = grp == 0, grp == g1
    est = np.nanmean(v[m1]) - np.nanmean(v[m0])
    K = clu.max() + 1; bs = []
    for _ in range(300):
        w = np.bincount(rng.integers(0, K, K), minlength=K)[clu]
        bs.append(np.nansum(w * m1 * v) / np.sum(w * m1 * np.isfinite(v)) - np.nansum(w * m0 * v) / np.sum(w * m0 * np.isfinite(v)))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return f'{est:+.2f} [{lo:+.2f},{hi:+.2f}]'


for li, L in enumerate(dA['leads']):
    if L not in (12, 24, 48, 120): continue
    fA, fB = dA['f'][li], dB['f'][li]
    d = s * (fA - fB)
    xA = s * (fA - y) - s * dA['bhat'][li]; xB = s * (fB - y) - s * dB['bhat'][li]
    ok = (dA['n'][li] > 20) & (dB['n'][li] > 20)
    d = np.where(ok, d, np.nan); dx = np.where(ok, xA - xB, np.nan)
    print(f' lead {L:3d}: paired d(A-B) unseen {np.nanmean(d[grp==0]):+.2f} | DiD gap {did(d,1)} beyond {did(d,2)} '
          f'|| residual DiD gap {did(dx,1)} beyond {did(dx,2)}', flush=True)
