"""Fig. 3: the verifying analysis decides the ranking."""
from style import *
import numpy as np
from common import grid15
M = [('graphcast', 'GraphCast', AI), ('pangu', 'Pangu-Weather', AI), ('fuxi', 'FuXi', AI), ('gencastmean', 'GenCast mean', ENS),
     ('era5fc', 'IFS from ERA5', PHYS2), ('ensmean', 'IFS ENS mean', PHYS2), ('hres', 'HRES', PHYS)]
L = 12
st = np.load(f'{grid15.ROOT}/data/metar/events_2020_max.npz') if hasattr(grid15, 'ROOT') else None
st = np.load(f'{ROOT}/data/metar/events_2020_max.npz')
vals = {}
for m, lab, _ in M:
    for tr in ('era5', 'hres_t0'):
        d = np.load(f'{grid15.D}/events/2020_t2m_max_{tr}_{m}.npz'); li = list(d['leads']).index(L)
        e = d['f'][li] - d['y']; ok = np.isfinite(e) & np.isfinite(d['bhat'][li]) & (d['n'][li] > 50); vals[(m, tr)] = e[ok].mean()
    vals[(m, 'st')] = st[f'{m}_15_{L}'][0].mean()
fig = plt.figure(figsize=(W2, 66 * MM))
G = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.95, 1.0], wspace=0.42, left=0.115, right=0.985, top=0.84, bottom=0.2)
ga = G[0].subgridspec(1, 3, wspace=0.18)
gs = [ga[0], ga[1], ga[2], G[1], G[2]]
TRS = [('era5', 'ERA5 reanalysis'), ('hres_t0', 'HRES analysis'), ('st', 'Weather stations')]
axa = []
for j, (tr, tl) in enumerate(TRS):
    ax = fig.add_subplot(gs[j]); grid(ax, 'x'); axa.append(ax)
    for i, (m, lab, col) in enumerate(M):
        v = vals[(m, tr)]; hl = m in ('hres', 'era5fc', 'ensmean')
        ax.plot([0, v], [i, i], color=col, lw=0.8, alpha=.5)
        ax.plot(v, i, 'o', ms=4.2, color=col, mec='white', mew=0.5)
    ax.set_ylim(len(M) - 0.5, -0.6); ax.axvline(0, color=INK2, lw=0.5)
    lo = min(vals[(m, tr)] for m, *_ in M); ax.set_xlim(lo * 1.18, 0.05 * -lo)
    ax.set_title(tl, fontsize=6.5, loc='center'); ax.tick_params(axis='y', length=0)
    if j == 0: ax.set_yticks(range(len(M))); ax.set_yticklabels([l for _, l, _ in M])
    else: ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    best = max(M, key=lambda x: vals[(x[0], tr)]); ib = M.index(best)
    ax.text(vals[(best[0], tr)], ib - 0.42, 'best', ha='center', fontsize=5, color=INK2, style='italic')
axa[1].set_xlabel('Mean error on 2020 heat records, 12 h ahead (K)')
label(axa[0], 'a', x=-0.95, y=1.08)
# b decomposition of the headline gap (0.25 deg, 12 h)
ax = fig.add_subplot(gs[3]); grid(ax, 'y')
W_ = 0.5
bars = [('GraphCast\nvs ERA5', -0.74, AI), ('HRES\nvs ERA5', -1.48, PHYS), ('HRES vs\nits own\nanalysis', -0.20, PHYS)]
for k, (lab, v, c) in enumerate(bars):
    ax.bar(k, v, width=W_, color=c, alpha=1 if k != 2 else 0.45, edgecolor=c, lw=0.6)
    ax.text(k, v - 0.04, f'{v:+.2f}', ha='center', va='top', fontsize=6)
for x0, x1, v0, v1, t in [(0, 1, -0.74, -1.48, 'same truth\n0.74 K'), (1, 2, -1.48, -0.20, 'truth\nswitched\n1.28 K')]:
    ax.plot([x0 + 0.25, x1 - 0.25], [v0, v0], color=INK2, lw=0.4, ls=(0, (2, 1.5)))
    xm = (x0 + x1) / 2
    ax.annotate('', xy=(xm, v1), xytext=(xm, v0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.plot([x1 - 0.28, xm], [v1, v1], color=INK2, lw=0.4, ls=(0, (2, 1.5))) if False else None
ax.text(0.0, -0.95, 'same truth,\nAI ahead\nby 0.74 K', ha='center', va='top', fontsize=5.5, color=INK2)
ax.text(2.0, -0.5, 'switching\nthe truth\n+1.28 K', ha='center', va='top', fontsize=5.5, color=INK2)
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=5.5); ax.tick_params(axis='x', length=0)
ax.set_ylim(-2.05, 0.1); ax.set_xlim(-0.55, 2.4); ax.set_ylabel('Mean error on heat records (K)')
ax.set_title('Anatomy of a 0.54 K gap', fontsize=7); label(ax, 'b', x=-0.42, y=1.08)

# c home-analysis gain vs station density
ax = fig.add_subplot(gs[4]); grid(ax, 'y')
cats = ['0', '1–2', '3–9', '10–29', '≥30']
g12 = [0.061, 0.048, 0.029, 0.025, 0.023]; g120 = [0.049, 0.045, 0.030, 0.024, 0.023]
ax.plot(range(5), g12, color=PHYS, marker='o', ms=3.5, mec='white', mew=0.5, label='12 h')
ax.plot(range(5), g120, color=PHYS, marker='o', ms=3.5, mec='white', mew=0.5, ls=(0, (3, 1.5)), alpha=.6, label='120 h')
ax.set_xticks(range(5)); ax.set_xticklabels(cats); ax.set_ylim(0, 0.07)
ax.set_xlabel('Stations within 300 km'); ax.set_ylabel('Correlation gain of HRES when\nverified on its own analysis')
ax.legend(loc='upper right'); ax.set_title('Largest where stations are few', fontsize=7); label(ax, 'c', x=-0.4, y=1.08)
save(fig, 'fig3')
for m, *_ in M: print(m, [round(float(vals[(m, t)]), 2) for t in ('era5', 'hres_t0', 'st')])
