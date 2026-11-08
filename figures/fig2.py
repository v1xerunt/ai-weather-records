"""Fig. 2: expected error on records from ordinary days. a schematic, b exceedance coordinates (0.25 deg, 48 h), c collapse (1.5 deg), d predictability limit."""
from style import *
import numpy as np
from common import grid15, grid025
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(W2, 118 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.42, left=0.065, right=0.985, top=0.93, bottom=0.09)

# a  schematic
ax = fig.add_subplot(gs[0, 0]); rng = np.random.default_rng(3)
rho, n = 0.85, 6000
y = rng.standard_normal(n); f = rho * y + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
r = 2.2; rec = y > r
ax.scatter(y[~rec], f[~rec], s=1.2, color='#d2d2d2', lw=0, rasterized=True)
ax.scatter(y[rec], f[rec], s=5, color=INK, lw=0, zorder=3)
xx = np.linspace(-3.6, 3.6, 2); ax.plot(xx, xx, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.plot(xx, rho * xx, color=AI, lw=1.1)
ax.axvline(r, color=INK2, lw=0.6)
ym, fm = y[rec].mean(), f[rec].mean()
ax.annotate('', xy=(ym, fm), xytext=(ym, ym), arrowprops=dict(arrowstyle='-|>', color=PHYS, lw=0.8, mutation_scale=6))
ax.plot([ym], [ym], 'o', ms=3.4, color=PHYS, mfc='white', mew=0.8, zorder=4); ax.plot([ym], [fm], 'o', ms=3.4, color=PHYS, mec='white', mew=0.5, zorder=4)
ax.annotate('expected shortfall\n$(1-\\rho a)\\,\\Delta$', xy=(ym + 0.05, (ym + fm) / 2), xytext=(2.45, -0.9), fontsize=6, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', color=INK2, lw=0.4))
ax.text(3.15, 1.0, 'records', fontsize=6, color=INK, ha='left')
ax.text(r - 0.08, -3.3, 'record', rotation=90, ha='right', va='bottom', fontsize=6, color=INK2)
ax.text(-3.4, 3.1, 'perfect agreement', fontsize=6, color=INK2)
ax.text(-3.4, 2.75, 'best forecast given\nwhat is predictable', fontsize=6, color=AI, va='top')
ax.set_xlim(-3.6, 4.6); ax.set_ylim(-3.6, 3.6)
ax.set_xlabel('Observed anomaly (s.d.)'); ax.set_ylabel('Forecast anomaly (s.d.)')
ax.set_title('Selecting on the outcome', fontsize=7); label(ax, 'a')

# b  record-exceedance coordinates, 0.25 deg, 48 h, ERA5
ax = fig.add_subplot(gs[0, 1])
edges = [0, 0.5, 1, 2, 3, 6]; xc = np.arange(len(edges) - 1)
for k, (m, col, lab) in enumerate([('graphcast', AI, 'GraphCast'), ('hres', PHYS, 'HRES')]):
    d = np.load(f'{grid025.D}/out/2020_t2m_max_era5_{m}.npz')
    li = list(d['leads']).index(48); e = d['f'][li] - d['y']; bh = d['bhat'][li]; ex = d['y'] - d['r']
    ok = np.isfinite(e) & np.isfinite(bh)
    mo = [e[ok & (ex >= a_) & (ex < b_)].mean() for a_, b_ in zip(edges[:-1], edges[1:])]
    mb = [bh[ok & (ex >= a_) & (ex < b_)].mean() for a_, b_ in zip(edges[:-1], edges[1:])]
    off = (k - 0.5) * 0.12
    ax.plot(xc + off, mb, color=col, lw=1.0, alpha=.9)
    ax.plot(xc + off, mo, 'o', ms=3.4, color=col, mec='white', mew=0.5, zorder=3)
ax.set_xticks(xc); ax.set_xticklabels(['0–0.5', '0.5–1', '1–2', '2–3', '>3'])
ax.set_xlabel('Amount by which the record is broken (K)'); ax.set_ylabel('Mean forecast error on records (K)')
ax.axhline(0, color=INK2, lw=0.5)
h = [Line2D([], [], color=AI, marker='o', ms=3, lw=0, label='GraphCast (AI)'), Line2D([], [], color=PHYS, marker='o', ms=3, lw=0, label='HRES (physics)'),
     Line2D([], [], color=INK2, lw=1, label='expected from ordinary days')]
ax.legend(handles=h, loc='lower left', handlelength=1.5)
ax.set_title('The apparent cap, 2 days ahead', fontsize=7); label(ax, 'b')

# c  collapse, 1.5 deg, all systems
ax = fig.add_subplot(gs[0, 2])
GROUPS = [('AI, deterministic', ['graphcast', 'pangu', 'fuxi'], AI), ('AI, ensemble', ['gencastmean', 'gencast10:0', 'gencast10:1'], ENS),
          ('Physics, deterministic', ['hres', 'era5fc'], PHYS), ('Physics, ensemble', ['ensmean', 'ens10:0', 'ens10:1'], PHYS2)]
MK = {('t2m', 'max'): 'o', ('t2m', 'min'): 's', ('ws', 'max'): '^'}
X, Y = [], []
for TR in ('era5', 'hres_t0'):
    for g, ms, col in GROUPS:
        for (V, K), mk in MK.items():
            s = 1 if K == 'max' else -1; xs, ys = [], []
            for m in ms:
                fn = f'{grid15.D}/events/2020_{V}_{K}_{TR}_{m}.npz'
                if not os.path.exists(fn): continue
                d = np.load(fn)
                for li in range(len(d['leads'])):
                    err = s * (d['f'][li] - d['y']); bh = s * d['bhat'][li]
                    ok = np.isfinite(err) & np.isfinite(bh) & (d['n'][li] > 50)
                    xs.append(bh[ok].mean()); ys.append(err[ok].mean())
            ax.scatter(xs, ys, s=5, color=col, marker=mk, lw=0.25, edgecolors='white', alpha=.85, rasterized=True)
            X += xs; Y += ys
X, Y = np.array(X), np.array(Y); r2 = 1 - ((Y - X) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
lim = [-8.6, 0.4]; ax.plot(lim, lim, color=INK2, lw=0.6, zorder=0); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Expected bias from ordinary days (K or m s$^{-1}$)'); ax.set_ylabel('Observed bias on records (K or m s$^{-1}$)')
ax.text(-8.2, -3.9, f'$R^2$ = {r2:.2f}\n$n$ = {len(X):,}', fontsize=6, va='top', ha='left')
h = [Line2D([], [], ls='', marker='o', ms=3, color=c, label=g) for g, _, c in GROUPS] + \
    [Line2D([], [], ls='', marker=m, ms=3, color=INK2, mfc='none', mew=0.6, label=l) for m, l in [('o', 'heat'), ('s', 'cold'), ('^', 'wind')]]
ax.legend(handles=h, loc='upper left', ncol=1, fontsize=5.5, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.2)
ax.set_title('One law for every system', fontsize=7); label(ax, 'c')
print('collapse R2', r2, len(X), 'median abs resid', np.median(np.abs(Y - X)))

# d  predictability limit vs lead
ax = fig.add_subplot(gs[1, :2])
Lp = np.array([12, 48, 72, 120, 168, 240])
pg = np.array([-0.42, -0.78, -1.00, -1.75, -3.03, -5.09]); pi = np.array([-0.69, -1.09, -1.35, -2.26, -3.20, -4.99])
ax.fill_between(Lp / 24, np.minimum(pg, pi), np.maximum(pg, pi), color='#e2e2e2', lw=0, label='perfect forecast (two model worlds)')
ax.plot(Lp / 24, pg, color=PERF, lw=0.7); ax.plot(Lp / 24, pi, color=PERF, lw=0.7, ls=(0, (2, 1.5)))
for m, col, lab in [('graphcast', AI, 'GraphCast (AI)'), ('gencastmean', ENS, 'GenCast mean (AI)'), ('hres', PHYS, 'HRES (physics)')]:
    d = np.load(f'{grid15.D}/events/2020_t2m_max_era5_{m}.npz'); L = d['leads']; v = []
    for li in range(len(L)):
        e = d['f'][li] - d['y']; ok = np.isfinite(e) & np.isfinite(d['bhat'][li]) & (d['n'][li] > 50); v.append(e[ok].mean())
    ax.plot(L / 24, v, color=col, lw=1.2, label=lab, marker='o', ms=2.2, mec='white', mew=0.3)
ax.set_xlabel('Lead time (days)'); ax.set_ylabel('Mean forecast error on\n2020 heat records (K)')
ax.set_xlim(0.4, 10.1); ax.set_xticks(range(1, 11)); ax.axhline(0, color=INK2, lw=0.5)
ax.text(5.05, -1.25, 'GenCast world', fontsize=5.5, color=PERF, ha='left'); ax.text(3.0, -1.95, 'IFS ensemble world', fontsize=5.5, color=PERF, ha='right')
ax.legend(loc='lower left', ncol=2)
ax.set_title('Most of the shortfall would remain for a perfect forecast', fontsize=7); label(ax, 'd', x=-0.075)

# e  ratio to limit
ax = fig.add_subplot(gs[1, 2])
rat = {'GraphCast (AI)': ([1.27, 1.31, 1.28, 1.22, 1.15, 1.22], AI), 'GenCast mean (AI)': ([1.14, 1.13, 1.12, 1.23, 1.25, 1.17], ENS),
       'HRES (physics)': ([3.06, 1.82, 1.60, 1.36, 1.34, 1.21], PHYS)}
for lab, (v, col) in rat.items():
    ax.plot(Lp / 24, v, color=col, lw=1.2, marker='o', ms=2.5, mec='white', mew=0.3, label=lab)
ax.axhline(1, color=PERF, lw=0.8); ax.text(9.9, 0.93, 'perfect forecast', ha='right', va='top', fontsize=5.5, color=PERF)
ax.set_ylim(0.75, 3.2); ax.set_xlim(0.3, 10.2); ax.set_xticks([1, 3, 5, 7, 10])
ax.set_xlabel('Lead time (days)'); ax.set_ylabel('Shortfall relative to\nperfect forecast (GenCast world)')
ax.legend(loc='upper right'); ax.set_title('Distance from the limit', fontsize=7); label(ax, 'e')
save(fig, 'fig2')
