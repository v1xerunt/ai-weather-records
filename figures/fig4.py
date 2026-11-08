"""Fig. 4: no training-maximum ceiling."""
from style import *
import numpy as np
from common import grid15, grid025
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(W2, 120 * MM))
G = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42, left=0.08, right=0.985, top=0.94, bottom=0.085)

# a local vs network units
ax = fig.add_subplot(G[0, 0])
d = np.load(f'{grid025.D}/out/2020_t2m_max_era5_graphcast.npz')
zl = (d['y'] - d['mu']) / d['sd']; zn = (d['y'] - 278.2) / 21.4
bins = np.linspace(-3, 7, 101)
ax.hist(zl, bins, color='#a9a9a9', alpha=.9, lw=0, label='local units\n(cell and season)')
ax.hist(zn, bins, color=AI, alpha=.85, lw=0, label='network units\n(global normalization)')
ax.axvline(2.25, color=INK, lw=0.7, ls=(0, (3, 1.5)))
ax.text(2.35, ax.get_ylim()[1] * 0.97, 'warmest land\nvalue in training', fontsize=5.5, va='top')
ax.set_xlim(-2.5, 7); ax.set_xlabel('Standardized value of 2020 heat records'); ax.set_ylabel('Number of record events')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v else '0'))
ax.text(0.9, 15500, 'network units\n(global\nnormalization)', fontsize=5.5, color=AI, ha='right')
ax.text(3.3, 9000, 'local units\n(cell and\nseason)', fontsize=5.5, color=INK2, ha='left')
ax.set_title('Local records are familiar to the network', fontsize=7); label(ax, 'a')

# b, c input-conditioned error
def curve(name, L, ri, si, w=0.25, lo=-2.5, hi=1.5):
    dd = np.load(f'{grid025.D}/input/2020_t2m_max_{name}.npz'); e = dd['edges']; c = 0.5 * (e[:-1] + e[1:])
    o = dd['out'][list(dd['leads']).index(L), ri, si]; xs, ms, ses = [], [], []
    for a_ in np.arange(lo, hi, w):
        k = (c >= a_) & (c < a_ + w); N = o[k, 0].sum()
        if N < 30: continue
        m = o[k, 1].sum() / N; v = o[k, 2].sum() / N - m * m
        xs.append(o[k, 3].sum() / N); ms.append(m); ses.append(np.sqrt(v / N) * 3)
    return np.array(xs), np.array(ms), np.array(ses)
LAB = {'graphcast': ('GraphCast (trained to 2019)', AI), 'pangu': ('Pangu-Weather (trained to 2017)', PANGU), 'hres': ('HRES', PHYS)}
for k, (L, si, ttl, lab) in enumerate([(12, 0, 'Inputs beyond the record, all land, 12 h', 'b'),
                                        (12, 2, 'Cells with new 2018–19 records, 12 h', 'c')]):
    ax = fig.add_subplot(G[0, 1 + k]); grid(ax)
    ax.axvspan(0, 1.5, color='#f3f3f3', lw=0)
    for m, (lb, col) in LAB.items():
        x, y, se = curve(m, L, 0, si)
        ax.fill_between(x, -(y + se), -(y - se), color=col, alpha=.18, lw=0); ax.plot(x, -y, color=col, lw=1.2, label=lb)
    ax.axvline(0, color=INK2, lw=0.6); ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylim(-1.0, 0.8); ax.set_xlim(-2.5, 1.45)
    ax.set_xlabel('Initial state relative to the record (s.d.)'); ax.set_ylabel('Mean forecast error (K)')
    ax.text(0.07, 0.72, 'input never seen\nat this place\nand season' if k == 0 else 'above the\n2017 record,\nunseen by\nPangu',
            fontsize=5.5, color=INK2, va='top')
    if k == 0: ax.legend(loc='lower left', fontsize=5.5)
    ax.set_title(ttl, fontsize=7); label(ax, lab)

# d same-architecture cutoff experiment (00 UTC 2020)
ax = fig.add_subplot(G[1, 0:2]); grid(ax, 'y')
data = {'ordinary days': ([0.031, 0.051, 0.060], [(0.024, 0.037), (0.040, 0.061), (0.040, 0.083)], '#c9d6e4'),
        'warm days near the record': ([0.026, 0.097, 0.183], [(0.003, 0.048), (0.050, 0.138), (0.098, 0.259)], '#7f9fc2'),
        'record events': ([0.068, 0.097, 0.299], [(-0.001, 0.133), (0.010, 0.185), (0.128, 0.501)], AI)}
for j, (lab, (v, ci, col)) in enumerate(data.items()):
    x = np.arange(3) + (j - 1) * 0.22
    ax.bar(x, v, width=0.2, color=col, label=lab)
    ax.errorbar(x, v, yerr=np.array([[vv - c[0] for vv, c in zip(v, ci)], [c[1] - vv for vv, c in zip(v, ci)]]), fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5, capthick=0.6)
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(['12 h', '48 h', '120 h']); ax.tick_params(axis='x', length=0)
ax.set_ylabel('Forecast warming from two\nextra training years (K)', labelpad=2)
ax.legend(loc='upper left')
ax.text(1.0, 0.24, 'records behave like\nother warm days', ha='center', fontsize=6, color=INK2)
ax.set_ylim(-0.03, 0.62); ax.text(2.0, 0.535, 'pull towards the\ntraining climate', ha='center', fontsize=6, color=INK2)
ax.set_title('Same network, two more years of training data', fontsize=7); label(ax, 'd', x=-0.085)

# e Pacific Northwest 2021, Pangu (trained to 2017)
ax = fig.add_subplot(G[1, 2]); grid(ax)
vt = None
for m, col, lb in [('hres', PHYS, 'HRES'), ('pangu', PANGU, 'Pangu-Weather')]:
    e = np.load(f'{grid15.D}/events/2021_t2m_max_era5_{m}.npz')
    t = e['vt'][e['t']].astype('datetime64[h]'); lat = grid15.LAT[e['j']]; lon = grid15.LON[e['i']]
    k = (lat >= 44) & (lat <= 56) & (lon >= 234) & (lon <= 244) & (t >= np.datetime64('2021-06-26T00')) & (t <= np.datetime64('2021-07-02T12'))
    li = list(e['leads']).index(24)
    ox = (e['y'] - e['r'])[k]; fx = (e['f'][li] - e['r'])[k]; ok = np.isfinite(fx)
    ax.scatter(ox[ok], fx[ok], s=5, color=col, alpha=.7, lw=0, label=f'{lb}, 24 h', zorder=3 if m == 'pangu' else 2)
    print(m, ok.sum(), ox[ok].mean(), fx[ok].mean())
ax.plot([0, 11], [0, 11], color=INK2, lw=0.6); ax.axhline(0, color=INK, lw=0.6)
ax.set_xlim(0, 11); ax.set_ylim(-4, 11)
ax.text(10.8, -0.5, '1979–2017 record', ha='right', va='top', fontsize=5.5, color=INK2)
ax.set_xlabel('Observed exceedance of the record (K)'); ax.set_ylabel('Forecast exceedance of the record (K)')
ax.legend(loc='upper left', markerscale=1.6)
ax.set_title('Pacific Northwest, June 2021', fontsize=7); label(ax, 'e')
save(fig, 'fig4')
