"""Fig. 1: study overview. a design, b error budget, c findings."""
from style import *
import numpy as np
from matplotlib.patches import FancyBboxPatch
C_PRED, C_VER, C_CEIL, C_MOD = '#7a7a7a', PHYS2, AI, '#7b5ea7'
TINT = {C_PRED: '#eeeeee', C_VER: '#f8eedb', C_CEIL: '#e3ebf4', C_MOD: '#ece6f3'}
fig = plt.figure(figsize=(W2, 150 * MM))
G = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], width_ratios=[1.3, 1], hspace=0.28, wspace=0.42,
                     left=0.03, right=0.985, top=0.95, bottom=0.03)

def box(ax, x, y, w, h, title, body, fc, ec, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.015', fc=fc, ec=ec, lw=0.6))
    ax.text(x + 0.02, y + h - 0.035, title, fontsize=6.5, fontweight='bold', color=tc, va='top')
    ax.text(x + 0.02, y + h - 0.105, body, fontsize=5.4, color=INK2, va='top', linespacing=1.35)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#9a9a9a', lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))

# a design
ax = fig.add_subplot(G[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); label(ax, 'a', x=0.0, y=1.0)
ax.add_patch(FancyBboxPatch((0.0, 0.50), 0.33, 0.47, boxstyle='round,pad=0,rounding_size=0.015', fc='#f6f6f6', ec='#cfcfcf', lw=0.6))
ax.text(0.02, 0.94, 'Forecasts', fontsize=6.5, fontweight='bold', va='top')
ax.text(0.02, 0.865, 'AI models', fontsize=5.8, fontweight='bold', color=AI, va='top')
ax.text(0.02, 0.815, 'GraphCast, Pangu-Weather,\nFuXi, GenCast ensemble', fontsize=5.4, color=INK2, va='top', linespacing=1.3)
ax.text(0.02, 0.705, 'Physics-based', fontsize=5.8, fontweight='bold', color=PHYS, va='top')
ax.text(0.02, 0.655, 'HRES, IFS ensemble,\nIFS started from ERA5', fontsize=5.4, color=INK2, va='top', linespacing=1.3)
box(ax, 0.0, 0.25, 0.33, 0.225, 'Record events', 'heat, cold and wind beyond\n1979–2017 local records', '#f6f6f6', '#cfcfcf')
box(ax, 0.0, 0.0, 0.33, 0.225, 'Three truths', 'ERA5, HRES analysis,\n2,787 weather stations', '#f6f6f6', '#cfcfcf')
# distribution icon: climatology with the record tail shaded
cx, cy = 0.475, 0.50
xs = np.linspace(-3, 3, 200); ys = np.exp(-xs ** 2 / 2)
X = cx + xs * 0.022; Y = cy - 0.06 + ys * 0.17
ax.fill_between(X, cy - 0.06, Y, color='#e3e3e3', lw=0, zorder=3)
t = xs >= 1.4
ax.fill_between(X[t], cy - 0.06, Y[t], color=PHYS, lw=0, zorder=4)
ax.plot(X, Y, color=INK2, lw=0.8, zorder=5)
ax.plot([cx - 0.075, cx + 0.075], [cy - 0.06, cy - 0.06], color=INK2, lw=0.7, zorder=5)
xr = cx + 1.4 * 0.022
ax.plot([xr, xr], [cy - 0.06, cy + 0.13], color=INK, lw=0.7, ls=(0, (2, 1.2)), zorder=6)
ax.text(xr, cy + 0.14, 'record', fontsize=5, color=INK2, va='bottom', ha='center')
ax.text(cx, cy - 0.085, 'Mean error\non records', ha='center', va='top', fontsize=6.2, fontweight='bold', color=INK)
src = [(0.71, 'Predictability', 'expected shortfall\nperfect forecasts', C_PRED),
       (0.475, 'Verification', 'swapping the truth\nweather stations', C_VER),
       (0.24, 'Training ceiling?', 'inputs beyond records\nlater training cutoff', C_CEIL),
       (0.005, 'Model itself', 'amplitude vs information\nwind vectors', C_MOD)]
for yy, t, b, c in src:
    box(ax, 0.66, yy, 0.34, 0.215, t, b, TINT[c], c, tc=c if c != C_PRED else INK)
# orthogonal connectors
LC = '#9a9a9a'; ym = cy + 0.02
lys = (0.735, 0.3625, 0.1125)
for yy in lys: ax.plot([0.33, 0.36], [yy, yy], color=LC, lw=0.7, solid_capstyle='butt')
ax.plot([0.36, 0.36], [min(lys), max(lys)], color=LC, lw=0.7)
arrow(ax, 0.36, ym, 0.392, ym)
rys = [yy + 0.107 for yy, *_ in src]
ax.plot([0.558, 0.605], [ym, ym], color=LC, lw=0.7)
ax.plot([0.605, 0.605], [min(rys), max(rys)], color=LC, lw=0.7)
for yy in rys: arrow(ax, 0.605, yy, 0.655, yy)
ax.set_title('Study design', fontsize=7, x=0.05)

# b error budget
ax = fig.add_subplot(G[0, 1]); label(ax, 'b', x=-0.3, y=1.0)
rows = [('GraphCast (AI)', -0.53, -2.14, AI), ('GenCast mean (AI)', -0.48, -2.15, ENS), ('HRES (physics)', -1.29, -2.38, PHYS),
        ('HRES, scored on own analysis', -0.12, -1.67, PHYS)]
lim = {12: -0.42, 120: -1.75}
for j, (L, k, off) in enumerate([(12, 1, 0), (120, 2, 5.4)]):
    for i, (lab, v12, v120, c) in enumerate(rows):
        v = v12 if L == 12 else v120; y = off + i
        u = max(v, lim[L]) if lab != 'HRES, scored on own analysis' else v
        ax.barh(y, lim[L] if lab != 'HRES, scored on own analysis' else v, color='#d2d2d2', height=0.62, lw=0)
        if lab != 'HRES, scored on own analysis' and v < lim[L]:
            ax.barh(y, v - lim[L], left=lim[L], color=c, height=0.62, lw=0)
        if lab == 'HRES, scored on own analysis':
            ax.barh(y, v, color='white', edgecolor=c, hatch='//////', height=0.62, lw=0.6)
        ax.text(v - 0.06, y, f'{v:.2f}', ha='right', va='center', fontsize=5.6)
    ax.plot([lim[L]] * 2, [off - 0.5, off + 3.5], color=INK, lw=0.8, ls=(0, (2, 1.2)))
    ax.text(-3.35, off - 0.75 if L == 120 else off - 0.75, f'{L} h' if L == 12 else '5 days', fontsize=6.5, fontweight='bold', va='center')
ax.set_yticks([i for i in range(4)] + [5.4 + i for i in range(4)])
ax.set_yticklabels([r[0] for r in rows] * 2, fontsize=5.8); ax.tick_params(axis='y', length=0)
ax.invert_yaxis(); ax.set_xlim(-3.4, 0.05); ax.axvline(0, color=INK2, lw=0.5)
ax.set_xlabel('Mean error on 2020 heat records (K)')
ax.text(-0.44, -0.75, 'perfect forecast', fontsize=5.5, ha='center', va='center', color=INK)
pass
ax.set_title('Error relative to a perfect forecast', fontsize=7)

# c findings table
ax = fig.add_subplot(G[1, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); label(ax, 'c', x=0.0, y=1.0)
cols = [0.015, 0.155, 0.43, 0.82]
hdr = ['Source', 'Finding', 'Evidence', 'Number']
yt = 0.955
for x, h in zip(cols, hdr): ax.text(x + 0.005, yt, h, fontsize=6.5, fontweight='bold', va='center', ha='left' if h != 'Number' else 'left')
ax.plot([0.015, 0.985], [yt - 0.035, yt - 0.035], color=INK, lw=0.7)
R = [(C_PRED, 'Predictability', 'Record errors are set on ordinary days', 'one relationship from ordinary days fits 7 systems,\n3 variables, 2 truths and all lead times', '$R^2$ = 0.93', 'Fig. 2'),
     (C_PRED, '', 'A perfect forecast shows most of the error', 'share of the AI shortfall that a perfect forecast\nin a model world also shows', '75–90%', 'Fig. 2'),
     (C_VER, 'Verification', 'Verification data reverse the ranking', 'HRES gain from switching ERA5 for its own analysis\n(GraphCast ahead by 0.74 K on the same truth)', '+1.28 K', 'Fig. 3'),
     (C_VER, '', 'AI models do as well or better at stations', 'record shortfall 12 h ahead, 2020 heat\nGraphCast vs HRES at 2,787 stations', '2.10 vs 2.41 K', 'Fig. 3'),
     (C_CEIL, 'Training ceiling?', 'No break beyond the record', 'error for inputs already above the local record,\n12 h ahead', '< 0.2 K', 'Fig. 4'),
     (C_CEIL, '', 'Extra training years do not favour records', 'same network trained two years longer,\nrecords minus other warm days at 48 h', '0.00 K', 'Fig. 4'),
     (C_MOD, 'Model itself', 'Remaining AI deficit is damped amplitude', 'extra smoothing of GraphCast relative to HRES,\nrecoverable by recalibration', '0.08–0.23 K', 'Fig. 5'),
     (C_MOD, '', 'Averaging wind vectors reduces speed', 'GenCast mean-vector speed, shortfall beyond\nexpectation one day ahead', '0.35 m s$^{-1}$', 'Fig. 5')]
h = (yt - 0.06) / len(R)
for p in range(len(R) // 2):
    c = R[2 * p][0]; ytop = yt - 0.05 - 2 * p * h; ybot = ytop - 2 * h
    ax.add_patch(plt.Rectangle((0.015, ybot + 0.008), 0.97, 2 * h - 0.016, color=TINT[c], lw=0))
    ax.plot([cols[1], 0.985], [ytop - h, ytop - h], color='white', lw=1.2)
    ax.text(cols[0] + 0.008, ytop - h, R[2 * p][1], fontsize=6.3, fontweight='bold', color=c if c != C_PRED else INK, va='center')
for i, (c, s_, f, e, n, fg) in enumerate(R):
    yc = yt - 0.05 - (i + 0.5) * h
    ax.text(cols[1] + 0.005, yc, f, fontsize=6.2, va='center', color=INK)
    ax.text(cols[2] + 0.005, yc, e, fontsize=5.6, va='center', color=INK2, linespacing=1.25)
    ax.text(cols[3] + 0.005, yc, n, fontsize=6.5, va='center', fontweight='bold', color=INK)
    ax.text(0.98, yc, fg, fontsize=5.4, va='center', ha='right', color=MUTED)
save(fig, 'fig1')
