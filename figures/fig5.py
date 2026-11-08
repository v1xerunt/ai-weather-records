"""Fig. 5: what is left for the model."""
from style import *
import numpy as np
VIO = '#7b5ea7'
fig = plt.figure(figsize=(W2, 112 * MM))
G = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32, left=0.1, right=0.985, top=0.94, bottom=0.11, width_ratios=[1.25, 1])

# a decomposition (0.25 deg, 2020 heat, ERA5)
ax = fig.add_subplot(G[0, 0]); grid(ax, 'x')
rows = [('GraphCast, 12 h', -0.11, -0.34, -0.30), ('HRES, 12 h', -0.26, -0.92, -0.22),
        ('GraphCast, 5 d', -0.41, -1.50, -0.49), ('HRES, 5 d', -0.32, -2.29, -0.26)]
obs = [-0.74, -1.48, -2.51, -2.70]
cols = [('mean offset', '#d2d2d2'), ('limited predictability  $-(1-\\rho)\\Delta$', '#8c8c8c'), ('damped amplitude  $-\\rho(1-a)\\Delta$', VIO)]
ypos = [0, 1, 2.5, 3.5]
for y, (lab, *v) in zip(ypos, rows):
    left = 0
    for k, (vv, (cl, c)) in enumerate(zip(v, cols)):
        ax.barh(y, vv, left=left, color=c, height=0.62, edgecolor='white', lw=0.6, label=cl if y == 0 else None)
        left += vv
    ax.plot([obs[ypos.index(y)]] * 2, [y - 0.42, y + 0.42], color=INK, lw=1.1, solid_capstyle='butt')
    ax.text(min(left, obs[ypos.index(y)]) - 0.06, y, f'{obs[ypos.index(y)]:+.2f} K', ha='right', va='center', fontsize=6)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
ax.axvline(0, color=INK2, lw=0.5); ax.set_xlim(-3.6, 0.05)
ax.set_xlabel('Contribution to mean error on 2020 heat records (K)')
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color=INK, lw=1.1)); l.append('observed')
ax.legend(h, l, loc='upper left', fontsize=5.5, bbox_to_anchor=(-0.01, 1.02))
ax.set_title('Where the shortfall comes from', fontsize=7); label(ax, 'a', x=-0.2)

# b rank information within records
ax = fig.add_subplot(G[0, 1]); grid(ax, 'y')
L = ['12 h', '2 d', '5 d']; gc = [0.58, 0.45, 0.22]; hr = [0.38, 0.35, 0.22]
x = np.arange(3)
ax.bar(x - 0.17, gc, 0.32, color=AI, label='GraphCast'); ax.bar(x + 0.17, hr, 0.32, color=PHYS, label='HRES')
for xi, a_, b_ in zip(x, gc, hr):
    ax.text(xi - 0.17, a_ + 0.01, f'{a_:.2f}', ha='center', fontsize=5.5); ax.text(xi + 0.17, b_ + 0.01, f'{b_:.2f}', ha='center', fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(L); ax.tick_params(axis='x', length=0); ax.set_ylim(0, 0.68)
ax.set_ylabel('Rank correlation of forecast and\nobserved exceedance among records')
ax.legend(loc='upper right'); ax.set_title('AI ranks records at least as well', fontsize=7); label(ax, 'b', x=-0.22)

# c identical monotone recalibration, 5 d
ax = fig.add_subplot(G[1, 0]); grid(ax, 'y')
bins = ['<0.5', '0.5–1', '1–2', '2–3', '>3']
series = [('GenCast mean', [-2.05, -2.65, -3.20, -3.74, -4.74], ENS), ('GraphCast', [-2.11, -2.73, -3.26, -3.80, -5.42], AI),
          ('HRES', [-2.79, -3.41, -3.86, -4.14, -4.74], PHYS)]
for k, (lab, v, c) in enumerate(series):
    ax.plot(np.arange(5) + (k - 1) * 0.08, v, color=c, marker='o', ms=3.5, mec='white', mew=0.5, label=lab)
ax.set_xticks(range(5)); ax.set_xticklabels(bins); ax.set_xlabel('Amount by which the record is broken (K)')
ax.set_ylabel('Mean error after identical\nrecalibration, 5 days ahead (K)')
ax.text(4, -5.85, '159 events from\n30 weather systems', fontsize=5.5, color=INK2, ha='center', va='top')
ax.set_ylim(-6.6, -1.6)
ax.legend(loc='lower left'); ax.set_title('After identical recalibration, AI matches or leads', fontsize=7); label(ax, 'c', x=-0.2)

# d wind: averaging vectors
ax = fig.add_subplot(G[1, 1]); grid(ax, 'x')
W = [('GenCast, mean of\nmember speeds', -0.06, -0.07, ENS), ('GenCast, speed of\nmean wind vector', -0.35, -0.54, ENS),
     ('GraphCast', -0.20, -0.18, AI), ('Pangu-Weather', -0.23, -0.13, AI), ('FuXi', -0.18, -0.12, AI), ('HRES', 0.14, 0.16, PHYS)]
for i, (lab, v24, v120, c) in enumerate(W):
    ax.plot([v24, v120], [i, i], color=c, lw=0.8, alpha=.5)
    ax.plot(v24, i, 'o', ms=4, color=c, mec='white', mew=0.5); ax.plot(v120, i, 'D', ms=3.4, color='white', mec=c, mew=0.9)
ax.set_yticks(range(len(W))); ax.set_yticklabels([w[0] for w in W], fontsize=5.8); ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
ax.axvline(0, color=INK2, lw=0.5); ax.set_xlim(-0.65, 0.3)
ax.set_xlabel('Error on wind records beyond the\nexpected shortfall (m s$^{-1}$)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=INK2, label='1 day'), Line2D([], [], ls='', marker='D', ms=3.4, mfc='white', mec=INK2, label='5 days')],
          loc='upper right')
ax.set_title('Averaging wind vectors costs speed', fontsize=7); label(ax, 'd', x=-0.42)
save(fig, 'fig5')
