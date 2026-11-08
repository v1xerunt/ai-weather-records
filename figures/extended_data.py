"""Extended Data figures."""
from style import *
import numpy as np
from common import grid15, grid025
from matplotlib.lines import Line2D
# ED1 exceedance coordinates at three leads
fig, axs = plt.subplots(1, 3, figsize=(W2, 55 * MM), sharey=False)
edges = [0, 0.5, 1, 2, 3, 6]; xc = np.arange(5)
for ax, L in zip(axs, (12, 48, 120)):
    grid(ax, 'y')
    for k, (m, col, lab) in enumerate([('graphcast', AI, 'GraphCast (AI)'), ('pangu', PANGU, 'Pangu-Weather (AI)'), ('hres', PHYS, 'HRES (physics)')]):
        d = np.load(f'{grid025.D}/out/2020_t2m_max_era5_{m}.npz'); li = list(d['leads']).index(L)
        e = d['f'][li] - d['y']; bh = d['bhat'][li]; ex = d['y'] - d['r']; ok = np.isfinite(e) & np.isfinite(bh)
        mo = [e[ok & (ex >= a) & (ex < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        mb = [bh[ok & (ex >= a) & (ex < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        off = (k - 1) * 0.1
        ax.plot(xc + off, mb, color=col, lw=1); ax.plot(xc + off, mo, 'o', ms=3.2, color=col, mec='white', mew=0.5, label=lab)
    ax.set_xticks(xc); ax.set_xticklabels(['0–0.5', '0.5–1', '1–2', '2–3', '>3'])
    ax.set_title(f'Lead {L} h', fontsize=7); ax.set_xlabel('Amount by which the record is broken (K)')
axs[0].set_ylabel('Mean error on 2020 heat records (K)')
h = [Line2D([], [], color=c, marker='o', ms=3, lw=0, label=l) for c, l in [(AI, 'GraphCast (AI)'), (PANGU, 'Pangu-Weather (AI)'), (PHYS, 'HRES (physics)')]] + [Line2D([], [], color=INK2, lw=1, label='expected')]
axs[0].legend(handles=h, loc='lower left'); fig.tight_layout(); save(fig, 'ed_exceedance')

# ED2 input-conditioned at three leads
def curve(name, L, ri, si, w=0.25, lo=-2.5, hi=1.5):
    dd = np.load(f'{grid025.D}/input/2020_t2m_max_{name}.npz'); e = dd['edges']; c = 0.5 * (e[:-1] + e[1:])
    o = dd['out'][list(dd['leads']).index(L), ri, si]; xs, ms, ses = [], [], []
    for a_ in np.arange(lo, hi, w):
        k = (c >= a_) & (c < a_ + w); N = o[k, 0].sum()
        if N < 30: continue
        m = o[k, 1].sum() / N; v = o[k, 2].sum() / N - m * m
        xs.append(o[k, 3].sum() / N); ms.append(m); ses.append(np.sqrt(v / N) * 3)
    return np.array(xs), np.array(ms), np.array(ses)
fig, axs = plt.subplots(1, 3, figsize=(W2, 55 * MM))
for ax, L in zip(axs, (12, 48, 120)):
    grid(ax); ax.axvspan(0, 1.5, color='#f3f3f3', lw=0)
    for m, lb, col in [('graphcast', 'GraphCast (AI)', AI), ('pangu', 'Pangu-Weather (AI)', PANGU), ('hres', 'HRES (physics)', PHYS)]:
        x, y, se = curve(m, L, 0, 0)
        ax.fill_between(x, -(y + se), -(y - se), color=col, alpha=.18, lw=0); ax.plot(x, -y, color=col, lw=1.2, label=lb)
    ax.axvline(0, color=INK2, lw=0.6); ax.axhline(0, color=INK2, lw=0.5); ax.set_xlim(-2.5, 1.45)
    ax.set_title(f'Lead {L} h', fontsize=7); ax.set_xlabel('Initial state relative to the record (s.d.)')
axs[0].set_ylabel('Mean forecast error (K)'); axs[0].legend(loc='lower left'); fig.tight_layout(); save(fig, 'ed_input')

# ED3 record-history length and ensemble members
fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 55 * MM), gridspec_kw=dict(wspace=0.45))
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.86)
ax = axs[0]; grid(ax, 'y')
H = [('', '1979–2017'), ('_h1999', '1999–2017'), ('_h2009', '2009–2017')]
for k, (m, col, lab) in enumerate([('graphcast', AI, 'GraphCast (AI)'), ('pangu', PANGU, 'Pangu-Weather (AI)'), ('hres', PHYS, 'HRES (physics)')]):
    mo, mb = [], []
    for suf, _ in H:
        d = np.load(f'{grid15.D}/events/2020_t2m_max_era5_{m}{suf}.npz'); li = list(d['leads']).index(24)
        e = d['f'][li] - d['y']; ok = np.isfinite(e) & np.isfinite(d['bhat'][li]) & (d['n'][li] > 50)
        mo.append(e[ok].mean()); mb.append(d['bhat'][li][ok].mean())
    x = np.arange(3) + (k - 1) * 0.1
    ax.plot(x, mb, color=col, lw=1); ax.plot(x, mo, 'o', ms=3.2, color=col, mec='white', mew=0.5, label=lab)
    print(m, np.round(mo, 2), np.round(mb, 2))
ax.set_xticks(range(3)); ax.set_xticklabels([h[1] for h in H]); ax.set_xlabel('Period used to define the record')
ax.set_ylabel('Mean error on 2020 heat records, 24 h (K)'); ax.legend(loc='center left', fontsize=5.5)
ax.set_title('Shorter history, smaller shortfall', fontsize=7); label(ax, 'a', x=-0.25)
ax = axs[1]; grid(ax, 'y')
for k, (base, col, lab) in enumerate([('gencast', ENS, 'GenCast (AI)'), ('ens', PHYS2, 'IFS ensemble (physics)')]):
    dm = np.load(f'{grid15.D}/events/2020_t2m_max_era5_{base}mean.npz'); L = dm['leads']
    def curve_(d):
        v = []
        for li in range(len(L)):
            e = d['f'][li] - d['y']; ok = np.isfinite(e) & np.isfinite(d['bhat'][li]) & (d['n'][li] > 50); v.append(e[ok].mean())
        return np.array(v)
    ax.plot(L / 24, curve_(dm), color=col, lw=1.3, label=f'{lab} mean')
    for j in range(3):
        ax.plot(L / 24, curve_(np.load(f'{grid15.D}/events/2020_t2m_max_era5_{base}10:{j}.npz')), color=col, lw=0.6, ls=(0, (2, 1.2)), label=f'{lab} single members' if j == 0 else None)
ax.set_xlabel('Lead time (days)'); ax.set_ylabel('Mean error on 2020 heat records (K)'); ax.legend(loc='lower left', fontsize=5.5)
ax.set_title('Members fall as short as the mean', fontsize=7); label(ax, 'b', x=-0.25)
save(fig, 'ed_history_members')
