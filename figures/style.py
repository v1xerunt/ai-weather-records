"""Shared Nature-style settings for manuscript figures."""
import os, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{ROOT}/src')
OUT = f'{ROOT}/output'
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
W2 = 183 * MM          # double column
W1 = 89 * MM           # single column
INK, INK2, MUTED, GRID = '#222222', '#555555', '#8f8f8f', '#ececec'
AI, PHYS, ENS, PHYS2, PERF = '#2c5d8f', '#c4553a', '#4a9c94', '#d9a441', '#7a7a7a'
plt.rcParams.update({
    'font.family': 'Nimbus Sans', 'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.linewidth': 0.45,
    'xtick.major.width': 0.45, 'ytick.major.width': 0.45, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'pdf.fonttype': 42, 'savefig.dpi': 400, 'lines.linewidth': 1.0, 'axes.titleweight': 'normal', 'axes.labelpad': 2.5, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Nimbus Sans', 'mathtext.it': 'Nimbus Sans:italic', 'mathtext.bf': 'Nimbus Sans:bold',
    'axes.titlelocation': 'left', 'axes.titlepad': 4})


def label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def grid(ax, axis='both'):
    pass


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=250)
    print('saved', name)
PANGU = '#79a6d2'
from matplotlib.colors import LinearSegmentedColormap
BLUES = LinearSegmentedColormap.from_list('b', ['#ffffff', '#dfe8f1', '#a9c1da', '#5f88b3', '#2c5d8f', '#173a5e'])
REDS = LinearSegmentedColormap.from_list('r', ['#ffffff', '#f6e0d6', '#e7a993', '#c4553a', '#8c2d1c'])
DIV = LinearSegmentedColormap.from_list('d', ['#173a5e', '#2c5d8f', '#8fb0d2', '#f4f4f2', '#e9b39f', '#c4553a', '#7d2414'])


def endlabel(ax, x, y, s, color, dx=4, dy=0, **kw):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', color=color, va='center', fontsize=6, **kw)


def soft(ax):
    ax.tick_params(colors=INK2)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color('#9a9a9a')
