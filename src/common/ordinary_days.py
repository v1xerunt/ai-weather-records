"""Cross-fitted ordinary-day statistics (window moments) and the expected error on selected events."""
import os
import numpy as np
from common.grid15 import LAND

W, EXCL = int(os.environ.get("GW", 30)), int(os.environ.get("GEX", 6))  # +-15 d window, exclude +-3 d (12-hourly steps)


def _sp3(x):
    """3x3 spatial box sum on (T, lon, lat); lon periodic, lat zero-padded."""
    s = x + np.roll(x, 1, 1) + np.roll(x, -1, 1)
    o = s.copy()
    o[:, :, 1:] += s[:, :, :-1]
    o[:, :, :-1] += s[:, :, 1:]
    return o


def _win(x, w=W, e=EXCL):
    """Sum over t' in [t-w, t+w] \\ [t-e, t+e] along axis 0."""
    T = x.shape[0]
    c = np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), np.cumsum(x, 0)])
    i = np.arange(T)
    def rng(a, b):
        return c[np.clip(b + 1, 0, T)] - c[np.clip(a, 0, T)]
    return rng(i - w, i + w) - rng(i - e, i + e)


def moments(ya, fa, mask=None, spatial=True):
    """ya, fa: anomalies (T, lon, lat). Returns dict of cross-fitted window moments, each (T, lon, lat)."""
    M = np.isfinite(ya) & np.isfinite(fa) & LAND[None]
    if mask is not None:
        M &= mask
    y = np.where(M, ya, 0).astype(np.float64)
    f = np.where(M, fa, 0).astype(np.float64)
    sp = _sp3 if spatial else (lambda z: z)
    S = {k: _win(sp(v)) for k, v in dict(n=M.astype(np.float64), y=y, f=f, yy=y * y, ff=f * f, fy=f * y).items()}
    n = np.maximum(S['n'], 1)
    mu_y, mu_f = S['y'] / n, S['f'] / n
    vy = S['yy'] / n - mu_y ** 2
    vf = S['ff'] / n - mu_f ** 2
    cv = S['fy'] / n - mu_f * mu_y
    return dict(n=S['n'], mu_y=mu_y, mu_f=mu_f, vy=vy, vf=vf, cv=cv)


def predict(m, ya):
    """expected error from ordinary days: E[f-y | y] and E[(f-y)^2 | y] (Gaussian) at anomaly ya."""
    B = m['cv'] / m['vy']
    bias = (m['mu_f'] - m['mu_y']) + (B - 1) * (ya - m['mu_y'])
    cvar = np.maximum(m['vf'] - m['cv'] ** 2 / m['vy'], 0)
    return bias, bias ** 2 + cvar, B
