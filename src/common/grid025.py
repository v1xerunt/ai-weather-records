"""0.25 degree land cells (flattened): data access, local records and climatology."""
import os, numpy as np
from common.grid15 import month_hour
from common.paths import ROOT, DATA025 as D
VARS = {'t2m': '2m_temperature', 'ws': '10m_wind_speed'}
_st = np.load(f'{D}/static.npz')
LAT = _st['lat'][_st['ii']]; LON = _st['lon'][_st['jj']]; NC = len(LAT)


def times6(y):
    return np.arange(np.datetime64(f'{y}-01-01T00'), np.datetime64(f'{y+1}-01-01T00'), np.timedelta64(6, 'h'))


def truth_year(var, y, src='era5', hours=(0, 12)):
    x = np.load(f'{D}/{src}/{VARS[var]}_{y}.npy', mmap_mode='r'); t = times6(y)
    _, h = month_hour(t); k = np.isin(h, hours)
    return t[k], np.asarray(x[k])


def records(var, y0, y1, kind='max'):
    fn = f'{D}/derived/rec_{var}_{kind}_{y0}_{y1}.npy'
    if os.path.exists(fn): return np.load(fn)
    os.makedirs(f'{D}/derived', exist_ok=True)
    R = np.full((12, NC), -np.inf if kind == 'max' else np.inf, np.float32)
    for y in range(y0, y1 + 1):
        t, x = truth_year(var, y, hours=(0, 6, 12, 18)); m, _ = month_hour(t)
        for mi in range(12):
            R[mi] = np.maximum(R[mi], x[m == mi].max(0)) if kind == 'max' else np.minimum(R[mi], x[m == mi].min(0))
    np.save(fn, R); return R


def build_records(var):
    """One pass over ERA5 years: running max/min, snapshot at 2015, 2017, 2019."""
    os.makedirs(f'{D}/derived', exist_ok=True)
    Rx = np.full((12, NC), -np.inf, np.float32); Rn = np.full((12, NC), np.inf, np.float32)
    for y in range(1979, 2020):
        t, x = truth_year(var, y, hours=(0, 6, 12, 18)); m, _ = month_hour(t)
        for mi in range(12):
            Rx[mi] = np.maximum(Rx[mi], x[m == mi].max(0)); Rn[mi] = np.minimum(Rn[mi], x[m == mi].min(0))
        if y in (2015, 2017, 2019):
            np.save(f'{D}/derived/rec_{var}_max_1979_{y}.npy', Rx); np.save(f'{D}/derived/rec_{var}_min_1979_{y}.npy', Rn)
        print('rec', var, y, flush=True)


def clim(var, y0=1979, y1=2017, nharm=4, hours=(0, 12)):
    fn = f'{D}/derived/clim_{var}_{y0}_{y1}.npy'
    if os.path.exists(fn): return np.load(fn)
    os.makedirs(f'{D}/derived', exist_ok=True)
    def basis(d):
        w = 2 * np.pi * d / 365.25
        return np.stack([np.ones_like(w)] + [f(k * w) for k in range(1, nharm + 1) for f in (np.sin, np.cos)], 1)
    G = np.zeros((2, len(hours), 1 + 2 * nharm, 1 + 2 * nharm)); Rx = np.zeros((2, len(hours), 1 + 2 * nharm, NC))
    S2 = np.zeros((len(hours), 1 + 2 * nharm, NC))
    # pass 1: mean
    for p in range(2):
        for y in range(y0, y1 + 1):
            t, x = truth_year(var, y, hours=hours); _, h = month_hour(t)
            doy = (t.astype('datetime64[D]') - t.astype('datetime64[Y]')).astype(int)
            for hi, hh in enumerate(hours):
                k = h == hh; Bx = basis(doy[k]); xx = x[k].astype(np.float64)
                if p == 0:
                    G[0, hi] += Bx.T @ Bx; Rx[0, hi] += Bx.T @ xx
                else:
                    mu = Bx @ coef[hi]; Rx[1, hi] += Bx.T @ (xx - mu) ** 2
        if p == 0:
            coef = [np.linalg.solve(G[0, hi], Rx[0, hi]) for hi in range(len(hours))]
    Bq = basis(np.arange(366))
    out = np.zeros((2, len(hours), 366, NC), np.float32)
    for hi in range(len(hours)):
        out[0, hi] = Bq @ coef[hi]
        out[1, hi] = np.sqrt(np.maximum(Bq @ np.linalg.solve(G[0, hi], Rx[1, hi]), 1e-6))
    np.save(fn, out); return out


def clim_at(C, t, hours=(0, 12)):
    _, h = month_hour(t); hi = np.searchsorted(np.array(hours), h)
    doy = (t.astype('datetime64[D]') - t.astype('datetime64[Y]')).astype(int)
    return C[0, hi, doy], C[1, hi, doy]


def forecast_valid(name, var, year, lead):
    vt = np.arange(np.datetime64(f'{year}-01-01T00'), np.datetime64(f'{year+1}-01-01T00'), np.timedelta64(12, 'h'))
    F = np.full((len(vt), NC), np.nan, np.float32)
    x = np.load(f'{D}/fc/{name}/{VARS[var]}_{year}.npy', mmap_mode='r')
    it = np.load(f'{D}/fc/{name}/inits_{year}.npy'); ls = list(np.load(f'{D}/fc/{name}/leads.npy'))
    v = it + np.timedelta64(int(lead), 'h'); k = np.searchsorted(vt, v)
    ok = (k < len(vt)) & (vt[np.minimum(k, len(vt) - 1)] == v)
    F[k[ok]] = x[np.where(ok)[0], ls.index(lead)]
    return vt, F
