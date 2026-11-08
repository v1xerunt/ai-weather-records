"""1.5 degree WeatherBench 2 grid: data access, local records and climatology. Arrays are (time, lon, lat)."""
import os, numpy as np
from common.paths import ROOT, DATA as D
LEADS = np.arange(12, 241, 12)          # hours
VARS = {'t2m': '2m_temperature', 'ws': '10m_wind_speed'}

_st = np.load(f'{D}/era5/static.npz')
LAT, LON, LSM, ZS = _st['lat'], _st['lon'], _st['lsm'], _st['z']
LAND = (LSM > 0.5) & (LAT[None, :] > -60)   # land, no Antarctica  (lon, lat)


def era5_year(var, y, hours=(0, 12), src='era5'):
    """ERA5 for calendar year y at given UTC hours -> (times, field)."""
    x = np.load(f'{D}/{src}/{VARS.get(var, var)}_{y}.npy', mmap_mode='r')
    t = np.arange(np.datetime64(f'{y}-01-01T00'), np.datetime64(f'{y+1}-01-01T00'), np.timedelta64(6, 'h'))
    assert len(t) == x.shape[0], (y, len(t), x.shape)
    h = (t - t.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
    k = np.isin(h, hours)
    return t[k], np.asarray(x[k])


def era5_range(var, y0, y1, hours=(0, 12)):
    ts, xs = zip(*[era5_year(var, y, hours) for y in range(y0, y1 + 1)])
    return np.concatenate(ts), np.concatenate(xs)


def month_hour(t):
    m = t.astype('datetime64[M]').astype(int) % 12          # 0..11
    h = (t - t.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
    return m, h


def records(var, y0, y1, kind='max', hours=(0, 6, 12, 18)):
    """Record definition of Zhang et al. (Sci. Adv. 2026): per calendar month & cell, over all 6-hourly ERA5 times of y0..y1 (not hour-specific).
    Returns (12, lon, lat)."""
    cache = f'{D}/derived/rec_{var}_{kind}_{y0}_{y1}.npy'
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(f'{D}/derived', exist_ok=True)
    R = np.full((12, 240, 121), -np.inf if kind == 'max' else np.inf, np.float32)
    for y in range(y0, y1 + 1):
        t, x = era5_year(var, y, hours)
        m, _ = month_hour(t)
        for mi in range(12):
            k = m == mi
            R[mi] = np.maximum(R[mi], x[k].max(0)) if kind == 'max' else np.minimum(R[mi], x[k].min(0))
    np.save(cache, R)
    return R


def rec_at(R, t):
    """Record threshold field for each valid time t -> (T, lon, lat)."""
    m, _ = month_hour(t)
    return R[m]


def clim(var, y0=1979, y1=2017, nharm=4, hours=(0, 12)):
    """Smooth day-of-year climatology mean & sd per hour/cell (harmonic LSQ). -> (2, nhour, 366, lon, lat)."""
    cache = f'{D}/derived/clim_{var}_{y0}_{y1}.npy'
    if os.path.exists(cache):
        return np.load(cache)
    t, x = era5_range(var, y0, y1, hours)
    _, h = month_hour(t)
    doy = (t.astype('datetime64[D]') - t.astype('datetime64[Y]')).astype(int)
    def basis(d):
        w = 2 * np.pi * d / 365.25
        return np.stack([np.ones_like(w)] + [f(k * w) for k in range(1, nharm + 1) for f in (np.sin, np.cos)], 1)
    out = np.zeros((2, len(hours), 366, 240, 121), np.float32)
    Bq = basis(np.arange(366))
    for hi, hh in enumerate(hours):
        k = h == hh
        Bx = basis(doy[k]); X = x[k].reshape(k.sum(), -1)
        coef = np.linalg.lstsq(Bx, X, rcond=None)[0]
        mu = Bx @ coef
        coef2 = np.linalg.lstsq(Bx, (X - mu) ** 2, rcond=None)[0]
        out[0, hi] = (Bq @ coef).reshape(366, 240, 121)
        out[1, hi] = np.sqrt(np.maximum(Bq @ coef2, 1e-6)).reshape(366, 240, 121)
    np.save(cache, out)
    return out


def clim_at(C, t, hours=(0, 12)):
    _, h = month_hour(t)
    hi = np.searchsorted(np.array(hours), h)
    doy = (t.astype('datetime64[D]') - t.astype('datetime64[Y]')).astype(int)
    return C[0, hi, doy], C[1, hi, doy]


def forecast_valid(name, var, year, leads=LEADS, member=None):
    """Forecast aligned on valid times of `year` (00/12 UTC). -> (vt, F[lead, T, lon, lat]) NaN where missing."""
    vt = np.arange(np.datetime64(f'{year}-01-01T00'), np.datetime64(f'{year+1}-01-01T00'), np.timedelta64(12, 'h'))
    F = np.full((len(leads), len(vt), 240, 121), np.nan, np.float32)
    for yy in (year - 1, year):
        fn = f'{D}/fc/{name}/{VARS.get(var, var)}_{yy}.npy'
        if not os.path.exists(fn):
            continue
        x = np.load(fn, mmap_mode='r')
        it = np.load(f'{D}/fc/{name}/inits_{yy}.npy')
        for li, L in enumerate(leads):
            lj = int(np.where(LEADS == L)[0][0])
            v = it + np.timedelta64(int(L), 'h')
            k = np.searchsorted(vt, v); ok = (k < len(vt)) & (vt[np.minimum(k, len(vt) - 1)] == v)
            src = x[np.where(ok)[0], lj] if member is None else x[np.where(ok)[0], member, lj]
            F[li, k[ok]] = src
    return vt, F
