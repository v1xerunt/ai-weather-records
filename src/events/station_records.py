"""Independent truth: METAR station 2 m temperature (MADIS QC). Records per station x calendar month over 2001-2017
(4 synoptic hours). Events in YEAR at 00/12 UTC. Forecasts from the nearest 0.25deg land cell (graphcast/pangu/hres/gc2017,
leads 12/48/120) and nearest 1.5deg land cell (other models, all leads). Per-station cross-fitted expected error from ordinary days.
usage: python src/events/station_records.py build            -> data/metar/stations.npz (QCed 00/06/12/18 series 2001-2020, station meta)
       python src/events/station_records.py eval YEAR KIND    -> prints obs vs expected record bias per model/lead, saves data/metar/events_YEAR_KIND.npz"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
MD = f'{ROOT}/data/metar'
Y0, Y1 = 2001, 2020
GOOD = {'S', 'V', 'G', 'C'}       # MADIS QC: screened / verified / subjective good / coarse pass


def build():
    t6 = pd.date_range(f'{Y0}-01-01', f'{Y1}-12-31 18:00', freq='6h')
    frames = []
    for y in range(Y0, Y1 + 1):
        df = pd.read_parquet(f'{MD}/{y}.parquet', columns=['stationName', 'latitude', 'longitude', 'elevation', 'timeObs',
                                                            'timeNominal', 'temperature', 'temperatureDD'])
        df = df[df.temperatureDD.isin(GOOD) & df.temperature.between(180, 340)]
        df['dt'] = (df.timeObs - df.timeNominal).abs()
        df = df[df.dt <= pd.Timedelta('30min')].sort_values('dt').drop_duplicates(['stationName', 'timeNominal'])
        frames.append(df[['stationName', 'latitude', 'longitude', 'elevation', 'timeNominal', 'temperature']])
        print('read', y, len(df), flush=True)
    df = pd.concat(frames)
    meta = df.groupby('stationName')[['latitude', 'longitude', 'elevation']].median()
    # completeness: >=60% of 6-hourly slots in 2001-2017 and >=80% in 2018 and 2020
    ti = ((df.timeNominal - t6[0]) / pd.Timedelta('6h')).astype(int).values
    sid = meta.index.get_indexer(df.stationName)
    X = np.full((len(t6), len(meta)), np.nan, np.float32); X[ti, sid] = df.temperature.values
    yrs = t6.year.values
    c_ref = np.isfinite(X[(yrs >= 2001) & (yrs <= 2017)]).mean(0)
    c18 = np.isfinite(X[yrs == 2018]).mean(0); c20 = np.isfinite(X[yrs == 2020]).mean(0)
    keep = (c_ref >= 0.6) & (c18 >= 0.8) & (c20 >= 0.8) & (meta.latitude.values > -60)
    X = X[:, keep]; meta = meta[keep]
    # spike check: drop values deviating > 12 K from both 6-h neighbours in the same direction
    d1 = X[1:-1] - X[:-2]; d2 = X[1:-1] - X[2:]
    spike = (np.abs(d1) > 12) & (np.abs(d2) > 12) & (np.sign(d1) == np.sign(d2))
    X[1:-1][spike] = np.nan
    print('stations kept', keep.sum(), 'spikes removed', spike.sum(), flush=True)
    np.savez(f'{MD}/stations.npz', X=X, t=t6.values, name=meta.index.values, lat=meta.latitude.values,
             lon=meta.longitude.values % 360, elev=meta.elevation.values)


if __name__ == '__main__' and sys.argv[1] == 'build':
    build()


def evaluate(Y, KIND):
    import grid15, grid025
    from common.ordinary_days import _win
    from scipy.spatial import cKDTree
    s = 1 if KIND == 'max' else -1
    d = np.load(f'{MD}/stations.npz', allow_pickle=True)
    X, t6 = d['X'], d['t']; S = X.shape[1]
    t6 = pd.DatetimeIndex(t6); yrs = t6.year.values; mon6 = t6.month.values - 1; hr6 = t6.hour.values
    ref = (yrs >= 2001) & (yrs <= 2017)
    R = np.full((12, S), np.nan, np.float32)
    for m in range(12):
        k = ref & (mon6 == m); Xm = X[k]
        cov = np.isfinite(Xm).mean(0)
        R[m] = np.where(cov >= 0.6, (np.nanmax(Xm, 0) if KIND == 'max' else np.nanmin(Xm, 0)), np.nan)
    # harmonic climatology (mean) per hour from ref period
    doy6 = t6.dayofyear.values - 1
    def basis(dd):
        w = 2 * np.pi * dd / 365.25
        return np.stack([np.ones_like(w)] + [f(k * w) for k in range(1, 5) for f in (np.sin, np.cos)], 1)
    ky = (yrs == Y) & np.isin(hr6, (0, 12))
    vt = t6[ky].values; y = X[ky]; mu = np.full_like(y, np.nan)
    for h in (0, 12):
        kr = ref & (hr6 == h); B = basis(doy6[kr]); kk = hr6[ky] == h; Bq = basis(doy6[ky][kk])
        for j in range(S):
            ok = np.isfinite(X[kr, j])
            if ok.sum() > 500:
                c = np.linalg.lstsq(B[ok], X[kr, j][ok], rcond=None)[0]; mu[kk, j] = Bq @ c
    mon = pd.DatetimeIndex(vt).month.values - 1
    r = R[mon]
    ev = np.isfinite(y) & np.isfinite(r) & (s * (y - r) > 0)
    print(f'{Y} {KIND}: stations {S}, record events {ev.sum()} at {np.unique(np.where(ev)[1]).size} stations', flush=True)
    # nearest land cells
    def xyz(la, lo):
        la, lo = np.deg2rad(la), np.deg2rad(lo); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    sx = xyz(d['lat'], d['lon'])
    i025 = cKDTree(xyz(grid025.LAT, grid025.LON)).query(sx)[1]
    li, lj = np.where(grid15.LAND); i15 = cKDTree(xyz(grid15.LAT[lj], grid15.LON[li])).query(sx)[1]
    ya = y - mu
    out = {}
    cl = (pd.DatetimeIndex(vt).dayofyear.values // 7)[:, None] * 1000 + ((d['lon'] // 30).astype(int) * 10 + ((d['lat'] + 90) // 30).astype(int))[None, :]
    rng = np.random.default_rng(0)
    MODELS = [('graphcast', 25), ('pangu', 25), ('hres', 25), ('gc2017', 25), ('fuxi', 15), ('gencastmean', 15),
              ('era5fc', 15), ('ensmean', 15), ('hres', 15), ('graphcast', 15), ('pangu', 15)]
    for name, res in MODELS:
        for L in (12, 48, 120):
            try:
                if res == 25:
                    if not os.path.exists(f'{grid025.D}/fc/{name}/2m_temperature_{Y}.npy'): continue
                    F = grid025.forecast_valid(name, 't2m', Y, L)[1][:, i025]
                else:
                    F = grid15.forecast_valid(name, 't2m', Y, leads=[L])[1][0][:, li[i15], lj[i15]]
            except (FileNotFoundError, ValueError):
                continue
            fa = F - mu
            M = np.isfinite(fa) & np.isfinite(ya)
            if M.sum() < 1000: continue
            z = lambda a: np.where(M, a, 0.0).astype(np.float64)
            Sm = {k: _win(v) for k, v in dict(n=M.astype(float), y=z(ya), f=z(fa), yy=z(ya) ** 2, fy=z(fa) * z(ya)).items()}
            n = np.maximum(Sm['n'], 1); my, mf = Sm['y'] / n, Sm['f'] / n
            Bc = (Sm['fy'] / n - mf * my) / (Sm['yy'] / n - my ** 2)
            bh = (mf - my) + (Bc - 1) * (ya - my)
            ok = ev & M & (Sm['n'] > 30) & np.isfinite(bh)
            if ok.sum() < 50: continue
            err = s * (F - y)[ok]; gal = s * bh[ok]
            _, cid = np.unique(cl[ok], return_inverse=True); K = cid.max() + 1; bs = []
            for _ in range(300):
                w = np.bincount(rng.integers(0, K, K), minlength=K)[cid]; bs.append(((err - gal) * w).sum() / w.sum())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'  {name:11s} {res:2d} lead {L:3d}: N={ok.sum():5d} obs {err.mean():+.2f}  expected {gal.mean():+.2f}  '
                  f'excess {err.mean()-gal.mean():+.2f} [{lo:+.2f},{hi:+.2f}]  median B {np.nanmedian(Bc[ok]):.2f}', flush=True)
            out[f'{name}_{res}_{L}'] = np.stack([err, gal, np.where(ok)[1].astype(float), np.where(ok)[0].astype(float)])
    np.savez(f'{MD}/events_{Y}_{KIND}.npz', i025=i025, i15=i15, **out)


if __name__ == '__main__' and sys.argv[1] == 'eval':
    evaluate(int(sys.argv[2]), sys.argv[3])
