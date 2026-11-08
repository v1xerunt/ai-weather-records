"""0.25 deg land-only download. Stored as flat land vectors (lsm>0.5, lat>-60), in ERA5 (lat desc, lon 0..) order.
  python src/download/wb2_025deg.py static
  python src/download/wb2_025deg.py truth SRC VAR Y0 Y1        SRC in era5|hres_t0   -> data025/SRC/VAR_Y.npy  (T6h, ncell)
  python src/download/wb2_025deg.py fc NAME PATH VAR YEAR LEADS -> data025/fc/NAME/VAR_YEAR.npy (init, lead, ncell)"""
import sys, os, numpy as np, xarray as xr, time, dask
from concurrent.futures import ThreadPoolExecutor
dask.config.set(scheduler='threads', num_workers=8)
B = 'gs://weatherbench2/datasets/'
SO = {'token': 'anon'}
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
OD = f'{ROOT}/data025'
E5 = 'era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
PATHS = {'era5': E5, 'hres_t0': 'hres_t0/2016-2022-6h-1440x721.zarr'}


def static():
    ds = xr.open_zarr(B + E5, storage_options=SO)
    lsm = ds['land_sea_mask'].values; lat = ds.latitude.values; lon = ds.longitude.values
    m = (lsm > 0.5) & (lat[:, None] > -60)
    ii, jj = np.where(m)
    os.makedirs(OD, exist_ok=True)
    np.savez(f'{OD}/static.npz', lat=lat, lon=lon, mask=m, ii=ii, jj=jj, z=ds['geopotential_at_surface'].values[m])
    print('land cells', m.sum())


def _std(da):
    ren = {k: v for k, v in (('lat', 'latitude'), ('lon', 'longitude')) if k in da.dims}
    da = da.rename(ren)
    if da.latitude.values[0] < da.latitude.values[-1]:
        da = da.isel(latitude=slice(None, None, -1))
    return da


def truth(src, var, y0, y1):
    st = np.load(f'{OD}/static.npz'); m = st['mask']
    da = _std(xr.open_zarr(B + PATHS[src], storage_options=SO)[var]).transpose('time', 'latitude', 'longitude')
    od = f'{OD}/{src}'; os.makedirs(od, exist_ok=True)
    for y in range(y0, y1 + 1):
        fn = f'{od}/{var}_{y}.npy'
        if os.path.exists(fn): continue
        sub = da.sel(time=slice(f'{y}-01-01', f'{y}-12-31T18'))
        T = sub.sizes['time']; out = np.empty((T, m.sum()), np.float32)
        def job(k):
            out[k:k + 40] = sub.isel(time=slice(k, k + 40)).values[:, m]
        with ThreadPoolExecutor(4) as ex: list(ex.map(job, range(0, T, 40)))
        np.save(fn + '.tmp.npy', out); os.rename(fn + '.tmp.npy', fn); print(src, var, y, out.shape, flush=True)


def fc(name, path, var, year, leads):
    st = np.load(f'{OD}/static.npz'); m = st['mask']
    da = _std(xr.open_zarr(B + path, storage_options=SO)[var])
    da = da.transpose('time', 'prediction_timedelta', 'latitude', 'longitude')
    od = f'{OD}/fc/{name}'; os.makedirs(od, exist_ok=True)
    fn = f'{od}/{var}_{year}.npy'
    if os.path.exists(fn): return
    td = da.prediction_timedelta.values
    lh = (td / np.timedelta64(1, 'h')).astype(int) if np.issubdtype(td.dtype, np.timedelta64) else td.astype(int)
    li = [int(np.where(lh == l)[0][0]) for l in leads]
    inits = np.arange(np.datetime64(f'{year}-01-01T00'), np.datetime64(f'{year+1}-01-01T00'), np.timedelta64(12, 'h'))
    tv = da.time.values; ti = np.searchsorted(tv, inits)
    ok = (ti < len(tv)) & (tv[np.minimum(ti, len(tv) - 1)] == inits)
    out = np.lib.format.open_memmap(fn + '.tmp.npy', mode='w+', dtype=np.float32, shape=(len(inits), len(leads), int(m.sum())))
    out[:] = np.nan
    idx = np.where(ok)[0]; t0 = time.time()
    def job(i):
        out[i] = da.isel(time=int(ti[i]), prediction_timedelta=li).values[:, m]
    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(job, idx)):
            if n % 100 == 0: print(name, var, year, n, len(idx), f'{time.time()-t0:.0f}s', flush=True)
    out.flush(); del out; os.rename(fn + '.tmp.npy', fn)
    np.save(f'{od}/inits_{year}.npy', inits); np.save(f'{od}/leads.npy', np.array(leads))
    print('done', fn, 'missing', (~ok).sum(), flush=True)


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'static': static()
    elif a[1] == 'truth': truth(a[2], a[3], int(a[4]), int(a[5]))
    else: fc(a[2], a[3], a[4], int(a[5]), [int(x) for x in a[6].split(',')])
