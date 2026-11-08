"""Download WB2 1.5deg fields to local npy. Usage:
  python src/download/wb2_15deg.py era5 VAR Y0 Y1
  python src/download/wb2_15deg.py fc NAME PATH VAR YEAR [members]
Forecasts saved as data/fc/<NAME>/<VAR>_<YEAR>.npy shape (init, lead, lon, lat) [member axis after init if members]
leads: 12..240h step 12h. inits: YEAR-01-01T00 .. YEAR-12-31T12 every 12h."""
import sys, os, numpy as np, xarray as xr, time, dask
from concurrent.futures import ThreadPoolExecutor
dask.config.set(scheduler='threads', num_workers=16)
B='gs://weatherbench2/datasets/'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
SO={'token':'anon'}
LEADS=np.arange(12,241,12)

def era5(var,y0,y1,src='era5'):
    path={'era5':'era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr',
          'hres_t0':'hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr'}[src]
    ds=xr.open_zarr(B+path,storage_options=SO)
    od=f'{ROOT}/data/{src}'; os.makedirs(od,exist_ok=True)
    if var=='static':
        np.savez(f'{od}/static.npz', lsm=ds['land_sea_mask'].values, z=ds['geopotential_at_surface'].values,
                 lat=ds.latitude.values, lon=ds.longitude.values); return
    for y in range(y0,y1+1):
        fn=f'{od}/{var}_{y}.npy'
        if os.path.exists(fn): continue
        x=ds[var].sel(time=slice(f'{y}-01-01',f'{y}-12-31T18')).transpose('time','longitude','latitude').values.astype(np.float32)
        np.save(fn+'.tmp.npy',x); os.rename(fn+'.tmp.npy',fn); print(var,y,x.shape,flush=True)

def fc(name,path,var,year,members=None):
    ds=xr.open_zarr(B+path,storage_options=SO)
    od=f'{ROOT}/data/fc/{name}'; os.makedirs(od,exist_ok=True)
    fn=f'{od}/{var}_{year}.npy'
    if os.path.exists(fn): print('exists',fn); return
    da=ds[var]
    td=da.prediction_timedelta.values
    if np.issubdtype(td.dtype,np.timedelta64): lh=(td/np.timedelta64(1,'h')).astype(int)
    else: lh=td.astype(int)
    li=[int(np.where(lh==l)[0][0]) for l in LEADS]
    inits=np.arange(np.datetime64(f'{year}-01-01T00'),np.datetime64(f'{year+1}-01-01T00'),np.timedelta64(12,'h'))
    tv=da.time.values; ti=np.searchsorted(tv,inits)
    ok=(ti<len(tv))&(tv[np.minimum(ti,len(tv)-1)]==inits)
    mdim=[d for d in ('number','sample','realization') if d in da.dims]
    order=['time']+mdim+['prediction_timedelta','longitude','latitude']
    da=da.transpose(*order)
    if mdim and members is not None: da=da.isel({mdim[0]:members})
    shp=(len(inits),)+((da.sizes[mdim[0]],) if mdim else ())+(len(LEADS),240,121)
    out=np.lib.format.open_memmap(fn+'.tmp.npy',mode='w+',dtype=np.float32,shape=shp); out[:]=np.nan
    idx=np.where(ok)[0]
    def job(k):
        i=idx[k]; x=da.isel(time=int(ti[i]),prediction_timedelta=li).values
        out[i]=x; return k
    t=time.time()
    with ThreadPoolExecutor(8) as ex:
        for n,k in enumerate(ex.map(job,range(len(idx)))):
            if n%100==0: print(name,var,year,n,len(idx),f'{time.time()-t:.0f}s',flush=True)
    out.flush(); del out; os.rename(fn+'.tmp.npy',fn)
    np.save(f'{od}/inits_{year}.npy',inits); print('done',fn,'missing inits',(~ok).sum(),flush=True)

if __name__=='__main__':
    a=sys.argv
    if a[1] in ('era5','hres_t0'): era5(a[2],int(a[3]) if len(a)>3 else 0,int(a[4]) if len(a)>4 else 0,a[1])
    else:
        mem=[int(m) for m in a[6].split(',')] if len(a)>6 else None
        fc(a[2],a[3],a[4],int(a[5]),mem)
