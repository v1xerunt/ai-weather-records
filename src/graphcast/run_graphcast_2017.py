"""Run the public GraphCast (ERA5 1979-2017, 0.25deg, 37 levels) checkpoint from WB2 ERA5 inputs.
  python src/graphcast/run_graphcast_2017.py validate                -> compares to WB2 graphcast/2018 for a few inits
  python src/graphcast/run_graphcast_2017.py run YEAR [hours=0,12]   -> data025/fc/gc2017/{2m_temperature,10m_wind_speed}_YEAR.npy  (init, lead[12,48,120], cell)
Outputs land-flat cells (data025/static.npz mask). Resumable: done-mask file."""
import os, sys, time, dataclasses, functools, threading, queue
import numpy as np, xarray, jax, haiku as hk
from concurrent.futures import ThreadPoolExecutor
from weathernext.utils import autoregressive, casting, checkpoint, data_utils, normalization, rollout
from weathernext.weathernext1_graph import graphcast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
P = os.environ.get('GC_PARAMS', f'{ROOT}/gc_params')   # checkpoint + normalization stats from the GraphCast release
E5 = 'gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2'
SO = {'token': 'anon'}
LEADS = [12, 48, 120]
NSTEP = max(LEADS) // 6

with open(f'{P}/GraphCast_ERA5_1979-2017_0.25_37.npz', 'rb') as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params, model_config, task_config = ckpt.params, ckpt.model_config, ckpt.task_config
stats = {k: xarray.load_dataset(f'{P}/{k}.nc').compute() for k in ('diffs_stddev_by_level', 'mean_by_level', 'stddev_by_level')}


def construct(model_config, task_config):
    p = graphcast.GraphCast(model_config, task_config)
    p = casting.Bfloat16Cast(p)
    p = normalization.InputsAndResiduals(p, **stats)
    return autoregressive.Predictor(p, gradient_checkpointing=True)


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    return construct(model_config, task_config)(inputs, targets_template=targets_template, forcings=forcings)


fwd = jax.jit(functools.partial(run_forward.apply, model_config=model_config, task_config=task_config))
run_fwd = lambda **kw: fwd(params=params, state={}, **kw)[0]

ds = xarray.open_zarr(E5, storage_options=SO)
VARS_IN = [v for v in task_config.input_variables if v in ds.data_vars and v not in ('geopotential_at_surface', 'land_sea_mask')]
STATIC = ds[['geopotential_at_surface', 'land_sea_mask']].load()
LEVELS = list(task_config.pressure_levels)
st = np.load(f'{ROOT}/data025/static.npz'); MASK = st['mask']


def fetch(init):
    """Download only the 2 input times + target-time TISR (small). Template is built in assemble()."""
    t_in = [init - np.timedelta64(6, 'h'), init]
    sub = ds[VARS_IN].sel(time=t_in, level=LEVELS)
    with ThreadPoolExecutor(16) as ex:
        arrs = dict(ex.map(lambda v: (v, sub[v].load()), VARS_IN))
    tt = [init + np.timedelta64(6 * k, 'h') for k in range(1, NSTEP + 1)]
    return xarray.Dataset(arrs), ds['toa_incident_solar_radiation'].sel(time=tt).load(), tt


def assemble(fetched):
    x, tisr, tt = fetched
    tgt = xarray.Dataset({v: (x[v].dims, np.full((NSTEP,) + x[v].shape[1:], np.nan, np.float32)) for v in x.data_vars},
                         coords={c: x.coords[c] for c in x.coords if c != 'time'}).assign_coords(time=tt)
    tgt['toa_incident_solar_radiation'] = tisr
    full = xarray.concat([x, tgt], dim='time')
    del tgt
    full = full.merge(STATIC)
    full = full.rename({'latitude': 'lat', 'longitude': 'lon'}).isel(lat=slice(None, None, -1))  # GraphCast wants lat ascending
    full = full.expand_dims('batch')
    full = full.assign_coords(datetime=(('batch', 'time'), full.time.values[None]))
    full = full.assign_coords(time=full.time.values - full.time.values[0])
    return full


def predict(batch):
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice('6h', f'{NSTEP*6}h'), **dataclasses.asdict(task_config))
    pred = rollout.chunked_prediction(run_fwd, rng=jax.random.PRNGKey(0), inputs=inputs,
                                      targets_template=targets * np.nan, forcings=forcings)
    out = {}
    for L in LEADS:
        p = pred.sel(time=np.timedelta64(L, 'h')).isel(batch=0).isel(lat=slice(None, None, -1))  # back to lat desc
        t2 = p['2m_temperature'].transpose('lat', 'lon').values
        ws = np.hypot(p['10m_u_component_of_wind'], p['10m_v_component_of_wind']).transpose('lat', 'lon').values
        out[L] = (t2, ws)
    return out


def prefetcher(inits, q):
    """Bounded prefetch: at most K downloads in flight; results handed over through a bounded queue."""
    from collections import deque
    K = int(os.environ.get('GC_PREFETCH', 4))
    with ThreadPoolExecutor(K) as ex:
        dq = deque()
        for i in inits:
            dq.append((i, ex.submit(fetch, i)))
            if len(dq) >= K:
                j, f = dq.popleft(); q.put((j, f.result()))
        while dq:
            j, f = dq.popleft(); q.put((j, f.result()))
    q.put(None)


def validate():
    wb = xarray.open_zarr('gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr', storage_options=SO)
    for s in ['2018-07-01T00', '2018-07-15T12', '2018-01-10T00']:
        init = np.datetime64(s); t0 = time.time()
        b = assemble(fetch(init)); t1 = time.time(); out = predict(b); t2 = time.time()
        for L in LEADS:
            w = wb['2m_temperature'].sel(time=init).isel(prediction_timedelta=int(np.where(np.asarray(wb.prediction_timedelta.values).astype('timedelta64[h]').astype(int) == L)[0][0]) if np.issubdtype(wb.prediction_timedelta.dtype, np.timedelta64) else int(np.where(wb.prediction_timedelta.values == L)[0][0])).transpose('lat', 'lon')
            w = w.isel(lat=slice(None, None, -1)).values if wb.lat.values[0] < wb.lat.values[-1] else w.values
            tr = ds['2m_temperature'].sel(time=init + np.timedelta64(L, 'h')).values
            o = out[L][0]
            print(f'{s} lead {L:3d}: rms(ours-WB2)={np.sqrt(np.mean((o-w)**2)):.3f}  land {np.sqrt(np.mean((o-w)[MASK]**2)):.3f} | '
                  f'rmse vs ERA5 ours {np.sqrt(np.mean((o-tr)**2)):.3f} WB2 {np.sqrt(np.mean((w-tr)**2)):.3f}', flush=True)
        print(f'  fetch {t1-t0:.0f}s predict {t2-t1:.0f}s', flush=True)


def run(year, hours=(0, 12)):
    od = f'{ROOT}/data025/fc/gc2017'; os.makedirs(od, exist_ok=True)
    inits = np.arange(np.datetime64(f'{year}-01-01T00'), np.datetime64(f'{year+1}-01-01T00'), np.timedelta64(12, 'h'))
    nc = int(MASK.sum())
    files = {}
    for v in ('2m_temperature', '10m_wind_speed'):
        fn = f'{od}/{v}_{year}.npy'
        if not os.path.exists(fn):
            a = np.lib.format.open_memmap(fn, mode='w+', dtype=np.float32, shape=(len(inits), len(LEADS), nc)); a[:] = np.nan; del a
        files[v] = np.load(fn, mmap_mode='r+')
    np.save(f'{od}/inits_{year}.npy', inits); np.save(f'{od}/leads.npy', np.array(LEADS))
    dfn = f'{od}/done_{year}.npy'
    done = np.load(dfn) if os.path.exists(dfn) else np.zeros(len(inits), bool)
    hr = (inits - inits.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
    todo = [i for i in range(len(inits)) if not done[i] and hr[i] in hours]
    print('todo', len(todo), flush=True)
    q = queue.Queue(maxsize=2)
    threading.Thread(target=prefetcher, args=([inits[i] for i in todo], q), daemon=True).start()
    t0 = time.time(); n = 0
    while (item := q.get()) is not None:
        init, fe = item; i = int(np.where(inits == init)[0][0])
        out = predict(assemble(fe)); del fe
        for li, L in enumerate(LEADS):
            files['2m_temperature'][i, li] = out[L][0][MASK]; files['10m_wind_speed'][i, li] = out[L][1][MASK]
        done[i] = True; n += 1
        rss = int([l for l in open('/proc/self/status') if l.startswith('VmRSS')][0].split()[1]) / 1e6
        if rss > 200:
            np.save(dfn, done); [a.flush() for a in files.values()]; print(f'RSS {rss:.0f} GB > 200 GB, abort', flush=True); os._exit(3)
        if n % 5 == 0:
            for a in files.values(): a.flush()
            np.save(dfn, done)
            print(f'{n}/{len(todo)} {str(init)} {(time.time()-t0)/n:.1f}s/init rss {rss:.0f}GB', flush=True)
    for a in files.values(): a.flush()
    np.save(dfn, done); print('finished', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        validate()
    else:
        run(int(sys.argv[2]), tuple(int(h) for h in sys.argv[3].split(',')) if len(sys.argv) > 3 else (0, 12))
