"""Download WB2 METAR (MADIS) station obs at nominal 00/06/12/18 UTC, selected columns -> data/metar/YEAR.parquet
usage: python src/download/metar.py Y0 Y1"""
import sys, os, gcsfs, pyarrow.parquet as pq, pyarrow as pa, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from common.paths import ROOT
B = 'weatherbench2/datasets/metar/metar-timeNominal-by-hour'
COLS = ['stationName', 'latitude', 'longitude', 'elevation', 'timeObs', 'timeNominal', 'temperature', 'temperatureDD',
        'windSpeed', 'windSpeedDD', 'reportType']
fs = gcsfs.GCSFileSystem(token='anon')
od = f'{ROOT}/data/metar'; os.makedirs(od, exist_ok=True)


def one(args):
    y, m, d, h = args
    p = f'{B}/year={y}/month={m}/day={d}/hour={h}/{y}-{m:02d}-{d:02d}T{h:02d}.parquet'
    try:
        with fs.open(p) as f:
            t = pq.read_table(f, columns=COLS)
        return t.to_pandas()
    except Exception:
        return None


for y in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    fn = f'{od}/{y}.parquet'
    if os.path.exists(fn): continue
    days = pd.date_range(f'{y}-01-01', f'{y}-12-31')
    jobs = [(y, t.month, t.day, h) for t in days for h in (0, 6, 12, 18)]
    with ThreadPoolExecutor(32) as ex:
        dfs = [x for x in ex.map(one, jobs) if x is not None]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(fn + '.tmp'); os.rename(fn + '.tmp', fn)
    print(y, len(df), 'rows', len(dfs), 'files of', len(jobs), flush=True)
