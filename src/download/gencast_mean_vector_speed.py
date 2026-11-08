"""Speed of the GenCast ensemble-mean wind vector, sqrt(mean(u)^2 + mean(v)^2), at 1.5 deg for 2020.
The WeatherBench 2 GenCast mean dataset also contains 10m_wind_speed, which is the mean of member speeds.
Output: data/fc/gcvec/10m_wind_speed_2020.npy (init, lead, lon, lat), used as model 'gcvec' in record_events_15deg.py.
usage: python src/download/gencast_mean_vector_speed.py"""
import os, sys, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from download import wb2_15deg as download_wb2_15deg
P = 'gencast/2020-240x121_equiangular_with_poles_conservative_mean.zarr'
for c in ('u', 'v'):
    download_wb2_15deg.fc('gencastmean', P, f'10m_{c}_component_of_wind', 2020)
d = f'{download_wb2_15deg.ROOT}/data/fc'
u = np.load(f'{d}/gencastmean/10m_u_component_of_wind_2020.npy', mmap_mode='r')
v = np.load(f'{d}/gencastmean/10m_v_component_of_wind_2020.npy', mmap_mode='r')
os.makedirs(f'{d}/gcvec', exist_ok=True)
np.save(f'{d}/gcvec/10m_wind_speed_2020.npy', np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2))
shutil.copy(f'{d}/gencastmean/inits_2020.npy', f'{d}/gcvec/inits_2020.npy')
