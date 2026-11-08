#!/usr/bin/env bash
# Download WeatherBench 2 fields (anonymous access to gs://weatherbench2) and MADIS METAR observations.
# Roughly 500 GB in total. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
D15="python src/download/wb2_15deg.py"
D025="python src/download/wb2_025deg.py"
G=240x121_equiangular_with_poles_conservative
VARS="2m_temperature 10m_wind_speed"

# 1.5 degree truths
$D15 era5 static 0 0
for V in $VARS; do
  $D15 era5 "$V" 1979 2022
  $D15 hres_t0 "$V" 2016 2022
done

# 1.5 degree forecasts
for V in $VARS; do
  $D15 fc graphcast graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-$G.zarr "$V" 2018
  $D15 fc graphcast graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-$G.zarr "$V" 2020
  for Y in 2018 2020 2022; do
    $D15 fc pangu pangu/2018-2022_0012_$G.zarr "$V" $Y
    $D15 fc hres hres/2016-2022-0012-$G.zarr "$V" $Y
    $D15 fc ensmean ifs_ens/2018-2022-${G}_mean.zarr "$V" $Y
  done
  $D15 fc fuxi fuxi/2020-$G.zarr "$V" 2020
  $D15 fc gencastmean gencast/2020-${G}_mean.zarr "$V" 2020
  $D15 fc era5fc era5-forecasts/2018-$G.zarr "$V" 2018
  $D15 fc era5fc era5-forecasts/2020-$G.zarr "$V" 2020
  $D15 fc fgnmean fgn/2022-${G}_mean.zarr "$V" 2022
  $D15 fc gcv2 graphcast_v2/2022-$G.zarr "$V" 2022
  $D15 fc aurora aurora/2022-$G.zarr "$V" 2022
  $D15 fc gencast10 gencast/2020-$G.zarr "$V" 2020 0,1,2,3,4,5,6,7,8,9
  $D15 fc ens10 ifs_ens/2018-2022-$G.zarr "$V" 2020 0,1,2,3,4,5,6,7,8,9
done
python src/download/gencast_mean_vector_speed.py

# 0.25 degree land cells, lead times 12, 48 and 120 h
$D025 static
for V in $VARS; do
  $D025 truth era5 "$V" 1979 2021
  $D025 truth hres_t0 "$V" 2018 2020
  $D025 fc graphcast graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr "$V" 2018 12,48,120
  $D025 fc graphcast graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr "$V" 2020 12,48,120
  for Y in 2018 2020; do
    $D025 fc pangu pangu/2018-2022_0012_0p25.zarr "$V" $Y 12,48,120
    $D025 fc hres hres/2016-2022-0012-1440x721.zarr "$V" $Y 12,48,120
  done
done

# MADIS METAR stations (2 m temperature)
python src/download/metar.py 2001 2020
python src/events/station_records.py build
