#!/usr/bin/env bash
# Record-event tables with the expected error from ordinary days. Run from the repository root after download.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
E15="python src/events/record_events_15deg.py"
E025="python src/events/record_events_025deg.py"
M2020=graphcast,pangu,fuxi,hres,ensmean,gencastmean,era5fc,gencast10:0,gencast10:1,gencast10:2,ens10:0,ens10:1,ens10:2
M2018=graphcast,pangu,hres,ensmean,era5fc

# 1.5 degree, two truths, heat, cold and wind records
for TRUTH in era5 hres_t0; do
  for VK in "t2m max" "t2m min" "ws max"; do
    $E15 2020 $VK $TRUTH $M2020
    $E15 2018 $VK $TRUTH $M2018
  done
done
$E15 2020 ws max era5 gcvec
$E15 2021 t2m max era5 pangu,hres
for TRUTH in era5 hres_t0; do $E15 2022 t2m max $TRUTH pangu,hres,ensmean,fgnmean,gcv2,aurora; done
$E15 2022 t2m min era5 pangu,hres,ensmean,fgnmean,gcv2,aurora

# sensitivity to the window of ordinary days (GW, GEX: half-widths in 12-h steps, default 30 and 6)
for Y in 2018 2020; do
  GW=30 GEX=14 TAG=_w30x14 $E15 $Y t2m max era5 graphcast,pangu,hres,ensmean
  GW=60 GEX=14 TAG=_w60x14 $E15 $Y t2m max era5 graphcast,pangu,hres,ensmean
done
# shorter record histories
for Y0 in 1999 2009; do
  REF0=$Y0 TAG=_h$Y0 $E15 2020 t2m max era5 graphcast,pangu,hres,ensmean,gencastmean
done

# 0.25 degree
python -c "import sys; sys.path.insert(0, 'src'); from common import grid025 as C; [(C.build_records(v), C.clim(v)) for v in ('t2m', 'ws')]"
for VK in "t2m max" "t2m min" "ws max"; do $E025 2020 $VK era5 graphcast,pangu,hres 12,48,120; done
$E025 2020 t2m max hres_t0 graphcast,pangu,hres 12,48,120

# weather stations
for YK in "2018 max" "2020 max" "2020 min"; do python src/events/station_records.py eval $YK; done
