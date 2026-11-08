# AI weather models forecast record extremes nearly as well as predictability allows

Data: [10.5281/zenodo.24045931](https://doi.org/10.5281/zenodo.24045931)

## Repository structure

```
├── environment.yml            conda environment for all analyses and figures
├── requirements.txt           the same dependencies for pip
├── scripts/
│   ├── download.sh            WeatherBench 2 fields and MADIS stations
│   ├── build_events.sh        record-event tables with the expected error from ordinary days
│   └── make_figures.sh        Fig. 1-5 and Extended Data figures
├── src/
│   ├── common/                grids, local records, climatology, ordinary-day statistics
│   ├── download/              data download
│   ├── events/                record events at 1.5 and 0.25 degrees and at weather stations
│   ├── analysis/              the tests reported in the paper
│   └── graphcast/             rerun of the public GraphCast checkpoint (separate environment)
└── figures/                   figure scripts
```

Data are read from `data/` and `data025/` in the repository root. Set `AWR_ROOT` to use another location.

## Setup

```bash
conda env create -f environment.yml
conda activate ai-weather-records
```

or `pip install -r requirements.txt` with Python 3.12.

## Reproducing the figures

Download `derived_data.tar` from Zenodo and unpack it in the repository root, which creates `data/` and `data025/`.

```bash
tar -xf derived_data.tar
bash scripts/make_figures.sh        # PDF and PNG files in output/
```

The figure scripts read the record-event tables directly. Panels that show summary statistics (perfect-model limits,
station-density gains, error components, rank correlations, recalibrated errors, wind residuals and the
same-network cutoff differences) take their values from the analysis scripts listed below.

## Full pipeline

```bash
bash scripts/download.sh            # about 500 GB
bash scripts/build_events.sh
```

| Result | Script |
|---|---|
| Expected error from ordinary days, Fig. 2b,c | `src/events/record_events_15deg.py`, `src/events/record_events_025deg.py`, `src/events/summarize_record_errors.py` |
| Perfect-model forecasts, Fig. 2d,e | `src/analysis/perfect_model.py` |
| Offset and regression terms of the expected error | `src/analysis/offset_regression_split.py` |
| Gain from verifying on the model's own analysis, Fig. 3c | `src/analysis/own_analysis_gain_vs_station_density.py` |
| Weather-station records, Fig. 3a, Extended Data Table 2 | `src/events/station_records.py`, `src/analysis/station_stratify.py` |
| Input-conditioned errors, Fig. 4b,c, Extended Data Fig. 2 | `src/analysis/input_conditioned_error.py`, `src/analysis/input_conditioned_error_multiyear.py` |
| Same network with an earlier training cutoff, Fig. 4d, Extended Data Table 7 | `src/graphcast/run_graphcast_2017.py`, `src/analysis/cutoff_same_network_records.py`, `src/analysis/cutoff_same_network_all_days.py` |
| Cutoff comparison across architectures | `src/analysis/cutoff_cross_architecture.py` |
| Case studies (Pacific Northwest 2021, United Kingdom 2022) | `src/analysis/case_studies.py` |
| Temperature steps towards records, Extended Data Table 5 | `src/analysis/steps_to_records.py` |
| Extreme wind increments | `src/analysis/increment_climatology.py`, `src/analysis/increment_tail_calibration.py` |
| Offset, predictability and amplitude components, Fig. 5a | `src/analysis/shortfall_components_025deg.py`, `src/analysis/shortfall_components_15deg.py` |
| Rank correlation among records, Fig. 5b | `src/analysis/rank_within_records.py` |
| Identical monotonic recalibration, Fig. 5c | `src/analysis/monotone_recalibration.py` |
| Local linear recalibration | `src/analysis/linear_recalibration.py` |
| Speed of the averaged wind vector, Fig. 5d | `src/download/gencast_mean_vector_speed.py` |
| Detection of the deepest records, Extended Data Table 6 | `src/analysis/deep_tail_detection.py` |
| Window sensitivity, Extended Data Table 4 | `src/events/record_events_15deg.py` with `GW`, `GEX`, `TAG` (see `scripts/build_events.sh`) |

Each script documents its arguments in its header, for example

```bash
python src/analysis/perfect_model.py gencast10 t2m max 12,24,48,72,120,168,240
python src/analysis/cutoff_cross_architecture.py 025 2020 t2m max era5 graphcast:2019 pangu:2017
python src/analysis/rank_within_records.py 025 2020 t2m max era5 hres graphcast,pangu
```

## Rerunning GraphCast trained to 2017

The rerun uses the public checkpoint
`GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz`
and the official [GraphCast code](https://github.com/google-deepmind/graphcast) in a separate GPU environment
(`src/graphcast/requirements.txt`). Place the checkpoint and its normalization statistics in `gc_params/`, or set
`GC_PARAMS`.

```bash
python src/graphcast/run_graphcast_2017.py validate         # compare with the published 2018 forecasts of this checkpoint
python src/graphcast/run_graphcast_2017.py run 2020 0,12    # 732 initializations, lead times 12, 48 and 120 h
```

The output is archived on Zenodo as `graphcast2017_rerun_2020.tar.gz`, with arrays of shape
(732 initializations, 3 lead times, 244,450 land cells) in the cell order of `data025/static.npz`.

## Licence and data sources

Code: MIT. Data on Zenodo: CC BY-NC-SA 4.0 for non-commercial research use, because the event tables contain values
derived from Pangu-Weather (weights licensed CC BY-NC-SA 4.0, commercial use forbidden) and from the IFS ensemble in
the TIGGE archive (research and education only).

- ERA5: Copernicus licence. Contains modified Copernicus Climate Change Service information 2026. Neither the European
  Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
- HRES forecasts and analyses, ERA5-initialized IFS forecasts and the IFS ensemble (TIGGE): based on data and products
  of the European Centre for Medium-Range Weather Forecasts (ECMWF), under ECMWF terms of use.
- GraphCast, GenCast and WeatherNext 2 (FGN) forecasts: Copyright Google LLC, CC BY 4.0.
- Pangu-Weather, FuXi and Aurora forecasts: obtained from WeatherBench 2.
- METAR observations: NOAA MADIS (no restrictions), via WeatherBench 2 (CC BY 4.0).
