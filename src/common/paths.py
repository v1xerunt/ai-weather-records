"""Repository-wide paths. Set AWR_ROOT to keep data outside the repository."""
import os

ROOT = os.environ.get('AWR_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = f'{ROOT}/data'          # 1.5 degree fields, event tables, station data
DATA025 = f'{ROOT}/data025'    # 0.25 degree land cells
