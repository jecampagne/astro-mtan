"""Partie sans torch de main_preprocessing_lsst.py : produit alerts_processed_<TAG>.parquet.
Usage: python make_parquet_lsst.py /path/to/ftransfer_lsst_xxx [diff|science]
"""
import sys
from prepare_lsst import load_lsst_alerts, select_objects

TOPIC_PATH = sys.argv[1]
FLUX_MODE = sys.argv[2] if len(sys.argv) > 2 else "diff"
TAG = TOPIC_PATH.rstrip('/').split('/')[-1].replace('-', '_')

df = load_lsst_alerts(TOPIC_PATH, flux_mode=FLUX_MODE, snr_min=5.0)
df = select_objects(df, min_total=10, min_per_band=3, min_bands=2,
                    min_duration_days=None, min_nights=3)

assert sorted(df['fid'].unique()) == [1, 2, 3, 4], \
    f"Bands present: {sorted(df['fid'].unique())} (expected 1..4)"

out = f'alerts_processed_{TAG}.parquet'
df.to_parquet(out)
print(f"saved {out}: {len(df):,} detections, {df.objectId.nunique():,} objects")
