"""Partie sans torch de main_preprocessing_lsst.py : produit alerts_processed_<TAG>.parquet.

Usage:
    python make_parquet_lsst.py /path/to/ftransfer_lsst_xxx [diff|science] [--combine | --no-combine]

--combine (défaut) : fusionne les détections d'un même objet, dans la même bande, la même nuit
                     (moyenne pondérée par 1/sigma^2 en flux).
--no-combine       : conserve toutes les détections (le fichier de sortie reçoit le suffixe
                     _nocombine pour ne pas écraser la version combinée).
"""
import argparse
from prepare_lsst import load_lsst_alerts, select_objects

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("topic_path", help="chemin du dossier ftransfer_lsst_xxx")
parser.add_argument("flux_mode", nargs="?", default="diff", choices=["diff", "science"])
parser.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True,
                    help="moyenne des mesures d'une même nuit dans la même bande (défaut: True)")
args = parser.parse_args()

TOPIC_PATH = args.topic_path
FLUX_MODE = args.flux_mode
COMBINE = args.combine
TAG = TOPIC_PATH.rstrip('/').split('/')[-1].replace('-', '_')
if not COMBINE:
    TAG += '_nocombine'

df = load_lsst_alerts(TOPIC_PATH, flux_mode=FLUX_MODE, snr_min=5.0, combine_nights=COMBINE)
df = select_objects(df, min_total=10, min_per_band=3, min_bands=2,
                    min_duration_days=None, min_nights=3)

assert sorted(df['fid'].unique()) == [1, 2, 3, 4], \
    f"Bands present: {sorted(df['fid'].unique())} (expected 1..4)"

out = f'alerts_processed_{TAG}.parquet'
df.to_parquet(out)
print(f"saved {out} (combine={COMBINE}): {len(df):,} detections, {df.objectId.nunique():,} objects")
