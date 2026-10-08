"""
Seconde partie de main_preprocessing_lsst.py (machine avec torch/GPU).
Lit le parquet produit par make_parquet_lsst.py.

Usage: python main_preprocessing_lsst_from_parquet.py alerts_processed_<TAG>.parquet
           [--weight {invvar,n,sqrt_n,none}] [--weight-floor-mag 0.05]

--weight (default invvar): per-point weight of the data-fit term, stored as an extra channel of the tensors
  ([data | mask | weight | time]) and used by tan_unsupervised_lsst.py (--use-weights). A point obtained by
  merging several detections of the same night/band therefore counts more than a single detection.
  'none' reproduces the previous output exactly ([data | mask | time], same file names).
  With a weighting, the output tag gets the suffix _w<mode> so that nothing is overwritten.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from prepare_data_lsst import prepare_data
from prepare_lsst import WEIGHT_MODES, compute_weights, weight_summary

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('parquet', help='alerts_processed_<TAG>.parquet produced by make_parquet_lsst.py')
parser.add_argument('--weight', default='invvar', choices=WEIGHT_MODES,
                    help='per-point weighting of the likelihood (default: invvar)')
parser.add_argument('--weight-floor-mag', type=float, default=0.05,
                    help="intrinsic/model error floor (mag) in the 'invvar' weights (default: 0.05)")
cli = parser.parse_args()

PARQUET = cli.parquet
DIM = 4
TAG = os.path.basename(PARQUET)[:-len('.parquet')].replace('alerts_processed_', '', 1)
if cli.weight != 'none':
    TAG += f'_w{cli.weight}'

df_alerts = pd.read_parquet(PARQUET)
assert sorted(df_alerts['fid'].unique()) == [1, 2, 3, 4], \
    f"Bands present: {sorted(df_alerts['fid'].unique())} (expected g,r,i,z = 1..4)"

weight_column = None
if cli.weight != 'none':
    w = compute_weights(df_alerts, cli.weight, floor_mag=cli.weight_floor_mag)
    df_alerts = df_alerts.assign(weight=w)
    weight_column = 'weight'
    print(f"weighting '{cli.weight}' (floor {cli.weight_floor_mag} mag): "
          + weight_summary(w, df_alerts['n_combined'].values if 'n_combined' in df_alerts.columns else None))
else:
    print("no weighting ('none'): tensors are [data | mask | time]")

data_obj = prepare_data(df_alerts, dim=DIM, train_size=0.8, train_batch_size=8, convert_to_tensor=True,
                        custom_train_min_time=None, custom_train_max_time=None, weight_column=weight_column)

for split in ['train', 'val', 'test']:
    torch.save(data_obj[f'{split}_dataloader'], f'{split}_dataloader_{TAG}.pth')
    torch.save(data_obj[f'{split}_data_combined'], f'{split}_data_combined_{TAG}.pth')
    np.save(f'{split}_objIds_{TAG}.npy', data_obj[f'{split}_objIds'])
np.save(f'total_objIds_{TAG}.npy', data_obj['total_objIds'])
np.save(f'total_common_finkclasses_{TAG}.npy', data_obj['total_common_finkclasses'])
np.save(f'duration_lcs_{TAG}.npy', data_obj['duration_lcs'])
np.save(f'num_datapoints_lcs_{TAG}.npy', data_obj['seq_len_all'])
np.save(f'min_max_magdiffs_{TAG}.npy', data_obj['min_max_magdiffs'])
np.save(f'min_max_mags_{TAG}.npy', data_obj['min_max_mags'])
np.save(f'train_val_test_min_max_times_{TAG}.npy', data_obj['train_val_test_min_max_times'])
np.save(f'train_min_max_times_{TAG}.npy', data_obj['train_min_max_times'])
print('Done. Now run tan_unsupervised_lsst.py with --dim 4 '
      f'--train_val_test_min_max_times_filename train_val_test_min_max_times_{TAG}.npy '
      f'--dataset lsst_{TAG}'
      + (' [--use-weights is on by default; --no-use-weights ignores the weight channel]' if weight_column else ''))
