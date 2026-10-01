"""
Equivalent of main_preprocessing.py for Fink/LSST alerts (g,r,i,z -> dim=4).
Produces the same files (dataloaders .pth, objIds .npy, time min/max, ...)
as expected by tan_unsupervised.py / evaluate_unsupervised.py.

Usage: python main_preprocessing_lsst.py /path/to/ftransfer_lsst_xxx [diff|science]
"""
import sys
import numpy as np
import torch
from prepare_lsst import load_lsst_alerts, select_objects
from prepare_data import prepare_data

TOPIC_PATH = sys.argv[1]
FLUX_MODE = sys.argv[2] if len(sys.argv) > 2 else "diff"
DIM = 4
TAG = TOPIC_PATH.rstrip('/').split('/')[-1].replace('-', '_')

df_alerts = load_lsst_alerts(TOPIC_PATH, flux_mode=FLUX_MODE, snr_min=5.0)
df_alerts = select_objects(df_alerts, min_total=10, min_per_band=3, min_bands=2)

# get_lc builds its channels with np.unique(fid) over the WHOLE dataframe: all 4 bands must be present.
assert sorted(df_alerts['fid'].unique()) == [1, 2, 3, 4], \
    f"Bands present: {sorted(df_alerts['fid'].unique())} (expected g,r,i,z = 1..4)"

df_alerts.to_parquet(f'alerts_processed_{TAG}.parquet')

data_obj = prepare_data(df_alerts, dim=DIM, train_size=0.8, train_batch_size=8, convert_to_tensor=True,
                        custom_train_min_time=None, custom_train_max_time=None)

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
print('Done. Now run tan_unsupervised.py with --dim 4 '
      f'--train_val_test_min_max_times_filename train_val_test_min_max_times_{TAG}.npy')
