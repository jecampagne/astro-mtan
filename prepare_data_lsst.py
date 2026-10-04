"""
Version réduite de prepare_data.py : uniquement ce qui est nécessaire à
main_preprocessing_lsst.py (fonction prepare_data + ses classes de transformation).

Dépendances externes (inchangées) : utils.get_lc, mtan_utils.variable_time_collate_fn,
mtan_utils.subsample_timepoints_continuous_window.
"""
import random
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import get_lc
from mtan_utils import variable_time_collate_fn, subsample_timepoints_continuous_window


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_combined, data_Ids, transform=None):
        super().__init__()
        self._data_combined = data_combined
        self._data_Ids = data_Ids
        self.transform = transform

    def __len__(self):
        return self._data_combined.shape[0]

    def __getitem__(self, index):
        x = self._data_combined[index]
        y = self._data_Ids[index]
        if self.transform:
            x = self.transform(x)
        return x, y


def _too_few_points(observed_mask, min_datapoints_each_filter):
    """True if the light curve is too sparse to be truncated.
    Only bands actually observed in this light curve are considered."""
    counts = observed_mask.sum(0)
    counts = counts[counts > 0]
    return counts.numel() == 0 or bool(torch.any(counts < min_datapoints_each_filter))


class ContinuousTruncateLightCurve(object):
    """Randomly keeps a continuous window of a fraction of the observed time points
    (fraction drawn in `percentage_tp_to_sample_range`). Applied at runtime (train only)."""
    def __init__(self, percentage_tp_to_sample_range=(0.3, 0.5), dim=2, min_datapoints_each_filter=10):
        lo, hi = percentage_tp_to_sample_range
        if not (0.0 <= lo <= 1.0) or not (0.0 <= hi <= 1.0):
            raise ValueError('`percentage_tp_to_sample_range` values must be in [0.0, 1.0].')
        if hi < lo:
            raise ValueError('`percentage_tp_to_sample_range[0]` must be <= `percentage_tp_to_sample_range[1]`')
        self.percentage_tp_to_sample_range = percentage_tp_to_sample_range
        self.dim = dim
        self.min_datapoints_each_filter = min_datapoints_each_filter

    def __call__(self, x):
        # x: (num_points, dim + dim + 1) = [data | mask | time]
        observed_mask = x[:, self.dim:2 * self.dim]
        if _too_few_points(observed_mask, self.min_datapoints_each_filter):
            return x
        pct = np.random.uniform(low=self.percentage_tp_to_sample_range[0],
                                high=self.percentage_tp_to_sample_range[1])
        observed_data = x[:, :self.dim]
        observed_tp = x[:, -1]
        # subsample_timepoints_continuous_window works on BATCHED tensors (B, T, D) / (B, T):
        # add a batch dimension of 1 for this single light curve, then remove it.
        sub_data, sub_tp, sub_mask = subsample_timepoints_continuous_window(
            observed_data.clone().unsqueeze(0), observed_tp.clone().unsqueeze(0),
            observed_mask.clone().unsqueeze(0), percentage_tp_to_sample=pct)
        sub_data, sub_tp, sub_mask = sub_data.squeeze(0), sub_tp.squeeze(0), sub_mask.squeeze(0)
        return torch.cat((sub_data, sub_mask, sub_tp.unsqueeze(-1)), 1)


class apply_truncate_transform_random:
    """Applies a truncate transform with probability `p`."""
    def __init__(self, truncate_transform, p=0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')
        self.truncate_transform = truncate_transform
        self.p = p

    def __call__(self, x):
        return self.truncate_transform(x) if torch.rand(1) < self.p else x


def prepare_data(df_alerts, dim=2, train_size=0.7, train_batch_size=32, classify=False, activity=False,
                 convert_to_tensor=False, custom_train_min_time=None, custom_train_max_time=None,
                 magpsf_column='magpsf', sigmapsf_column='sigmapsf'):
    """Same behaviour/outputs as prepare_data.prepare_data (minus the AGN <3 months side-file).
    Pass `custom_train_min_time`/`custom_train_max_time` only when applying to new data."""
    device = 'cpu'          # no GPU needed to prepare the data
    time_in_hrs = True

    # --- Pass 1: global min/max times + per-LC mag stats
    min_time, max_time, duration_lcs, min_max_magdiffs, min_max_mags = np.inf, -np.inf, [], [], []
    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor,
                         normalize_times=False, max_time=None, min_time=None, time_in_hrs=time_in_hrs,
                         magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column)
        assert lc_data[0] == objId
        if lc_data[1][0].numpy()[0] < min_time:
            min_time = lc_data[1][0].numpy()[0]
        if lc_data[1][-1].numpy()[0] > max_time:
            max_time = lc_data[1][-1].numpy()[0]
        duration_lcs.append(float(lc_data[1][-1] - lc_data[1][0]))
        obs = lc_data[2]
        min_max_magdiffs.append(obs.max() - obs[obs != 0.0].min())   # 0.0 == unobserved
        min_max_mags.append((objId, obs[obs != 0.0].min().item(), obs.max().item()))
    print(f'Max and Min time values (hrs) across the dataset: {max_time}, {min_time}')

    # --- Pass 2: build the light curves
    total_data, total_objId, total_common_finkclasses = [], [], []
    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor,
                         normalize_times=False, local_time_normalization=False,
                         max_time=max_time, min_time=min_time, time_in_hrs=time_in_hrs)
        assert lc_data[0] == objId
        total_data.append(lc_data)
        total_objId.append(objId)
        total_common_finkclasses.append(lc_data[-1])

    data_min, data_max = None, None

    # --- Split (same seeds as the original => same split)
    train_data, test_data, train_data_objId, test_data_objId = train_test_split(
        total_data, total_objId, train_size=train_size, random_state=42, shuffle=True)
    train_data, val_data, train_data_objId, val_data_objId = train_test_split(
        train_data, train_data_objId, train_size=0.8, random_state=42, shuffle=True)

    if custom_train_min_time is None or custom_train_max_time is None:
        train_min_time = np.min([td[1].min() for td in train_data])
        train_max_time = np.max([td[1].max() for td in train_data])
        val_min_time = np.min([td[1].min() for td in val_data])
        val_max_time = np.max([td[1].max() for td in val_data])
        test_min_time = np.min([td[1].min() for td in test_data])
        test_max_time = np.max([td[1].max() for td in test_data])
    else:
        train_min_time = val_min_time = test_min_time = custom_train_min_time
        train_max_time = val_max_time = test_max_time = custom_train_max_time

    print(f'Time range (hrs) train: {train_min_time}, {train_max_time} | '
          f'val: {val_min_time}, {val_max_time} | test: {test_min_time}, {test_max_time}')

    # --- Collate (normalisation of the time with the TRAIN min/max, for all three splits)
    kw = dict(classify=classify, activity=activity, data_min=data_min, data_max=data_max,
              train_min_time=train_min_time, train_max_time=train_max_time)
    train_data_combined, train_data_Ids = variable_time_collate_fn(train_data, device, **kw)
    val_data_combined, val_data_Ids = variable_time_collate_fn(val_data, device, **kw)
    test_data_combined, test_data_Ids = variable_time_collate_fn(test_data, device, **kw)

    assert np.all(train_data_objId == train_data_Ids)
    assert np.all(test_data_objId == test_data_Ids)
    assert np.all(val_data_objId == val_data_Ids)

    print(f'train/val/test shapes: {tuple(train_data_combined.shape)}, '
          f'{tuple(val_data_combined.shape)}, {tuple(test_data_combined.shape)}')

    # --- Sanity check on sequence length
    seq_len_all = df_alerts.groupby('objectId', sort=False).size().loc[df_alerts['objectId'].unique()].values
    max_seq_len = int(seq_len_all.max())
    print(f'Max. sequence length: {max_seq_len}')
    assert max_seq_len == max(train_data_combined.shape[1], val_data_combined.shape[1], test_data_combined.shape[1])

    # --- Datasets / loaders
    transform = apply_truncate_transform_random(
        ContinuousTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim,
                                     min_datapoints_each_filter=10),
        p=0.5)
    train_dataset = MyDataSet(train_data_combined, train_data_Ids, transform=transform)
    val_dataset = MyDataSet(val_data_combined, val_data_Ids, transform=None)
    test_dataset = MyDataSet(test_data_combined, test_data_Ids, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    # finkclasses may be heterogeneous: pad before saving as a numpy array
    pad = len(max(total_common_finkclasses, key=len))
    total_common_finkclasses = np.array([i + [0] * (pad - len(i)) for i in total_common_finkclasses])

    return {
        "duration_lcs": np.array(duration_lcs),
        "seq_len_all": np.array(seq_len_all),
        "min_max_mags": np.array(min_max_mags),
        "min_max_magdiffs": np.array(min_max_magdiffs),
        "total_objIds": np.array(total_objId),
        "train_objIds": train_data_objId,
        "val_objIds": val_data_objId,
        "test_objIds": test_data_objId,
        "train_data_combined": train_data_combined,
        "val_data_combined": val_data_combined,
        "test_data_combined": test_data_combined,
        "total_common_finkclasses": total_common_finkclasses,
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim": dim,
        "train_val_test_min_max_times": np.array([train_min_time, train_max_time, val_min_time,
                                                  val_max_time, test_min_time, test_max_time]),
        "train_min_max_times": np.array([train_min_time, train_max_time]),
    }


# ---------------------------------------------------------------------------
# Compatibilité des pickles : torch.save(dataloader) référence les classes par
# leur nom de module. On les déclare comme appartenant à `prepare_data` pour que
# les .pth soient relisibles par tan_unsupervised.py / evaluate_unsupervised.py
# (qui font `import prepare_data`), sans avoir besoin de prepare_data_lsst.
# ---------------------------------------------------------------------------
if 'prepare_data' not in sys.modules:
    for _cls in (MyDataSet, ContinuousTruncateLightCurve, apply_truncate_transform_random):
        _cls.__module__ = 'prepare_data'
    sys.modules['prepare_data'] = sys.modules[__name__]
