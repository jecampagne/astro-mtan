import io
import random
import numpy as np
import requests
import pandas as pd
from utils import get_lc, pad_rows_to_match_columns
from mtan_utils import variable_time_collate_fn, get_data_min_max, get_data_min_max_single_record, subsample_timepoints, subsample_timepoints_continuous_window

import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, TensorDataset, DataLoader

def get_tns_tde_alerts():
    tns_tde_objIds = ['ZTF24aaahxwr', 'ZTF24aaecooj', 'ZTF20aahmtso', 'ZTF22aafujzv', 'ZTF22aadesap', 'ZTF18aabdajx', 'ZTF21aanxhjv', 'ZTF22abegjtx']
    flags = [1, 1, 0, 0, 0, 0, 0, 0]  # 1 means use entire light curve since it contains few points already. 0 means need to manually select a subset.
    subset_indices = [(None, None), (None, None), (1,35), (0,25), (0,20), (0,9), (0,15), (0,21)]  # indices are 0:len(pdf), 0:len(pdf), 1:35, 0:25, etc.

    alert_poll_columns = ['objectId', 'candid', 'magpsf', 'sigmapsf', 'fid', 'jd', 'ra', 'dec',
        'tnsclass', 'cdsxmatch', 'roid', 'mulens', 'snn_snia_vs_nonia',
        'snn_sn_vs_all', 'rf_snia_vs_nonia', 'rf_kn_vs_nonkn', 'tracklet',
        'lc_features_g', 'lc_features_r', 'finkclass']
    # NOTE: The below is a fix because these caused problems while saving the dataframe into a parquet file we don't need these two columns.
    alert_poll_columns.remove('lc_features_g')
    alert_poll_columns.remove('lc_features_r')

    alerts = []

    total_data, total_objId, total_common_finkclasses = [], [], []
    for counter, tobjId in enumerate(tns_tde_objIds):
        X, Y = subset_indices[counter]

        r = requests.post(
          'https://fink-portal.org/api/v1/objects',
          json={
            'objectId': tobjId,
            'output-format': 'json'
          }
        )

        fid_column, objectId_column, jd_column = 'fid', 'i:objectId', 'i:jd'

        # Format output in a DataFrame
        pdf = pd.read_json(io.BytesIO(r.content))
        pdf = pdf[pdf[objectId_column] == tobjId].sort_values(by=jd_column)
        pdf.columns = pdf.columns.str[2:]  # this is required to match the column names of the dataframe obtained from polling the alerts.
        # Update DataFrame A to have the same columns as B, filling missing ones with NaN
        for column in alert_poll_columns:
            if column not in pdf.columns:
                pdf[column] = np.nan  # Add missing columns to pdf with NaN values

        # Ensure the order of columns in pdf matches that of alert_poll.
        pdf = pdf[alert_poll_columns]
        # I manually checked that these two columns are not in pdf but were present in alert_poll.
        pdf['finkclass'] = 'TNS (TDE)'
        pdf['tnsclass'] = 'TNS (TDE)'

        if X is not None and Y is not None:
            for filt in np.unique(pdf[fid_column]):
                # select data from one filter at a time
                maskFilt = pdf[fid_column] == filt    
                alerts.append(pdf[maskFilt][X:Y])
        else:
            alerts.append(pdf)

    alerts = pd.concat(alerts)
    return alerts

def add_tns_tde():  # TODO: Generalize this function to allow any object, not just TDEs. This will help in curating a confident, labelled dataset.
    """THIS IS A OLD FUNCTION. Use get_tns_tde_alerts instead."""
    tns_tde_objIds = ['ZTF24aaahxwr', 'ZTF24aaecooj', 'ZTF20aahmtso', 'ZTF22aafujzv', 'ZTF22aadesap', 'ZTF18aabdajx', 'ZTF21aanxhjv', 'ZTF22abegjtx']
    flags = [1, 1, 0, 0, 0, 0, 0, 0]  # 1 means use entire light curve since it contains few points already. 0 means need to manually select a subset.
    subset_indices = [(None, None), (None, None), (1,35), (0,25), (0,20), (0,9), (0,15), (0,21)]  # indices are 0:len(pdf), 0:len(pdf), 1:35, 0:25, etc.

    total_data, total_objId, total_common_finkclasses = [], [], []
    for counter, tobjId in enumerate(tns_tde_objIds):
        X, Y = subset_indices[counter]

        r = requests.post(
          'https://fink-portal.org/api/v1/objects',
          json={
            'objectId': tobjId,
            'output-format': 'json'
          }
        )

        fid_column, jd_column, magpsf_column, sigmapsf_column, objectId_column = 'i:fid', 'i:jd', 'i:magpsf', 'i:sigmapsf', 'i:objectId'

        # Format output in a DataFrame
        pdf = pd.read_json(io.BytesIO(r.content))
        pdf = pdf[pdf[objectId_column] == tobjId].sort_values(by=jd_column)

        if X is not None and Y is not None:
            jds, magpsfs, sigmapsfs, filters = [],[],[],[]
            for filt in np.unique(pdf[fid_column]):
                # select data from one filter at a time
                maskFilt = pdf[fid_column] == filt
                jds.append(pdf[maskFilt][jd_column][X:Y])
                magpsfs.append(pdf[maskFilt][magpsf_column][X:Y])
                sigmapsfs.append(pdf[maskFilt][sigmapsf_column][X:Y])
                for _ in range(Y-X):
                    filters.append(filt)

            jds = np.expand_dims(np.array(jds).flatten(), 1)
            magpsfs = np.expand_dims(np.array(magpsfs).flatten(), 1)
            sigmapsfs = np.expand_dims(np.array(sigmapsfs).flatten(), 1)
            df = pd.DataFrame(np.hstack((jds, magpsfs, sigmapsfs)))
            df.columns = ['i:jd', 'i:magpsf', 'i:sigmapsf']
            df['i:fid'] = filters
            df['i:objectId'] = tobjId
            df['i:finkclass'] = 'TDE'

            # NOTE: IMPORTANT CAVEAT: We have decided to use global normalization and we are finding the max and min times in prepare_data. Those min and max times will not account for these additional cases. We are simply assuming the max times of these light curves will mostly be less than the max times calculated inside prepare_data. If this is not the case, the result is that the normalized time value for this additional light curve will have max value > 1. This may be fine assuming not many such cases will be present. In the below lines, max_time is not defined in this function, but sincee we use this function inside prepare_data, that global context of the variable will be used. Same with min times.

            # ALSO NOTE: We are no longer using add_tns_tde so the above caveats are not to worry about.

            lc_data = get_lc(df, tobjId, fid_column=fid_column, jd_column=jd_column, magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column, objectId_column=objectId_column, finkclass_column='i:finkclass', convert_to_tensor=True, normalize_times=True, local_time_normalization=False, max_time=max_time, min_time=min_time)
        else:
            lc_data = get_lc(pdf, tobjId, fid_column=fid_column, jd_column=jd_column, magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column, objectId_column=objectId_column, finkclass_column=None, convert_to_tensor=True, normalize_times=True, local_time_normalization=False, max_time=max_time, min_time=min_time)

        total_data.append(lc_data)
        assert lc_data[0] == tobjId
        total_objId.append(tobjId)
        total_common_finkclasses.append(lc_data[-1] if lc_data[-1] != [None] else 'TDE')

    return total_data, total_objId, total_common_finkclasses


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_combined, data_Ids, transform=None):
        super(MyDataSet, self).__init__()
        # store the raw tensors
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
    Only bands that are actually observed in this light curve are considered
    (with 4 bands griz, one band is often totally unobserved)."""
    counts = observed_mask.sum(0)
    counts = counts[counts > 0]
    return counts.numel() == 0 or bool(torch.any(counts < min_datapoints_each_filter))

class TruncateLightCurve(object):
    """Takes in a light curve, randomly selects percentage of observed time points to sample within the range
    provided by `percentage_tp_to_sample_range` and selects observed time points randomly from the
    light curve. This function is only applied at runtime to light curves with >=min_datapoints_each_filter
    data points in its light curve, summed across all bands."""
    def __init__(self, percentage_tp_to_sample_range=(0.3, 0.5), dim=2, min_datapoints_each_filter=10):
        if not (0.0 <= percentage_tp_to_sample_range[0] <= 1.0):
            raise ValueError('Lower end probability in `percentage_tp_to_sample_range` should be a floating point value in the interval [0.0, 1.0].')
        if not (0.0 <= percentage_tp_to_sample_range[1] <= 1.0):
            raise ValueError('Higher end probability in `percentage_tp_to_sample_range` should be a floating point value in the interval [0.0, 1.0].')
        if percentage_tp_to_sample_range[1] < percentage_tp_to_sample_range[0]:
            raise ValueError("`percentage_tp_to_sample_range[0]` must be less than or equal to `percentage_tp_to_sample_range[1]`")

        self.percentage_tp_to_sample_range = percentage_tp_to_sample_range
        self.dim = dim
        self.min_datapoints_each_filter = min_datapoints_each_filter

    def __call__(self, x):
        # Shape of x: (num_points, dim+dim+1)
        observed_mask = x[:, self.dim:2 * self.dim]
        # Check if no. of datapoints is <self.min_datapoints_each_filter across any filter
        # If so, not sufficient points to truncate the light curve, so don't apply truncate transformation.
        if _too_few_points(observed_mask, self.min_datapoints_each_filter):
            return x
        else:
            #if torch.rand(1) >= self.p:
            #    return x
            percentage_tp_to_sample = np.random.uniform(low=self.percentage_tp_to_sample_range[0], high=self.percentage_tp_to_sample_range[1])
            observed_data = x[:, :self.dim]
            observed_tp = x[:, -1]

            # NOTE: `percentage_tp_to_sample` in subsample_timepoints would sample time points combined across all bands, not every band. So if `percentage_tp_to_sample=0.3` for a light curve with 20 points combined across all bands, this function will randomly sample 6 points across all bands and it's possible one of the bands get <3 points, although the chances are small if min_datapoints_each_filter is a high value (min_datapoints_each_filter=10 may be a good minimum value to ensure chances of any band receiving <3 points is small, assuming two bands).
            subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(observed_data.clone(), observed_tp.clone(), observed_mask.clone(), percentage_tp_to_sample=percentage_tp_to_sample)
            subsampled_sample = torch.cat((subsampled_data, subsampled_mask, subsampled_tp.unsqueeze(-1)), 1)

            return subsampled_sample


class ContinuousTruncateLightCurve(object):
    """Takes in a light curve, randomly selects percentage of observed time points to sample within the range
    provided by `percentage_tp_to_sample_range` and selects continuous observed time points from the
    light curve. It is ensured that the continuous time points selected don't fall outside the observed time range.
    This function is only applied at runtime to light curves with >=min_datapoints_each_filter data points in its
    light curve, summed across all bands."""
    def __init__(self, percentage_tp_to_sample_range=(0.3, 0.5), dim=2, min_datapoints_each_filter=10):
        #if not(0.0 <= p <= 1.0):
        #    raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')

        if not (0.0 <= percentage_tp_to_sample_range[0] <= 1.0):
            raise ValueError('Lower end probability in `percentage_tp_to_sample_range` should be a floating point value in the interval [0.0, 1.0].')
        if not (0.0 <= percentage_tp_to_sample_range[1] <= 1.0):
            raise ValueError('Higher end probability in `percentage_tp_to_sample_range` should be a floating point value in the interval [0.0, 1.0].')
        if percentage_tp_to_sample_range[1] < percentage_tp_to_sample_range[0]:
            raise ValueError("`percentage_tp_to_sample_range[0]` must be less than or equal to `percentage_tp_to_sample_range[1]`")

        self.percentage_tp_to_sample_range = percentage_tp_to_sample_range 
        self.dim = dim
        self.min_datapoints_each_filter = min_datapoints_each_filter

    def __call__(self, x):
        # Shape of x: (num_points, dim+dim+1)
        observed_mask = x[:, self.dim:2 * self.dim]
        # Check if no. of datapoints is <self.min_datapoints_each_filter across any filter
        # If so, not sufficient points to truncate the light curve, so don't apply truncate transformation.
        if _too_few_points(observed_mask, self.min_datapoints_each_filter):
            return x
        else:
            #if torch.rand(1) >= self.p:
            #    return x
            percentage_tp_to_sample = np.random.uniform(low=self.percentage_tp_to_sample_range[0], high=self.percentage_tp_to_sample_range[1])
            #n_to_sample = np.random.randint(low=self.n_to_sample_range[0], high=self.n_to_sample_range[1])
            observed_data = x[:, :self.dim]
            observed_tp = x[:, -1]

            subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints_continuous_window(observed_data.clone(), observed_tp.clone(), observed_mask.clone(), percentage_tp_to_sample=percentage_tp_to_sample)
            subsampled_sample = torch.cat((subsampled_data, subsampled_mask, subsampled_tp.unsqueeze(-1)), 1)

            return subsampled_sample


class apply_truncate_transforms_random:
    """Takes in two truncate transforms as input, selects one among them randomly, and then apply it with probability, `p`"""
    def __init__(self, truncate_transform_A, truncate_transform_B, p=0.5):
        if not(0.0 <= p <= 1.0):
            raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')
        self.truncate_transform_A = truncate_transform_A
        self.truncate_transform_B = truncate_transform_B
        self.p = p

    def __call__(self, x):
        transform_to_apply = self.truncate_transform_A if random.choice([0, 1]) == 0 else self.truncate_transform_B
        # print(type(transform_to_apply).__name__)  # to see which truncate transform is actually being applied
        return transform_to_apply(x) if torch.rand(1) < self.p else x


class apply_truncate_transform_random:
    """Takes in a truncate transform and apply it with probability `p`"""
    def __init__(self, truncate_transform, p=0.5):
        if not(0.0 <= p <= 1.0):
            raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')
        self.truncate_transform = truncate_transform
        self.p = p

    def __call__(self, x):
        return self.truncate_transform(x) if torch.rand(1) < self.p else x


### Added code for the AGN truncation experiment on the test set
class TruncateFirstXMonthsLightCurve(object):
    """Keeps only the first X months of a light curve by removing all time points
    that fall outside [0, truncate_months/max_time_months] in normalized time.
    Always returns the truncated light curve, even if fewer than
    min_datapoints_each_filter points remain after truncation.

    Args:
        truncate_months (float): Number of months to truncate from the start.
        min_time_months (float): Total time span corresponding to observed_tp=0.0, in months.
        max_time_months (float): Total time span corresponding to observed_tp=1.0, in months.
        dim (int): Number of data dimensions (e.g. 2 for g and r bands).
        min_datapoints_each_filter (int): Minimum datapoints per filter to apply truncation.
            If not met on the ORIGINAL LC, return original x unchanged.
    """
    def __init__(self, truncate_months, max_time_months, min_time_months, dim=2, min_datapoints_each_filter=10):
        if truncate_months <= 0:
            raise ValueError("`truncate_months` must be positive.")
        total_span_months = max_time_months - min_time_months
        if truncate_months >= total_span_months:
            raise ValueError("`truncate_months` must be less than total span (max_time - min_time).")

        self.truncate_threshold = truncate_months / total_span_months
        self.truncate_months = truncate_months
        self.dim = dim
        self.min_datapoints_each_filter = min_datapoints_each_filter

    def __call__(self, x):
        # Shape of x: (num_points, dim + dim + 1)
        # Columns: [observed_data (dim) | observed_mask (dim) | observed_tp (1)]
        observed_mask = x[:, self.dim:2 * self.dim]
        observed_tp   = x[:, -1]

        # Only check min_datapoints on the ORIGINAL LC before truncation
        if _too_few_points(observed_mask, self.min_datapoints_each_filter):
            return x

        # Keep only time points WITHIN the first X months
        keep_mask = observed_tp <= self.truncate_threshold
        truncated_x = x[keep_mask]

        return truncated_x


def prepare_data(df_alerts, dim=2, train_size=0.7, train_batch_size=32, classify=False, activity=False, convert_to_tensor=False, custom_train_min_time=None, custom_train_max_time=None, magpsf_column='magpsf', sigmapsf_column='sigmapsf'):#, truncate_agn_months=None):#, remove_less_than_3months_agns=False, truncate_agn_months=None):
    """
    Pass in `custom_train_min_time` and `custom_train_max_time` if applying on totally new set of polled alerts. In that case, set this to thecorresponding to training.
    """
    """
    if remove_less_than_3months_agns:
        print('Removing AGNs with <3months duration')
        print(f'No. of objectIds before <3month removal: {len(df_alerts["objectId"].unique())}')
        # Remove AGNs where the time difference between the first and last alert < 91 days
        durations = df_alerts.groupby('objectId')['jd'].transform(lambda x: x.max() - x.min())
        to_drop_mask = (df_alerts['finkclass'] == 'custom_agn') & (durations < 91)
        df_alerts = df_alerts[~to_drop_mask]
        #df_alerts = df_alerts.groupby('objectId').filter(
        #    lambda x: (x['jd'].max() - x['jd'].min()) > 91
        #)
        print(f'No. of objectIds after <3month removal: {len(df_alerts["objectId"].unique())}')


    if truncate_agn_months is not None:
        print(f"Truncating AGNs to first {truncate_agn_months} months...")
        print(f'No. of objectIds before truncation: {len(df_alerts["objectId"].unique())}')

        # Calculate t0 (first detection) for every object
        t0s = df_alerts.groupby('objectId')['jd'].transform('min')

        # Define limit (X months converted to days)
        limit_days = truncate_agn_months * 30.

        # Logic: KEEP row if (NOT an AGN) OR (IS an AGN and current_time <= t0 + limit)
        is_agn = (df_alerts['finkclass'] == 'custom_agn')
        within_time = (df_alerts['jd'] <= (t0s + limit_days))

        # Apply the filter: We keep it if it's NOT an AGN, or if it IS an AGN within the window
        df_alerts = df_alerts[~is_agn | within_time].copy()

        print(f'No. of objectIds after truncation of {truncate_agn_months} months: {len(df_alerts["objectId"].unique())}')
    """
    durations = df_alerts.groupby('objectId')['jd'].transform(lambda x: x.max() - x.min())
    short_agns_mask = (df_alerts['finkclass'] == 'custom_agn') & (durations < 91)
    agns_with_less_than_3months_ids = np.array(df_alerts[short_agns_mask]['objectId'].unique())
    np.save('agns_with_less_than_3months_ids.npy', agns_with_less_than_3months_ids)
    print(f'{len(agns_with_less_than_3months_ids)} AGNs with <=3 months, saved to agns_with_less_than_3months_ids.npy')

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'  # We don't require GPU fr preparing the data but only for training.
    time_in_hrs = True

    # Since we aim to do global time normalization, we first need to find the max and min times (the absolute values and not the no. of datapoints) across the entire dataset. **NOTE: This loop should NOT be used if applying the model on a totally new data instance since we must use the min_time/max_time calculated during training and not calculate it again.**
    # FIRST, we do the loop only to find the min/max times across the dataset. Then the second loop performs the global time normalization using the max time found in the first loop.
    # min_max_mags to store the min/max mag so that these values can be used to unnormalize the light curves later. This is because mag normalization is locally done for each lc. We don't need to save the min/max values for the time (x-axis) since time is normalized globally, so a singel value across the dataset suffices.
    min_time, max_time, duration_lcs, min_max_magdiffs, min_max_mags = np.inf, -np.inf, [], [], []
    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor, normalize_times=False, max_time=None, min_time=None, time_in_hrs=time_in_hrs, magpsf_column=magpsf_column, sigmapsf_column=sigmapsf_column)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
        assert lc_data[0] == objId
        #assert lc_data[1][0] == 0.0  # Because lc_data[1] is the time array and it must start with zero because we use make_first_time_zero=True above.
        if lc_data[1][0].numpy()[0] < min_time:
            min_time = lc_data[1][0].numpy()[0]
        if lc_data[1][-1].numpy()[0] > max_time:
            max_time = lc_data[1][-1].numpy()[0]
        duration_lcs.append(float(lc_data[1][-1] - lc_data[1][0]))
        _lc_data_obs = lc_data[2]
        min_max_magdiffs.append(_lc_data_obs.max() - _lc_data_obs[_lc_data_obs != 0.0].min())  # Ignoring zero values for min because zero values mean unobserved.
        min_max_mags.append((objId, _lc_data_obs[_lc_data_obs != 0.0].min().item(), _lc_data_obs.max().item()))
   
    #assert min_time = 0.0   # Because first time is always zero for all lcs because we use make_first_time_zero=True.
    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the dataset: {max_time}, {min_time}')

    total_data, total_objId, total_common_finkclasses = [], [], []
    for objId in df_alerts['objectId'].unique():
        # NOTE: It's important to note that here we use normalize_times=False since we want min_time and max_time calculated on the specific dataset (train, val, OR test), just as done in mTAN phyionet data preprocessing. This normalization is done in variable_time_collate_fn.
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor, normalize_times=False, local_time_normalization=False, max_time=max_time, min_time=min_time, time_in_hrs=time_in_hrs)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
        total_data.append(lc_data)
        assert lc_data[0] == objId
        total_objId.append(objId)
        total_common_finkclasses.append(lc_data[-1])

    ################################################
    # Add cases manually. Currently we only add TDEs
    # EDIT: We don't need this since I am adding these cases in the processed alerts dataframe directly. Otherwise it becomes difficult to access these alerts for post-testing analysis
    #tns_tde_total_data, tns_tde_total_objId, tns_tde_total_common_finkclasses = add_tns_tde()
    #total_data.extend(tns_tde_total_data)
    #total_objId.extend(tns_tde_total_objId)
    #total_common_finkclasses.extend(tns_tde_total_common_finkclasses)
    ################################################

    #data_min, data_max = get_data_min_max(total_data)
    #print(f'data_min, data_max: {data_min, data_max}')
    #data_min, data_max = data_min.to(device), data_max.to(device)
    data_min, data_max = None, None  # Since we don't use min/max calculated across the entire train/val/test dataset.

    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId --> can do for classification, not required for unsupervised learning.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output since I am using train_test_independently for splitting the data and the objIds.
    # ensure multiple runs of label encoding on the same number gives the same encoded value. UPDATE: Not applicable now since we are not encoding the labels anymore.
    # We are encoding the objectIds just for efficiency because string types may not be efficient with PyTorch.
    # NOTE: I commented the below two lines since I am thinking the labels need NOT be encoded, and can keep it as strings only.
    #le = preprocessing.LabelEncoder()
    #total_objId_encoded = le.fit_transform(total_objId)  # use le.inverse_transform to get the string from the encoded value.

    train_data, test_data, train_data_objId, test_data_objId = train_test_split(total_data, total_objId, train_size=train_size, random_state=42, shuffle=True)
    train_data, val_data, train_data_objId, val_data_objId = train_test_split(train_data, train_data_objId, train_size=0.8, random_state=42, shuffle=True)

    print('DEBUG: train_data and test_data last time printing for a few cases.')
    for i, td in enumerate(train_data):
        if i == 5:
            break
        print(td[1][-1])

    for i, td in enumerate(test_data):
        if i == 5:
            break
        print(td[1][-1])

    if custom_train_min_time is None or custom_train_max_time is None:
        # Find the min and max times for train, val, and test sets.
        train_min_time = np.min([td[1].min() for td in train_data])
        train_max_time = np.max([td[1].max() for td in train_data])
        val_min_time = np.min([td[1].min() for td in val_data])
        val_max_time = np.max([td[1].max() for td in val_data])
        test_min_time = np.min([td[1].min() for td in test_data])
        test_max_time = np.max([td[1].max() for td in test_data])
    else:
        train_min_time, train_max_time, val_min_time, val_max_time, test_min_time, test_max_time = custom_train_min_time, custom_train_max_time, custom_train_min_time, custom_train_max_time, custom_train_min_time, custom_train_max_time

    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the train dataset: {train_max_time}, {train_min_time}')
    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the val dataset: {val_max_time}, {val_min_time}')
    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the test dataset: {test_max_time}, {test_min_time}')


    train_data_combined, train_data_Ids = variable_time_collate_fn(train_data, device, classify=classify, activity=activity, data_min=data_min, data_max=data_max, train_min_time=train_min_time, train_max_time=train_max_time)
    val_data_combined, val_data_Ids = variable_time_collate_fn(val_data, device, classify=classify, activity=activity,
                                                      data_min=data_min, data_max=data_max, train_min_time=train_min_time, train_max_time=train_max_time)
    test_data_combined, test_data_Ids = variable_time_collate_fn(test_data, device, classify=classify, activity=activity,
                                                      data_min=data_min, data_max=data_max, train_min_time=train_min_time, train_max_time=train_max_time)

    assert np.all(train_data_objId == train_data_Ids)
    assert np.all(test_data_objId == test_data_Ids)
    assert np.all(val_data_objId == val_data_Ids)
    
    # Q) Instead of inserting zero in the observed values array where no observed value exists, is it better to put a sufficient low mag instead, like 25?
    # Answer: I have confirmed that training, validation, and testing does NOT get affected by keeping unobserved values as 0 or 23 because these are essentially masked anyways.

    print(train_data_combined.shape, len(train_data_Ids))

    print(f'train_data_combined.shape, val_data_combined.shape, test_data_combined.shape: {train_data_combined.shape, val_data_combined.shape, test_data_combined.shape}')
    print('Printing train_data_combined[0, :, -1], test_data_combined[0, :, -1] => these are the time values (after all processing and to be used in the model) where the min value must be zero and maximum value must be one. Max value can also be less than one, but must not be greater than one.')
    print(train_data_combined[0, :, -1], test_data_combined[0, :, -1])

    ##### A quick check #####
    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)  # max_seq_len does not denote the maximum time duration, i.e., if max_seq_len = 60, it doesn't mean 60 hours/minutes.
    print(f'Max. sequence length (or the max no. of datapoints of lightcurves) in the dataset (max_seq_len): {max_seq_len}')
    assert max_seq_len == max([train_data_combined.shape[1], val_data_combined.shape[1], test_data_combined.shape[1]])
    print('max_seq_len AND THE MAX(1ST DIMENSION OF TRAIN_DATA_COMBINED, VAL_DATA_COMBINED, TEST_DATA_COMBINED) MUST BE SAME -- CHECK THAT')
    #########################

    # Create a tensor dataset just for the sake of storing the objectIds corresponding to each light curve, which may be helpful downstream. Especially for analysis while testing.
    #print(train_data_objId)
    #print(train_data_combined.shape)
    #train_data_combined = Dataset(train_data_combined, train_data_objId)
    #val_data_combined = Dataset(val_data_combined, val_data_objId)
    #test_data_combined = Dataset(test_data_combined, test_data_objId)

    # Below four lines added latest by me.
    #transform = SemiRandomTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.5), p=0.5, dim=dim, min_datapoints_each_filter=10)
    #transform = SemiRandomContinuousTruncateLightCurve(n_to_sample_range=(6, 20), p=0.5, dim=2, min_datapoints_each_filter=10)
    #transform = apply_truncate_transforms_random(
    #    ContinuousTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
    #    TruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
    #    p=0.5
    #)
    
    transform = apply_truncate_transform_random(
        ContinuousTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
        p=0.5
    )
    train_dataset = MyDataSet(train_data_combined, train_data_Ids, transform=transform)
    val_dataset = MyDataSet(val_data_combined, val_data_Ids, transform=None)
    """
    ### Added code for AGN truncation experiment
    train_val_test_min_max_times = np.load('train_val_test_min_max_times.npy')
    train_min_time, train_max_time = train_val_test_min_max_times[0], train_val_test_min_max_times[1]

    _test_transform_experiment = TruncateFirstXMonthsLightCurve(
        truncate_months=6,
        max_time_months=train_max_time/(24 * 30),
        min_time_months=train_min_time/(24 * 30),
        dim=2,
    )
    """
    test_dataset = MyDataSet(test_data_combined, test_data_Ids, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    # Since total_common_finkclass may be heterogenous, we pad them just for convenience before saving as an numpy array.
    # Credit for below code: https://stackoverflow.com/a/43146373
    # TODO: Padding can be avoided if you use np.savez. Can keep total_common_finkclasses as a list here and in the main_processing.py, can use np.savez instead. LOW_PRIORITY
    pad = len(max(total_common_finkclasses, key=len))
    total_common_finkclasses = np.array([i + [0]*(pad-len(i)) for i in total_common_finkclasses])

    #train_data_objId_raw = le.inverse_transform(train_data_objId)
    #val_data_objId_raw = le.inverse_transform(val_data_objId)
    #test_data_objId_raw = le.inverse_transform(test_data_objId)
 
    data_obj = {
        #"final_data": np.array(total_data),  # This may give error since total_data is a list containing variable length entries.
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
        #"total_objIds_encoded": np.array(total_objId_encoded),
        "total_common_finkclasses": total_common_finkclasses,
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim": dim,
        "train_val_test_min_max_times": np.array([train_min_time, train_max_time, val_min_time, val_max_time, test_min_time, test_max_time]),
        "train_min_max_times": np.array([train_min_time, train_max_time])
    }

    return data_obj




def prepare_data_old(df_alerts, dim=2, train_size=0.8, train_batch_size=32):
    """

    dim: no. of channels/passbands in the light curve.
    """
    # Mine specific cases and add them to the data
    # simbad_CataclyV_objId_list = ['ZTF18abucxou']
    # tns_cv_objId_list = []
    tns_tde_objId_list = [
        'ZTF24aaecooj', 'ZTF18aahqkbt', 'ZTF19accskvu', 'ZTF22aafujzv', 'ZTF18aabdajx', 'ZTF24aaahxwr',
        'ZTF23abvzeqp', 'ZTF18aabtxvd', 'ZTF20aahmtso', 'ZTF22aadesap', 'ZTF20abwtifz', 'ZTF21aanxhjv',
        'ZTF18achzddr', 'ZTF23abohtqf', 'ZTF19abzrhgq', 'ZTF23abgnxfv', 'ZTF22abegjtx', 'ZTF17aaazdba'
    ]  # These 18 objects are obtained from the fink portal (https://fink-portal.org/; class="(TNS) TDE"). TDEs are not present in SIMBAD.
    # Out of these, I see these to show a good peak in both bands and can be selected and others can be discarded: tns_tde_objIds = ['ZTF24aaahxwr', 'ZTF24aaecooj', 'ZTF20aahmtso', 'ZTF22aafujzv', 'ZTF22aadesap', 'ZTF18aabdajx', 'ZTF21aanxhjv', 'ZTF22abegjtx']
    tns_novae_objId_list = []
    simbad_novae_objId_list = []

    # get data for many objects
    r = requests.post(
      'https://fink-portal.org/api/v1/objects',
      json={
        'objectId': ','.join(tns_tde_objId_list),
        'output-format': 'json'
      }
    )

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))
    # pdf[['i:jd', 'i:magpsf', 'i:sigmapsf']]

    # These below conditions are the same as used for polling the alerts, except the ndethist condition (see the commented line)
    # The ndethist condition may not be appropriate here. Since these are full light curves, ndethist will be > 80 in non-trivial no. of cases. Instead, we manually look at the light curves and select subregions ourselves, majorly around the event of interest.
    conditions = (
        (pdf['i:nbad'] == 0)  # 1142
        & (pdf['i:fwhm'] <= 5)  # 1136
        & (pdf['i:elong'] <= 1.2)  # 1030
        & (pdf['i:magdiff'].abs() <= 0.1)  # 586
        & ((pdf['i:drb'] > 0.8) & (pdf['i:rb'] > 0.8))  # 551
        #& ((pdf['i:ndethist'] > 2) & (pdf['i:ndethist'] < 80))  # 148
        & (pdf['i:isdiffpos'] == "t")  # 146
        & ((pdf['i:ssdistnr'] >= 10) | (pdf['i:ssdistnr'] <= 0))  # 146
        & (~((pdf['i:sgscore1'] > 0.5) & (pdf['i:distpsnr1'] < 1.5) & (pdf['i:distpsnr1'] >= 0)))  # 146
        & (pdf['i:jd'] - pdf['i:jdstarthist'] > 30/60/24)  # 144
    )
    print(f'No. of rows of additional alerts (before): {len(pdf)}')
    pdf_filtered = pdf[conditions]
    print(f'No. of rows of additional alerts (after): {len(pdf_filtered)}')

    pdf_filtered_filtered = pdf_filtered.groupby('i:objectId').filter(
        lambda group: (len(group[group['i:fid'] == 1]) >= 3) or (len(group[group['i:fid'] == 2]) >= 3)
    )

    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        # lc_data = get_lc(df_alerts, objId)  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId. 5 because 2 dims for observation in the two bands, 2 dims for observation_mask in the two bands, and the last dimension for time values.
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)
    # max_seq_len does not denote the maximum time duration, i.e., if max_seq_len = 60, it doesn't mean 60 hours/minutes, for example.
    print(f'Max. sequence length (or the max no. of datapoints of lightcurves) in the dataset: {max_seq_len}')

    final_data, final_objIds = [], []

    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId)
        obs_time = lc_data[:, -1]
        obs_mask = lc_data[:, dim:2*dim]
        obs_data = lc_data[:, :dim]
        # Make the first time to zero.
        obs_time = obs_time - obs_time[0]
        obs_time = obs_time * 24  # to convert times into hours.
        observed_tp_final = np.expand_dims(
            np.pad(obs_time, (0, max_seq_len-len(lc_data)), 'constant'),  # pad the time array with zero elements before the first time, and `max_seq_len-len(lc_data)` zeros after the last datapoint.
            1
        )
        observed_mask_final = pad_rows_to_match_columns(obs_mask.T, max_seq_len).T
        observed_data_final = pad_rows_to_match_columns(obs_data.T, max_seq_len).T
        data = np.concatenate((observed_data_final, observed_mask_final, observed_tp_final), axis=1)

        final_data.append(data)
        final_objIds.append(objId)

    # Now add separately mined transients (TDEs).
    # TODO: Remove this and instead manually seelct regions of each light curve and add them separately.
    for objId in pdf_filtered_filtered['i:objectId'].unique():
        lc_data = get_lc(
            pdf_filtered_filtered, objId,
            fid_column='i:fid', magpsf_column='i:magpsf', jd_column='i:jd',
            sigmapsf_column='i:sigmapsf', finkclass_column=None, objectId_column='i:objectId'
        )  # of shape (n, 5), n is the total no. of alerts (including all bands) for that objectId
        obs_time = lc_data[:, -1]
        obs_mask = lc_data[:, dim:2*dim]
        obs_data = lc_data[:, :dim]
        # Make the first time to zero.
        obs_time = obs_time - obs_time[0]
        obs_time = obs_time * 24  # to convert times into hours.
        observed_tp_final = np.expand_dims(
            np.pad(obs_time, (0, max_seq_len-len(lc_data)), 'constant'),  # pad the time array with zero elements before the first time, and `max_seq_len-len(lc_data)` zeros after the last datapoint.
            1
        )
        observed_mask_final = pad_rows_to_match_columns(obs_mask.T, max_seq_len).T
        observed_data_final = pad_rows_to_match_columns(obs_data.T, max_seq_len).T
        data = np.concatenate((observed_data_final, observed_mask_final, observed_tp_final), axis=1)

        final_data.append(data)
        final_objIds.append(objId)

    final_data = np.array(final_data)
    print(final_data.shape, len(final_objIds))
 
    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId --> can do for classification, not required for unsupervised learning.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output since I am using train_test_independently for splitting the data and the objIds.
    # TODO: ensure multiple runs of label encoding on the same number gives the same encoded value.
    le = preprocessing.LabelEncoder()
    final_objIds_encoded = le.fit_transform(final_objIds)  # use le.inverse_transform to get the string from the encoded value.

    train_data, test_data, train_data_objId, test_data_objId = train_test_split(final_data, final_objIds_encoded, train_size=train_size, random_state=42, shuffle=True)
    train_data, val_data, train_data_objId, val_data_objId = train_test_split(train_data, train_data_objId, train_size=train_size, random_state=42, shuffle=True)

    train_data_objId = torch.as_tensor(train_data_objId)
    val_data_objId = torch.as_tensor(val_data_objId)
    test_data_objId = torch.as_tensor(test_data_objId)
    train_data = torch.as_tensor(train_data)
    val_data = torch.as_tensor(val_data)
    test_data = torch.as_tensor(test_data)

    print(f'train_data.shape, val_data.shape, test_data.shape, train_data_objId.shape, val_data_objId.shape, test_data_objId.shape: {train_data.shape, val_data.shape, test_data.shape, train_data_objId.shape, val_data_objId.shape, test_data_objId.shape}')

    train_dataset = TensorDataset(train_data, train_data_objId)
    val_dataset = TensorDataset(val_data, val_data_objId)
    test_dataset = TensorDataset(test_data, test_data_objId)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    data_obj = {
        "final_data": final_data,
        "final_objIds": final_objIds,
        "final_objIds_encoded": final_objIds_encoded,
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim": dim
    }
    return data_obj


def prepare_data_graph_test(df_alerts, dim=2, train_size=0.7, train_batch_size=32, classify=False, activity=False, convert_to_tensor=False, custom_train_min_time=None, custom_train_max_time=None):
    """
    Pass in `custom_train_min_time` and `custom_train_max_time` if applying on totally new set of polled alerts. In that case, set this to the values corresponding to training.
    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'  # We don't require GPU fr preparing the data but only for training.
    time_in_hrs = True

    # Since we aim to do global time normalization, we first need to find the max and min times (the absolute values and not the no. of datapoints) across the entire dataset. **NOTE: This loop should NOT be used if applying the model on a totally new data instance since we must use the min_time/max_time calculated during training and not calculate it again.**
    # FIRST, we do the loop only to find the min/max times across the dataset. Then the second loop performs the global time normalization using the max time found in the first loop.
    # min_max_mags to store the min/max mag so that these values can be used to unnormalize the light curves later. This is because mag normalization is locally done for each lc. We don't need to save the min/max values for the time (x-axis) since time is normalized globally, so a singel value across the dataset suffices.
    min_time, max_time, duration_lcs, min_max_magdiffs, min_max_mags = np.inf, -np.inf, [], [], []
    for objId in df_alerts['objectId'].unique():
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor, normalize_times=False, max_time=None, min_time=None, time_in_hrs=time_in_hrs)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
        assert lc_data[0] == objId
        #assert lc_data[1][0] == 0.0  # Because lc_data[1] is the time array and it must start with zero because we use make_first_time_zero=True above.
        if lc_data[1][0].numpy()[0] < min_time:
            min_time = lc_data[1][0].numpy()[0]
        if lc_data[1][-1].numpy()[0] > max_time:
            max_time = lc_data[1][-1].numpy()[0]
        duration_lcs.append(float(lc_data[1][-1] - lc_data[1][0]))
        _lc_data_obs = lc_data[2]
        min_max_magdiffs.append(_lc_data_obs.max() - _lc_data_obs[_lc_data_obs != 0.0].min())  # Ignoring zero values for min because zero values mean unobserved.
        min_max_mags.append((objId, _lc_data_obs[_lc_data_obs != 0.0].min().item(), _lc_data_obs.max().item()))

    #assert min_time = 0.0   # Because first time is always zero for all lcs because we use make_first_time_zero=True.
    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the dataset: {max_time}, {min_time}')

    total_data, total_objId, total_common_finkclasses = [], [], []
    for objId in df_alerts['objectId'].unique():
        # NOTE: It's important to note that here we use normalize_times=False since we want min_time and max_time calculated on the specific dataset (train, val, OR test), just as done in mTAN phyionet data preprocessing. This normalization is done in variable_time_collate_fn.
        lc_data = get_lc(df_alerts, objId, make_first_time_zero=True, convert_to_tensor=convert_to_tensor, normalize_times=False, local_time_normalization=False, max_time=max_time, min_time=min_time, time_in_hrs=time_in_hrs)  # returns a tuple (object_Id, tt, vals, mask, labels). objectId will be a string, no. of entries/rows in tt, vals, and mask will be `n` = the total no. of alerts (including all bands) for that objectId
        total_data.append(lc_data)
        assert lc_data[0] == objId
        total_objId.append(objId)
        total_common_finkclasses.append(lc_data[-1])

    ################################################
    # Add cases manually. Currently we only add TDEs
    # EDIT: We don't need this since I am adding these cases in the processed alerts dataframe directly. Otherwise it becomes difficult to access these alerts for post-testing analysis
    #tns_tde_total_data, tns_tde_total_objId, tns_tde_total_common_finkclasses = add_tns_tde()
    #total_data.extend(tns_tde_total_data)
    #total_objId.extend(tns_tde_total_objId)
    #total_common_finkclasses.extend(tns_tde_total_common_finkclasses)
    ################################################

    #data_min, data_max = get_data_min_max(total_data)
    #print(f'data_min, data_max: {data_min, data_max}')
    #data_min, data_max = data_min.to(device), data_max.to(device)
    data_min, data_max = None, None  # Since we don't use min/max calculated across the entire train/val/test dataset.

    # TODO: should we use stratified split? stratifying based on the most common finkclass across all alerts of a given objId --> can do for classification, not required for unsupervised learning.
    # TODO: Ensure that using random_state=42 and shuffle=True gives the same output since I am using train_test_independently for splitting the data and the objIds.
    # ensure multiple runs of label encoding on the same number gives the same encoded value. UPDATE: Not applicable now since we are not encoding the labels anymore.
    # We are encoding the objectIds just for efficiency because string types may not be efficient with PyTorch.
    # NOTE: I commented the below two lines since I am thinking the labels need NOT be encoded, and can keep it as strings only.
    #le = preprocessing.LabelEncoder()
    #total_objId_encoded = le.fit_transform(total_objId)  # use le.inverse_transform to get the string from the encoded value.

    #train_data, test_data, train_data_objId, test_data_objId = train_test_split(total_data, total_objId, train_size=train_size, random_state=42, shuffle=True)
    #train_data, val_data, train_data_objId, val_data_objId = train_test_split(train_data, train_data_objId, train_size=0.8, random_state=42, shuffle=True)

    #print('DEBUG: train_data and test_data last time printing for a few cases.')
    #for i, td in enumerate(train_data):
    #    if i == 5:
    #        break
    #    print(td[1][-1])

    #for i, td in enumerate(test_data):
    #    if i == 5:
    #        break
    #    print(td[1][-1])

    if custom_train_min_time is None or custom_train_max_time is None:
        # Find the min and max times for train, val, and test sets.
        train_min_time = np.min([td[1].min() for td in train_data])
        train_max_time = np.max([td[1].max() for td in train_data])
        val_min_time = np.min([td[1].min() for td in val_data])
        val_max_time = np.max([td[1].max() for td in val_data])
        test_min_time = np.min([td[1].min() for td in test_data])
        test_max_time = np.max([td[1].max() for td in test_data])
    else:
        train_min_time, train_max_time, val_min_time, val_max_time, test_min_time, test_max_time = custom_train_min_time, custom_train_max_time, custom_train_min_time, custom_train_max_time, custom_train_min_time, custom_train_max_time

    print(f'Max and Min time values (in {"days" if not time_in_hrs else "hrs"}) across the train dataset: {train_max_time}, {train_min_time}')

    data_combined, data_Ids = variable_time_collate_fn(total_data, device, classify=classify, activity=activity, data_min=data_min, data_max=data_max, train_min_time=train_min_time, train_max_time=train_max_time)

    assert np.all(total_objId == data_Ids)

    # Q) Instead of inserting zero in the observed values array where no observed value exists, is it better to put a sufficient low mag instead, like 25?
    # Answer: I have confirmed that training, validation, and testing does NOT get affected by keeping unobserved values as 0 or 23 because these are essentially masked anyways.

    print(data_combined.shape, len(data_Ids))

    print(f'data_combined.shape: {data_combined.shape}')#, val_data_combined.shape, test_data_combined.shape: {train_data_combined.shape, val_data_combined.shape, test_data_combined.shape}')

    ##### A quick check #####
    seq_len_all = []  # stores the sequence length of all light curves.
    for objId in df_alerts['objectId'].unique():
        pdf = df_alerts[df_alerts['objectId'] == objId]
        seq_len_all.append(len(pdf))

    max_seq_len = max(seq_len_all)  # max_seq_len does not denote the maximum time duration, i.e., if max_seq_len = 60, it doesn't mean 60 hours/minutes.
    print(f'Max. sequence length (or the max no. of datapoints of lightcurves) in the dataset (max_seq_len): {max_seq_len}')
    assert max_seq_len == data_combined.shape[1]
    #########################

    # Create a tensor dataset just for the sake of storing the objectIds corresponding to each light curve, which may be helpful downstream. Especially for analysis while testing.
    #print(train_data_objId)
    #print(train_data_combined.shape)
    #train_data_combined = Dataset(train_data_combined, train_data_objId)
    #val_data_combined = Dataset(val_data_combined, val_data_objId)
    #test_data_combined = Dataset(test_data_combined, test_data_objId)

    # Below four lines added latest by me.
    #transform = SemiRandomTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.5), p=0.5, dim=dim, min_datapoints_each_filter=10)
    #transform = SemiRandomContinuousTruncateLightCurve(n_to_sample_range=(6, 20), p=0.5, dim=2, min_datapoints_each_filter=10)
    #transform = apply_truncate_transforms_random(
    #    ContinuousTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
    #    TruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
    #    p=0.5
    #)

    #transform = apply_truncate_transform_random(
    #    ContinuousTruncateLightCurve(percentage_tp_to_sample_range=(0.3, 0.7), dim=dim, min_datapoints_each_filter=10),
    #    p=0.5
    #)
    test_dataset = MyDataSet(data_combined, data_Ids, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    # Since total_common_finkclass may be heterogenous, we pad them just for convenience before saving as an numpy array.
    # Credit for below code: https://stackoverflow.com/a/43146373
    # TODO: Padding can be avoided if you use np.savez. Can keep total_common_finkclasses as a list here and in the main_processing.py, can use np.savez instead. LOW_PRIORITY
    pad = len(max(total_common_finkclasses, key=len))
    total_common_finkclasses = np.array([i + [0]*(pad-len(i)) for i in total_common_finkclasses])

    #train_data_objId_raw = le.inverse_transform(train_data_objId)
    #val_data_objId_raw = le.inverse_transform(val_data_objId)
    #test_data_objId_raw = le.inverse_transform(test_data_objId)

    data_obj = {
        #"final_data": np.array(total_data),  # This may give error since total_data is a list containing variable length entries.
        "duration_lcs": np.array(duration_lcs),
        "seq_len_all": np.array(seq_len_all),
        "min_max_mags": np.array(min_max_mags),
        "min_max_magdiffs": np.array(min_max_magdiffs),
        "total_objIds": np.array(total_objId),
        "test_data_combined": data_combined,
        #"total_objIds_encoded": np.array(total_objId_encoded),
        "total_common_finkclasses": total_common_finkclasses,
        "test_dataloader": test_loader,
        "input_dim": dim,
        "train_val_test_min_max_times": np.array([train_min_time, train_max_time, val_min_time, val_max_time, test_min_time, test_max_time]),
        "train_min_max_times": np.array([train_min_time, train_max_time])
    }

    return data_obj

