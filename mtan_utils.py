# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn import model_selection
from sklearn import metrics

import utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask, weight=None):
    """Masked MSE. With `weight` (same shape as mask), a weighted MSE: sum(w*mask*err^2) / sum(w*mask)."""
    error = (orig - pred) ** 2
    w = mask if weight is None else mask * weight
    error = error * w
    return error.sum() / w.sum()


def has_weight_channel(batch, dim):
    """True if `batch` is [data | mask | weight | time] (3*dim + 1 channels) rather than [data | mask | time]."""
    return batch.shape[-1] == 3 * dim + 1


def get_point_weight(batch, dim, args):
    """The (B, T, dim) weight block of `batch` if args.use_weights is set and the data has one, else None."""
    if getattr(args, 'use_weights', False) and has_weight_channel(batch, dim):
        return batch[:, :, 2 * dim:3 * dim]
    return None

def residual(orig, pred, mask):
    error = orig - pred
    error = error * mask
    return error

def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        #data_norm = (data - att_min) / (att_max - att_min)
        # Convert mag to flux with zero point of 27.5 (leading to FLUXCAL units)
        #flux = 10 ** (-0.4 * (data - 27.5))
        #max_flux = 10 ** (-0.4 * (att_min - 27.5))  # att_min instead of att_max because input data is magnitudes.
        data_norm = (data - att_min) / 2.5
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    # NOTE: I have confirmed that if I replace all unobserved values to 23 instead of 0, then the training, validation, and testing, nothing is affected.
    data_norm[mask == 0] = 0

    #print(att_min, att_max, max_flux, data_norm.min(), data_norm.max(), data_norm.mean(), 'check')

    return data_norm, att_min, att_max


def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cuda", kl_coef=None, return_mse=True, train_val_test_min_max_times_filename='train_val_test_min_max_times.npy',
             weighted=None, return_both=False):
    """
    If return_mse is False, the average ELBO will be returned. In this case, both kl_coef and k_iwae will be used; the latter is found from `args` and kl_coef must be given.
    If return_mse is True, mse is returned.

    The MSE is the mean over curves of the per-curve MSE. `weighted` selects whether the points are weighted inside
    each curve (default: args.use_weights, and only if the data has a weight channel). With return_both=True and
    return_mse=True, the tuple (mse, mse_unweighted) is returned, where mse follows `weighted`. The ELBO always follows
    args.use_weights.
    """
    if not return_mse and kl_coef is None:
        raise ValueError('kl_coef must be provided is return_mse is False.')

    # NOTE: We use max_time calculated on train set for val and test also
    train_val_test_min_max_times = np.load(train_val_test_min_max_times_filename)
    train_max_time = train_val_test_min_max_times[1]
    delta_t = (args.ref_resolution_days * 24) / train_max_time

    mse, mse_unw, test_n = 0.0, 0.0, 0.0
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            test_batch = batch[0]
            test_batch = test_batch.to(device)
            batch_len = test_batch.shape[0]
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask

            if args.sample_tp == 1.:
                assert torch.all(observed_tp == subsampled_tp)

            #assert subsampled_tp.max() <= 1
            ##query = torch.linspace(0, subsampled_tp.max(), args.num_ref_points)

            #subsampled_tp_for_reference = utils.trim_padded_zeros_tensor_2d(subsampled_tp)
            # We simply flatten the time value list for the different bands. That's okay because we only need the min and max.
            #subsampled_tp_for_reference = torch.cat(subsampled_tp_for_reference)
            #assert subsampled_tp_for_reference[0] == subsampled_tp_for_reference.min()
            #assert subsampled_tp_for_reference[0] == 0.0

            #query = torch.arange(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max() + delta_t, delta_t)
            #query = query[query <= 1.0]

            #query = torch.linspace(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max(), steps=args.num_ref_points).to(device)

            #query = torch.arange(subsampled_tp.min(), subsampled_tp.max() + delta_t, delta_t)
            #query = query[query <= 1.0]
            query = utils.generate_query_matrix(subsampled_tp, delta_t, padding_value=-999)

            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp, query)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps, query)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            has_w = has_weight_channel(test_batch, dim)
            use_w = (getattr(args, 'use_weights', False) if weighted is None else weighted) and has_w
            mse_u = mean_squared_error(observed_data, pred_x, observed_mask)
            mse_w = mean_squared_error(observed_data, pred_x, observed_mask,
                                       weight=test_batch[:, :, 2 * dim:3 * dim]) if use_w else mse_u
            mse += mse_w * batch
            mse_unw += mse_u * batch
            test_n += batch

            # compute loss
            if not return_mse:
                logpx, analytic_kl = compute_losses(
                    dim, test_batch, qz0_mean, qz0_logvar, pred_x, args, device)
                loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
                test_loss += loss.item() * batch_len

    if return_mse:
        if return_both:
            return mse / test_n, mse_unw / test_n
        return mse / test_n
    else:
        return test_loss / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    # Optional per-point weights: the data-fit term of a point is multiplied by its weight (equivalent to a
    # Gaussian of variance std^2 / weight). Weights are 0 where the mask is 0, so padding stays inert.
    point_weight = get_point_weight(dec_train_batch, dim, args)
    loss_mask = observed_mask if point_weight is None else observed_mask * point_weight
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           loss_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auc = metrics.roc_auc_score(
        true, pred[:, 1]) if not args.classify_pertp else 0.
    return test_loss/pred.shape[0], acc, auc


def get_mimiciii_data(args):
    input_dim = 12
    x = np.load('../../../neuraltimeseries/Dataset/final_input3.npy')
    y = np.load('../../../neuraltimeseries/Dataset/final_output3.npy')
    x = x[:, :25]
    x = np.transpose(x, (0, 2, 1))

    # normalize values and time
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                  :input_dim], x[:, :, input_dim:2*input_dim], x[:, :, -1]
    if np.max(observed_tp) != 0.:
        observed_tp = observed_tp / np.max(observed_tp)

    if not args.nonormalize:
        for k in range(input_dim):
            data_min, data_max = float('inf'), 0.
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            #print(data_min, data_max)
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = (
                observed_vals[:, :, k] - data_min)/data_max
    # set masked out elements back to zero
    observed_vals[observed_mask == 0] = 0
    print(observed_vals[0], observed_tp[0])
    print(x.shape, y.shape)
    kfold = model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0)
    splits = [(train_inds, test_inds)
              for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]
    x_train, y_train = x[splits[args.split][0]], y[splits[args.split][0]]
    test_data_x, test_data_y = x[splits[args.split]
                                 [1]], y[splits[args.split][1]]
    if not args.old_split:
        train_data_x, val_data_x, train_data_y, val_data_y = \
            model_selection.train_test_split(
                x_train, y_train, stratify=y_train, test_size=0.2, random_state=0)
    else:
        frac = int(0.8*x_train.shape[0])
        train_data_x, val_data_x = x_train[:frac], x_train[frac:]
        train_data_y, val_data_y = y_train[:frac], y_train[frac:]

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))
    train_data_combined = TensorDataset(torch.from_numpy(train_data_x).float(),
                                        torch.from_numpy(train_data_y).long().squeeze())
    val_data_combined = TensorDataset(torch.from_numpy(val_data_x).float(),
                                      torch.from_numpy(val_data_y).long().squeeze())
    test_data_combined = TensorDataset(torch.from_numpy(test_data_x).float(),
                                       torch.from_numpy(test_data_y).long().squeeze())
    train_dataloader = DataLoader(
        train_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=args.batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}
    return data_objects


def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None, train_min_time=None, train_max_time=None,
                             weights=None):
    """
    If `weights` is given (a list parallel to `batch` of (T, D) tensors, 0 where the mask is 0), the returned
    tensor is [data | mask | weight | time] (3D + 1 channels) instead of [data | mask | time] (2D + 1).
    The time stays in the last channel, and data / mask keep their indices.

    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if weights is not None:
        assert len(weights) == len(batch)
        enc_combined_weight = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)


    # NOTE: Added by me
    #combined_record_ids = torch.zeros([len(batch), 1], dtype=object)
    combined_record_ids = []

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        #print('inside variable_time_collate_fn')
        #print(tt.shape, vals.shape, mask.shape, tt.size(0), b, maxlen)

        # Below two lines are added now.
        data_min, data_max = get_data_min_max_single_record((record_id, tt, vals, mask, labels))
        if b == 5 or b == 10 or b == 1:  # print for random cases
            print(f'data_min, data_max: {data_min}, {data_max}')
        vals, _, _ = normalize_masked_data(vals, mask, att_min=data_min, att_max=data_max)

        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.squeeze().to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)

        enc_combined_mask[b, :currlen] = mask.to(device)
        if weights is not None:
            assert weights[b].shape == mask.shape, (record_id, weights[b].shape, mask.shape)
            enc_combined_weight[b, :currlen] = weights[b].to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)
        
        # NOTE: Added by me
        # combined_record_ids[b] = record_id
        combined_record_ids.append(record_id)

    """  # This was the code used in mTAN. We use min/max statistics of the data itself for normalizing. So that's why it's done inside the for loop.
    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)
    """

    if torch.max(enc_combined_tt) != 0.:
        # The below assertion is valid only if normalize_times=True in get_lc inside prepare_data since the time values will already be normalized to [0, 1].
        #assert torch.max(enc_combined_tt) == 1.0
        # The below assertion is valid only if make_first_time_zero=True in get_lc inside prepare_data since only then the first time value will be zero.
        assert torch.min(enc_combined_tt) == 0.0
        if train_min_time is None or train_max_time is None:
            enc_combined_tt = (enc_combined_tt - torch.min(enc_combined_tt)) / (torch.max(enc_combined_tt) - torch.min(enc_combined_tt))
        else:
            assert train_min_time == 0.0
            enc_combined_tt = (enc_combined_tt - train_min_time) / (train_max_time - train_min_time)

        #assert torch.all((enc_combined_tt >= 0) & (enc_combined_tt <= 1))
        #enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    if weights is not None:
        combined_data = torch.cat(
            (enc_combined_vals, enc_combined_mask, enc_combined_weight, enc_combined_tt.unsqueeze(-1)), 2)
    else:
        combined_data = torch.cat(
            (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data, combined_record_ids

def irregularly_sampled_data_gen(n=10, length=20, seed=0):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    obs_values, ground_truth, obs_times = [], [], []
    for i in range(n):
        t1 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t2 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t3 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        a = 10 * np.random.randn()
        b = 10 * np.random.rand()
        f1 = .8 * np.sin(20*(t1+a) + np.sin(20*(t1+a))) + \
            0.01 * np.random.randn()
        f2 = -.5 * np.sin(20*(t2+a + 20) + np.sin(20*(t2+a + 20))
                          ) + 0.01 * np.random.randn()
        f3 = np.sin(12*(t3+b)) + 0.01 * np.random.randn()
        obs_times.append(np.stack((t1, t2, t3), axis=0))
        obs_values.append(np.stack((f1, f2, f3), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        t = np.linspace(0, 1, 100)
        fg1 = .8 * np.sin(20*(t+a) + np.sin(20*(t+a)))
        fg2 = -.5 * np.sin(20*(t+a + 20) + np.sin(20*(t+a + 20)))
        fg3 = np.sin(12*(t+b))
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2, fg3), axis=0))
    return obs_values, ground_truth, obs_times


def sine_wave_data_gen(args, seed=0):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        t = np.sort(np.random.choice(np.linspace(
            0, 1., 101), size=args.length, replace=True))
        b = 10 * np.random.rand()
        f = np.sin(12*(t+b)) + 0.1 * np.random.randn()
        obs_times.append(t)
        obs_values.append(f)
        tc = np.linspace(0, 1, 100)
        fg = np.sin(12*(tc + b))
        ground_truth.append(fg)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def kernel_smoother_data_gen(args, alpha=100., seed=0, ref_points=10):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)

        query_points = np.sort(np.random.choice(
            np.linspace(0, 1., 101), size=args.length, replace=True))
        # query_points = np.sort(np.random.uniform(low=0.0, high=1.0, size=args.length))
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        obs_values.append(query_values)
        obs_times.append(query_points)

        query_points = np.linspace(0, 1, 100)
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def get_toy_data(args):
    dim = 3
    obs_values, ground_truth, obs_times = irregularly_sampled_data_gen(
        args.n, args.length)
    obs_times = np.array(obs_times).reshape(args.n, -1)
    obs_values = np.array(obs_values)
    combined_obs_values = np.zeros((args.n, dim, obs_times.shape[-1]))
    mask = np.zeros((args.n, dim, obs_times.shape[-1]))
    for i in range(dim):
        combined_obs_values[:, i, i *
                            args.length: (i+1)*args.length] = obs_values[:, i]
        mask[:, i, i*args.length: (i+1)*args.length] = 1.
    #print(combined_obs_values.shape, mask.shape, obs_times.shape, np.expand_dims(obs_times, axis=1).shape)
    combined_data = np.concatenate(
        (combined_obs_values, mask, np.expand_dims(obs_times, axis=1)), axis=1)
    combined_data = np.transpose(combined_data, (0, 2, 1))
    print(combined_data.shape)
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def compute_pertp_loss(label_predictions, true_label, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    n_traj, n_tp, n_dims = label_predictions.size()
    label_predictions = label_predictions.reshape(n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj * n_tp, n_dims)
    mask = torch.sum(mask, -1) > 0
    mask = mask.reshape(n_traj * n_tp,  1)
    _, true_label = true_label.max(-1)
    ce_loss = criterion(label_predictions, true_label.long())
    ce_loss = ce_loss * mask
    return torch.sum(ce_loss)/mask.sum()


def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask


def subsample_timepoints_continuous_window(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        start = np.random.randint(0, n_tp_current - n_to_sample + 1)
        subsampled_idx = non_missing_tp[start:start + n_to_sample]
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask

import os
import utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader


def get_data_min_max(records):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:,i][mask[:,i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max

def get_data_min_max_single_record(record):
    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0]

    (record_id, tt, vals, mask, labels) = record

    n_features = vals.size(-1)

    batch_min = []
    batch_max = []
    for i in range(n_features):
        non_missing_vals = vals[:,i][mask[:,i] == 1]
        if len(non_missing_vals) == 0:
            batch_min.append(inf)
            batch_max.append(-inf)
        else:
            batch_min.append(torch.min(non_missing_vals))
            batch_max.append(torch.max(non_missing_vals))

    batch_min = torch.stack(batch_min)
    batch_max = torch.stack(batch_max)

    if (data_min is None) and (data_max is None):
        data_min = batch_min
        data_max = batch_max
    else:
        data_min = torch.min(data_min, batch_min)
        data_max = torch.max(data_max, batch_max)

    assert data_min.numel() == n_features
    assert data_max.numel() == n_features
    # NOTE: We don't want filter/channel-wise min/max values since that will lose color information when normalizing.
    # We want the min/max across all filters. So we do the below operation.
    data_min = torch.min(data_min)
    data_max = torch.max(data_max)
    assert data_min.numel() == 1
    assert data_max.numel() == 1

    return data_min, data_max


