from models import enc_mtan_rnn, dec_mtan_rnn
from mtan_utils import subsample_timepoints, mean_squared_error
from sklearn import preprocessing
import torch
import numpy as np

from torch.utils.data import DataLoader
from prepare_data_lsst import MyDataSet
from torch.nn.utils.rnn import pad_sequence

import time
import argparse

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True,
                    help='Suffix of the preprocessing files, e.g. ftransfer_lsst_2026_06_29_882968')
parser.add_argument('--setting', type=str, default='test', choices=['test', 'val', 'train'])
parser.add_argument('--model_file', type=str, default=None,
                    help='Default: lsst_<tag>_mtan_rnn_mtan_rnn.h5 (name written by tan_unsupervised_lsst.py --dataset lsst_<tag>)')
parser.add_argument('--latent-dim', type=int, default=4, help='Must match training.')
args = parser.parse_args()
TAG = args.tag

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')  # Force to use CPU for evaluation since that closely mimics how the model will be used in real applications.
# TODO: Add option to pass these arguments as argument parsers. These values must match from training. So instead save a training parameter file and simply load it here.
ref_resolution_days = 2
latent_dim = args.latent_dim
learn_emb = True
rec_hidden = 64
gen_hidden = 50
enc_num_heads = 1
dec_num_heads = 1
sample_tp = 1.0
num_sample = 1
embed_time = 128
dim = 4  # g,r,i,z (Rubin/LSST). Use 2 for ZTF g,r.
seed = 42
store_decoded_lcs = True
SETTING = args.setting  # 'test', 'val' or 'train'
DATA_COMBINED_PATH = f'{SETTING}_data_combined_{TAG}.pth'
DATA_IDS_PATH = f'{SETTING}_objIds_{TAG}.npy'
model_file_path = args.model_file or f'lsst_{TAG}_mtan_rnn_mtan_rnn.h5'

if __name__ == '__main__':
    # Set seed during testing as well since this script samples random values for the variable, epsilon.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    data_combined = torch.load(DATA_COMBINED_PATH, weights_only=False)
    data_Ids = np.load(DATA_IDS_PATH)
    dataset = MyDataSet(data_combined, data_Ids)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    train_val_test_min_max_times_filename = f'train_val_test_min_max_times_{TAG}.npy'
    train_val_test_min_max_times = np.load(train_val_test_min_max_times_filename)
    train_max_time = train_val_test_min_max_times[1]
    delta_t = (ref_resolution_days * 24) / train_max_time

    #test_loader = torch.load('test_dataloader.pth') # NOTE: This script is only tested for dataloaders with batch size=1; for greater batch sizes, some bugs may be introduced.
    #total_objIds = np.load('total_objIds.npy')
    #total_objIds_encoded = np.load('total_objIds_encoded.npy')

    #le = preprocessing.LabelEncoder()
    #total_objIds_encoded_again  = le.fit_transform(total_objIds)  # use le.inverse_transform to get the string from the encoded value.
    #assert np.all(total_objIds_encoded == total_objIds_encoded_again)  # since the label encoding must be deterministic. So the values here and found during main_preprocessing must match.


    rec = enc_mtan_rnn(
        dim, latent_dim, rec_hidden,  # torch.linspace(0, 1., num_ref_points)
        embed_time=embed_time, learn_emb=learn_emb, num_heads=enc_num_heads, device=device
    ).to(device)

    dec = dec_mtan_rnn(
        dim, latent_dim, gen_hidden,
        embed_time=embed_time, learn_emb=learn_emb, num_heads=dec_num_heads, device=device).to(device)

    model_file = torch.load(model_file_path, weights_only=False, map_location=device)  # Namespace in checkpoint; tensors saved on GPU -> map to CPU
    rec.load_state_dict(model_file['rec_state_dict'])
    rec.eval()
    dec.load_state_dict(model_file['dec_state_dict'])
    dec.eval()

    ########### Check model size ###########
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f'Encoder size [in MB]: {get_model_size(rec)}')
    print(f'Decoder size [in MB]: {get_model_size(dec)}')

    ########################################

    outputs = []
    objIds = []
    test_n, mse = 0, 0.0
    if store_decoded_lcs:
        decoded_lcs = []

    start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            test_batch = batch[0]
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if sample_tp and sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask

            if sample_tp == 1.:
                assert torch.all(observed_tp == subsampled_tp)
            #assert subsampled_tp.max() <= 1
            ##query = torch.linspace(0, subsampled_tp.max(), num_ref_points)

            #subsampled_tp_for_reference = utils.trim_padded_zeros_tensor_2d(subsampled_tp)
            # We simply flatten the time value list for the different bands. That's okay because we only need the min and max.
            #subsampled_tp_for_reference = torch.cat(subsampled_tp_for_reference)
            #assert subsampled_tp_for_reference[0] == subsampled_tp_for_reference.min()
            #assert subsampled_tp_for_reference[0] == 0.0
            
            #query = torch.arange(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max() + delta_t, delta_t)
            #query = query[query <= 1.0]
            #query = torch.linspace(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max(), steps=num_ref_points).to(device)

            #query = torch.arange(subsampled_tp.min(), subsampled_tp.max() + delta_t, delta_t)
            #query = query[query <= 1.0]

            query = utils.generate_query_matrix(subsampled_tp, delta_t, padding_value=-999)

            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp, query)
            qz0_mean, qz0_logvar = (
                out[:, :, : latent_dim],
                out[:, :, latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            
            # NOTE: `outputs` only contain the mean vector since it's expected to be used for visualization of latent representations.
            # Each element in `outputs` is of shape (batch_size x seqlen x dim), where seqlen is different for each element.

            # Only preserve encoded vector values where query time is valid, else set it to -999.
            output_masked = torch.where(
                query.unsqueeze(-1) != -999,
                qz0_mean.view(-1, qz0_mean.shape[1], qz0_mean.shape[2]),
                -999
            )
            outputs.append(output_masked)
            objIds.append(batch[1])

            if store_decoded_lcs:
                batch, seqlen = observed_tp.size()
                time_steps = (
                    observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
                )
                pred_x = dec(z0, time_steps, query)
                pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
                pred_x = pred_x.mean(0)
                mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
                test_n += batch

                #print(pred_x.shape, time_steps.shape, time_steps.unsqueeze(2).shape)
                print(pred_x.shape, observed_data.shape, observed_mask.shape)
                decoded_lcs.append(
                    np.vstack(
                        (pred_x.cpu().detach().numpy(), observed_data.cpu().detach().numpy(), observed_mask.cpu().detach().numpy(), np.repeat(time_steps.unsqueeze(2).cpu().detach().numpy(), dim, axis=2))
                    )
                )

    end = time.time()

    print(f'Time elapsed = {end-start:.2f}s for processing {len(dataset)} light curves.')

    #outputs_condensed = np.array([o.cpu().detach().numpy() for o in outputs])  # this will be an array of shape (num_test_examples, num_sample, num_ref_points, latent_dim). The num_sample dimension can be averaged or compressed somehow if it contains more than one entry.
    padding_value = -99
    print(outputs[0].shape, outputs[1].shape)
    outputs_condensed = pad_sequence([o.cpu().detach().permute((1,0,2)) for o in outputs], padding_value=padding_value, batch_first=True)
    outputs_condensed[outputs_condensed == padding_value] = np.nan
    print(outputs_condensed.shape)
    outputs_condensed = outputs_condensed.permute(0,2,1,3)
    print(outputs_condensed.shape)

    from itertools import chain
    objIds = list(chain.from_iterable(objIds))

    if store_decoded_lcs:
        print(f'MSE = {mse/test_n}')

    np.save(f'evaluate_{SETTING}_outputs_condensed_{TAG}.npy', outputs_condensed)
    np.save(f'evaluate_{SETTING}_objIds_dataloader_{TAG}.npy', objIds)

    assert np.all(objIds == data_Ids)

    if store_decoded_lcs:
        np.savez(f'evaluate_{SETTING}_decoded_lcs_{TAG}.npz', *decoded_lcs)
