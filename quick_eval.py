import models
import torch
dim = 4  # g,r,i,z (Rubin/LSST). Use 2 for ZTF g,r.
device='cuda' if torch.cuda.is_available() else 'cpu'
val_loader = torch.load('val_dataloader.pth')
from mtan_utils import *

rec = models.enc_mtan_rnn(
    dim, torch.linspace(0, 1., 16), 16, 64,
    embed_time=128, learn_emb=True, num_heads=1, device=device).to(device)

dec = models.dec_mtan_rnn(
    dim, torch.linspace(0, 1., 16), 16, 50,
    embed_time=128, learn_emb=True, num_heads=1, device=device).to(device)


data_model = torch.load('')
rec.load_state_dict(data_model['rec_state_dict'])
dec.load_state_dict(data_model['dec_state_dict'])

def evaluate(dim, rec, dec, test_loader, num_sample=10, device="cuda"):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if 0.9 and 0.9 < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), 0.9)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : 16],
                out[:, :, 16:],
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
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n

val_mse = evaluate(dim, rec, dec, val_loader, 1, device=device)
print(val_mse)

