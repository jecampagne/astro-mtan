# pylint: disable=E1101, E0401, E1102, W0621, W0221
import os
import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models

import time

import mtan_utils
import utils
# Enregistre l'alias sys.modules['prepare_data'] (voir fin de prepare_data_lsst.py) :
# necessaire pour que torch.load relise les DataLoaders pickles, sans l'ancien prepare_data.py.
import prepare_data_lsst  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
#parser.add_argument('--n', type=int, default=8000)
#parser.add_argument('--batch-size', type=int, default=50)
#parser.add_argument('--quantization', type=float, default=0.016,
#                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
#parser.add_argument('--length', type=int, default=20)
#parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
#parser.add_argument('--only-periodic', type=str, default=None)
#parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--topic', type=str, help='Name of the topic of the data transfer that contains the alerts.')
parser.add_argument('--dim', type=int, help='dim value')
parser.add_argument('--use_wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--train_val_test_min_max_times_filename', type=str, default='train_val_test_min_max_times.npy')
parser.add_argument('--ref-resolution-days', type=float, default=2)
parser.add_argument('--tag', type=str, default=None,
                    help="Suffix of the preprocessing files (train_dataloader_<tag>.pth, ...). "
                         "Default: deduced from --train_val_test_min_max_times_filename.")
args = parser.parse_args()
if args.tag is None:
    _base = os.path.basename(args.train_val_test_min_max_times_filename)[:-len('.npy')]
    _pref = 'train_val_test_min_max_times'
    args.tag = _base[len(_pref) + 1:] if _base.startswith(_pref + '_') else ''
_sfx = f'_{args.tag}' if args.tag else ''


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    train_loader = torch.load(f'train_dataloader{_sfx}.pth', weights_only=False)
    val_loader = torch.load(f'val_dataloader{_sfx}.pth', weights_only=False)
    test_loader = torch.load(f'test_dataloader{_sfx}.pth', weights_only=False)
    dim = args.dim

    # model
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, args.embed_time, learn_emb=args.learn_emb, device=device).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, args.latent_dim, args.rec_hidden,   # torch.linspace(0, 1., args.num_ref_points)
            embed_time=args.embed_time, learn_emb=args.learn_emb, num_heads=args.enc_num_heads, device=device).to(device)
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, args.embed_time, learn_emb=args.learn_emb, device=device).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, args.latent_dim, args.gen_hidden,   # torch.linspace(0, 1., args.num_ref_points)
            embed_time=args.embed_time, learn_emb=args.learn_emb, num_heads=args.dec_num_heads, device=device).to(device)

    if args.use_wandb:  # TODO: See if wandb is working as expected and is logging what we want.
        import wandb
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        run = wandb.init(project="fast_transients_mTAN", name=f'run-{experiment_id}', config=args)
        wandb.watch(rec, log_freq=50)
        wandb.watch(dec, log_freq=50)

    train_val_test_min_max_times = np.load(args.train_val_test_min_max_times_filename)
    train_max_time = train_val_test_min_max_times[1]
    delta_t = (args.ref_resolution_days * 24) / train_max_time
    #delta_t = 0.00547746036  # = (2 * 24) / max_time across train set, which is 8763.185277599841

    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))  # betas as in the paper
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    print('parameters:', mtan_utils.count_parameters(rec), mtan_utils.count_parameters(dec))
    if args.fname is not None:
        checkpoint = torch.load(args.fname, weights_only=False)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 1, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 3, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 10, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 20, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 30, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 50, return_mse=True, train_val_test_min_max_times_filename=args.train_val_test_min_max_times_filename, device=device))

    best_val_metric = float('inf')  # NOTE: It is assumed the val metric must be minimized.
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        val_avg_reconst, val_avg_kl, val_loss = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        start_time = time.time()

        for batch in train_loader:
            train_batch = batch[0]  # batch contains the tensor and also the labels due to the recent change in the code.
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            if args.sample_tp and args.sample_tp < 1:
                # NOTE: I think this was designed for synthetic experiments (see Appendix A2 of the mTAN paper), perhaps for creating train-test splits in a given lc to cehck interpolation performance.
                # So this subsampling is not needed in our case since we only deal with observations.
                subsampled_data, subsampled_tp, subsampled_mask = mtan_utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask

            if args.sample_tp == 1.:
                assert torch.all(observed_tp == subsampled_tp)

            # This is okay to do on the train set but not necessarily for val/test sets
            # because we normalize val/test time values using train set statistics,
            # so some normalized time values might be > 1 in val/test.
            assert subsampled_tp.max() <= 1
            ##query = torch.linspace(0, subsampled_tp.max(), args.num_ref_points)

            #subsampled_tp_for_reference = utils.trim_padded_zeros_tensor_2d(subsampled_tp)
            #running_min_t = torch.tensor(float('inf'), device=device)
            #running_max_t = torch.tensor(float('-inf'), device=device)
            #for t in subsampled_tp:
            #    running_min_t = torch.min(running_min_t, t.min())
            #    running_max_t = torch.max(running_max_t, t.max())

            #assert running_min_t == 0.0
            #subsampled_tp_for_reference = torch.cat(subsampled_tp_for_reference)
            #assert subsampled_tp_for_reference[0] == subsampled_tp_for_reference.min()
            #assert subsampled_tp_for_reference[0] == 0.0

            #query = torch.arange(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max() + delta_t, delta_t)

            #print('using query = torch.arange(subsampled_tp.min(), subsampled_tp.max() + delta_t, delta_t)')
            #query = torch.arange(subsampled_tp.min(), subsampled_tp.max() + delta_t, delta_t)
            #query = query[query <= 1.0]

            query = utils.generate_query_matrix(subsampled_tp, delta_t, padding_value=-999)

               # print('using query = torch.linspace(0, 1, 270)')
           # query = torch.linspace(0, 1, 270)

            ####query = [torch.arange(subsampled_tp.min(), t.max() + delta_t, delta_t) for t in subsampled_tp]

            #query = torch.linspace(subsampled_tp_for_reference.min(), subsampled_tp_for_reference.max(), steps=args.num_ref_points).to(device)
            #query = utils.get_reference_times_quantiles(subsampled_tp_for_reference, K=args.num_ref_points).to(device)
            #print('query')
            #print(query)
            #print('subsampled_tp')
            #print(subsampled_tp)
            #print('subsampled_tp_for_reference')
            #print(subsampled_tp_for_reference)
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp, query)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]),
                query
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            # compute loss
            logpx, analytic_kl = mtan_utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += mtan_utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len

            if args.use_wandb:
                # NOTE: Logging is done for each batch. See https://docs.wandb.ai/guides/integrations/pytorch
                wandb.log({'train_loss': train_loss, 'train_avg_reconst': avg_reconst, 'train_avg_kl': avg_kl, 'train_mse': mse})
 
        total_time += time.time() - start_time
        # Run validation
        return_mse = True
        val_metric = mtan_utils.evaluate(dim, rec, dec, val_loader, args, 1, device=device, kl_coef=kl_coef, return_mse=return_mse)
        if args.use_wandb:
            if return_mse:
                wandb.log({'val_mse': val_metric})
            else:
                wandb.log({'val_avg_elbo': val_metric})
        if val_metric <= best_val_metric:
            best_val_metric = min(best_val_metric, val_metric)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            optimizer_state_dict = optimizer.state_dict()

            print(f'Saving the model at epoch {itr}')
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
            }, args.dataset + '_' + args.enc + '_' + args.dec + '.h5')

        # Validation end.
        #scheduler.step(val_metric)
        #print(f'learning rate at iteration {itr} = {scheduler.get_last_lr()}')

        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}, val_metric: {:.6f}'
                .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n, val_metric))
        if itr % 5 == 0:
            print('Test Mean Squared Error', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 1, device=device, return_mse=True))

    print(f'Time elapsed {total_time/60:.2f} min')
    if args.use_wandb:
        wandb.finish()

        #if itr % 10 == 0 and args.save:
        #    torch.save({
        #        'args': args,
        #        'epoch': itr,
        #        'rec_state_dict': rec_state_dict,
        #        'dec_state_dict': dec_state_dict,
        #        'optimizer_state_dict': optimizer_state_dict,
        #        'loss': -loss,
        #    }, args.dataset + '_' + args.enc + '_' + args.dec + '_' +
        #        str(itr) + '.h5')

