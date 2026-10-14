#!/bin/bash
#SBATCH -A ixh@v100
#SBATCH --job-name=mTan_train
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-dev            # dev: 2h max
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=mTan_train_%j.out
#SBATCH --error=mTan_train_%j.out

# Usage: sbatch train.slurm train_val_test_min_max_times_<TAG>.npy [extra args for tan_unsupervised_lsst.py]
#
# Weighting of the likelihood: nothing to set here. tan_unsupervised_lsst.py is weighted iff the data has a weight
# channel (preprocessing with --weight n|sqrt_n|invvar, TAG suffixed _w<mode>); the default preprocessing
# (--weight none) has none, so the run is unweighted. Add --no-use-weights to ignore an existing channel.
# The checkpoint is named lsst_<TAG>_mtan_rnn_mtan_rnn_weights|_noweights[<--run-suffix>].h5.
# Options given after the .npy file override the ones below (argparse keeps the last value),
# e.g.  --niters 1000 --seed 1 --run-suffix _s1
MINMAX=$1; shift
TAG=$(basename "$MINMAX" .npy); TAG=${TAG#train_val_test_min_max_times_}

module purge
module load pytorch-gpu/py3/2.6.0
set -x

# Pas d'internet sur les noeuds de calcul : ne pas utiliser --use_wandb
srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u ./tan_unsupervised_lsst.py \
    --dim 4 --latent-dim 4 --rec-hidden 64 --gen-hidden 50 --embed-time 128 \
    --enc mtan_rnn --dec mtan_rnn --enc-num-heads 1 --dec-num-heads 1 \
    --lr 3e-4 --std 0.01 --k-iwae 5 --kl --learn-emb --ref-resolution-days 2 \
    --niters 2000 --dataset lsst_${TAG} \
    --train_val_test_min_max_times_filename "$MINMAX" "$@"
