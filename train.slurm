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
MINMAX=$1; shift
TAG=$(basename "$MINMAX" .npy); TAG=${TAG#train_val_test_min_max_times_}

module purge
module load pytorch-gpu/py3/2.6.0
set -x

# Pas d'internet sur les noeuds de calcul : ne pas utiliser --use_wandb
srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u ./tan_unsupervised_lsst.py \
    --dim 4 --latent-dim 4 --rec-hidden 64 --gen-hidden 50 --embed-time 128 \
    --enc mtan_rnn --dec mtan_rnn --enc-num-heads 1 --dec-num-heads 1 \
    --lr 1e-4 --k-iwae 5 --kl --learn-emb --ref-resolution-days 2 \
    --niters 300 --dataset lsst_${TAG} \
    --train_val_test_min_max_times_filename "$MINMAX" "$@"
