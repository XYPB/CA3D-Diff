#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --comment=CA3D-Diff-Baseline
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=yuexi.du@yale.edu

#SBATCH --account prio_dvornek
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15g
#SBATCH --partition priority_gpu
#SBATCH --nodes 1
#SBATCH --gpus a100:1
#SBATCH --constraint a100-80g
#SBATCH --output=logs/%x_%j.out


set -e

git pull
module load miniconda
conda env list
sleep 10
conda activate CA3D-Diff
conda env list


python train.py -b configs/train_csaw.yaml \
    --finetune_from ckpt/sd-image-conditioned-v2.ckpt \
    -l ckpt/log_csaw  \
    -c ckpt/checkpoint_csaw \
    --gpus 0,


python generate.py \
    --cfg configs/train_csaw.yaml \
    --ckpt ckpt/checkpoint_csaw/train/step=00039999.ckpt \
    --input data/CSAW-CC-paired/CC/test \
    --output runs/cc2mlo_test_csaw \
    --cfg_scale 3.0 \
    --device cuda:0 \
    --batch_size 16 \
    --cc2mlo \

python generate.py \
    --cfg configs/train_csaw.yaml \
    --ckpt ckpt/checkpoint_csaw/train/step=00039999.ckpt \
    --input data/CSAW-CC-paired/MLO/test \
    --output runs/mlo2cc_test_csaw \
    --cfg_scale 3.0 \
    --device cuda:0 \
    --batch_size 16 \
