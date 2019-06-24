#!/bin/bash

## Configure job
#SBATCH -c40
#SBATCH -w compute-1-12
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:4
#SBATCH -p RUSS_RESERVED
#SBATCH --mem=MaxMemPerNode
hostname
module load singularity

# Kill all GPU processes before training, since there might be some "ghost" still running
kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')

CUDA_VISIBLE_DEVICES=0 singularity exec --nv c.img python3.6 -u music/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 200 --nlevels 12 --batch_size 32 --embed_dim 384 &
CUDA_VISIBLE_DEVICES=1 singularity exec --nv c.img python3.6 -u music/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 500 --nlevels 12 --batch_size 32 --embed_dim  384 &
CUDA_VISIBLE_DEVICES=2 singularity exec --nv c.img python3.6 -u music/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 1000 --nlevels 12 --batch_size 32 --embed_dim 384 & 
CUDA_VISIBLE_DEVICES=3 singularity exec --nv c.img python3.6 -u music/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 1500 --nlevels 12 --batch_size 32 --embed_dim 384

