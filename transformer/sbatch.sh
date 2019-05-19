#!/bin/bash

## Configure job
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclude=compute-0-26
#SBATCH -p RUSS_RESERVED,MATRIX_CLUSTER
#SBATCH --mem=16g
srun hostname
module load singularity

singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 0& 
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 1&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 2&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 3&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 0&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 1&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 2&
singularity exec --nv c.img python3.6 -u muqiaoy/dl_signal/transformer/train_low_mem.py --data music --path /home/qianlim/low_mem --time_step 128 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --nhorizons 1 --modal_lengths 2048 2048 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.0001 --clip 0.35 --optim Adam --hidden_size 2000 --nlevels 1 --batch_size 2 --embed_dim 512 --cuda_device 3
