#!/bin/sh
# To test different models on matrix cluster
# usage: sbatch --gres=gpu:4 --mem=64g -t 8:00:00 -p MATRIX_CLUSTER dl_signal/run_fnn.sh complex &
# See the updated running session: watch -n 1 cat slurm-JOBNUM.out

singularity shell --nv c.img -c "
cd music/dl_signal/
if [[ $1 == 'real' ]]; then
    echo 'running real-valued fully connected...'
    python3.6 fnn_iq.py \
	--path ../dataset/iq \
	--batch_size 256 \
	--hidden_size 200 \
	--num_layers 1 \
	--dropout 0.1 \
	--learning_rate 0.1 \
	--momentum 0 \
	--weight_decay 0

elif [[ $1 == 'complex' ]]; then
    echo 'running complex-valued fully connected...'
    python3.6 fnn_crelu.py \
        --path ../dataset/iq \
        --batch_size 100 \
        --hidden_size 200 \
        --num_layers 2 \
        --dropout 0.1 \
        --learning_rate 0.1 \
        --momentum 0 \
        --weight_decay 0

elif [[ $1 == 'transformer' ]]; then
    echo 'running transformer...'
    python3.6 -u transformer/train.py \
        --dataset ../data/ \
        --lr 0.001 \
        --time_step 1024 \
	--hidden_size 2000 \
        --num_epochs 2000 \
	--attn_dropout 0.1 \
        --relu_dropout 0.1 \
        --res_dropout 0.1 \
        --batch_size 1 \
        --nlevels 6 \
        --nhorizons 1
else
    echo 'unknown command, use real/complex/transformer'
fi
"

