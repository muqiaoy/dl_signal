# Deep Learning Model for Signal Data Prediction
This is a python package that is aimed to build various deep learning model on complex-valued time series data. <br /> 

As deep learning models are well developped in real-valued domains like image and text, we want to explore how should we apply those model to complex-valued domains such as signal, music and speech.<br />

This package includes models like fnn, rnn, gru, lstm, and transformer, and we test the performance of models based on iq and TA1 dataset.<br />

## transformer model
<p align="center">
  <img src="https://github.com/martinmamql/dl_signal/blob/master/img/transformer.png" alt="Complex Transformer"/>
</p>

## iq dataset:
Dimension of data:<br />

|               | Size                | Note                                               |
| ------------- |:-------------------:| --------------------------------------------------:|
| train         | (5589, 32, 1600, 2) | (outer_batch, inner_batch, time_step, feature_dim) |
| test          | (5589, 32, 1000)    | (outer_batch, inner_batch, class(one hot))         |


## Instruction
Preprocess the MusicNet dataset as stated in the paper: <br />
`python parse_file.py`<br />

Sample command line for automatic music transcription: <br />
`python -u transformer/train.py --data music --path path --time_step 64 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --modal_lengths 2048 2048 --embed_dim 320 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 2048 --nlevels 6 --batch_size 32`<br />

Concatenated transformer or automatic music transcription: <br />
`python -u transformer/train_concat.py --data music --path path --time_step 64 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --modal_lengths 2048 2048 --embed_dim 320 --output_dim 128 --num_heads 8 --seed 1111 --lr 0.001 --clip 0.35 --optim Adam --hidden_size 2048 --nlevels 6 --batch_size 32`<br />

For MusicNet generation tasks: <br />
`python -u transformer/train_gen.py --path path --lr 0.001 --hidden_size 2048 --output_dim 128 --num_epochs 2000 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --batch_size 32 --src_time_step 40 --trg_time_step 24 --nlevels 6 --embed_dim 320 --out_dropout 0.5`<br />

Concatenated transformer: <br />
`python -u transformer/train_gen_concat.py --path path --lr 0.001 --hidden_size 2048 --output_dim 128 --num_epochs 2000 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --batch_size 32 --src_time_step 40 --trg_time_step 24 --nlevels 6 --embed_dim 320 --out_dropout 0.5`<br />

For IQ generation tasks: <br />
`python -u transformer/train_gen_iq.py --path path --data iq --lr 0.001 --hidden_size 2048 --num_epochs 2000 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --batch_size 32 --src_time_step 40 --trg_time_step 24 --nlevels 6 --embed_dim 200 --out_dropout 0.5`<br />

Concatenated transformer: <br />
`python -u transformer/train_gen_iq_concat.py --path path --data iq --lr 0.001 --hidden_size 2048 --num_epochs 2000 --attn_dropout 0 --relu_dropout 0.1 --res_dropout 0.1 --batch_size 32 --src_time_step 40 --trg_time_step 24 --nlevels 6 --embed_dim 200 --out_dropout 0.5`<br />

LSTM for MusicNet generation: <br />
`python -u lstm_music_gen.py --path path --batch_size 8 --hidden_size 800 --num_layers 3 --fc_hidden_size 2048 --output_dim 128 --lr 0.001 --dropout 0.5  --src_time_step 40 --trg_time_step 24`<br />

LSTM for IQ generation: <br />
`python3.6 -u lstm_iq_gen.py --path path --batch_size 8 --hidden_size 800 --num_layers 3 --fc_hidden_size 800 --src_time_step 40 --trg_time_step 24 --output_dim 50   `<br />
<!-- To configure parameters, directly configure it through command line.<br />
## Model Selection
If we want to train the dataset using fnn, set <br />
`python fnn.py --path ......(as above)`.<br />
if we want to train the dataset using gru, set <br />
`python gru.py --path ......(as above)`.<br />

## Path Configuration
Set `path = "PATH OF YOUR DATASET(TA1 or TA2)"`, for example: `path = "data/TA1"`<br />
## Parameter Tuning
You could tune each parameter by change the value of specific parameter. For example, change the batch size to 200 would be:<br />
`...... --batch_size 20 ......`<br />
## Files
- `models.py`: including all the architecture of all models.<br />
- `utils.py`: data loading using pytorch dataloader and dataset
- `fnn.py`: train and test setup of fnn model
- `gru.py`: train and test setup of gru model
- `rnn.py`: train and test setup of rnn model
- `rnn_iq.py`: train and test setup of rnn model for the iq dataset -->


