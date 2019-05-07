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

<!---
## Instruction
An example command line to start the training and testing.<br />
`python gru.py --path ../dataset/TA1 --batch_size 100 --hidden_size 100 --num_layers 2 --dropout 0.1 --learning_rate 0.05 --momentum 0.95 --weight_decay 0 --input_size 160 --time_step 20 --feature_dim 160`<br />
To configure parameters, directly configure it through command line.<br />
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
- `rnn_iq.py`: train and test setup of rnn model for the iq dataset
-->

