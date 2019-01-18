# Deep Learning Model for Signal Data Prediction

## Instruction
An example command line to start the training and testing.<br />
`python gru.py --path ../dataset/TA1 --batch_size 100 --hidden_size 100 --num_layers 2 --dropout 0.1 --learning_rate 0.05 --momentum 0.95 --weight_decay 0`<br />
To configure parameters, directly configure it through command line.<br />
## Model Selection
If we want to train the dataset using fnn, set <br />
`python fnn.py --path ......(as above)`.<br />
if we want to train the dataset using gru, set <br />
`python gru.py --path ......(as above)`.<br />

## Path Configuration
Set `path = "PATH OF YOUR DATASET(TA1 or TA2)"`, for example: `path = "data/TA1"`<br />
## Parameter Tuning
In the command line example:<br />
`python gru.py --path ../dataset/TA1 --batch_size 100 --hidden_size 100 --num_layers 2 --dropout 0.1 --learning_rate 0.05 --momentum 0.95 --weight_decay 0`<br />
You could tune each parameter by change the value of specific parameter. For example, change the batch size to 200 would be:<br />
`...... --batch_size 20 ......`<br />
## Files
- `models.py`: including all the architecture of all models.<br />
- `utils.py`: data loading using pytorch dataloader and dataset
- `fnn.py`: train and test setup of fnn model
- `gru.py`: train and test setup of gru model
- `rnn.py`: train and test setup of rnn model
- `rnn_iq.py`: train and test setup of rnn model for the iq dataset

