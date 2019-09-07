# Complex Transformer
A deep learning model which incorporates the transformer model as a backbone and develop complex attention and encoder/decoder network operating on complex-valued input.<br />
This model achieves state-of-the-art results on music transcription tasks. The detail is in a paper we will publish soon.

## transformer model
<p align="center">
  <img src="https://github.com/martinmamql/dl_signal/blob/master/img/transformer.png" alt="Complex Transformer"/>
</p>

## Instruction
Preprocess the MusicNet dataset as stated in the paper: <br />
`python parse_file.py`<br />

Sample command line for automatic music transcription: <br />
`python -u transformer/train.py`<br />

Concatenated transformer for automatic music transcription: <br />
`python -u transformer/train_concat.py`<br />

For MusicNet generation tasks: <br />
`python -u transformer/train_gen.py`<br />

Concatenated transformer for MusicNet generation: <br />
`python -u transformer/train_gen_concat.py`<br />

For IQ classification task: <br />
`python -u transformer/train_iq.py`<br />

Concatenated transformer for IQ classification task: <br />
`python -u transformer/train_iq_concat.py`<br />

For IQ generation tasks: <br />
`python -u transformer/train_gen_iq.py`<br />

Concatenated transformer for IQ generation: <br />
`python -u transformer/train_gen_iq_concat.py`<br />

LSTM for MusicNet generation: <br />
<<<<<<< HEAD
`python -u lstm_music_gen.py`<br />

LSTM for IQ generation: <br />
`python -u lstm_iq_gen.py`<br />
=======
`python -u lstm_music_gen.py --path path --batch_size 32 --hidden_size 800 --num_layers 3 --fc_hidden_size 2048 --output_dim 128 --lr 0.001 --dropout 0.5  --src_time_step 40 --trg_time_step 24`<br />

LSTM for IQ generation: <br />
`python -u lstm_iq_gen.py --path path --batch_size 128 --hidden_size 800 --num_layers 3 --fc_hidden_size 800 --src_time_step 40 --trg_time_step 24 --output_dim 50`<br />
>>>>>>> 45c40071f4733c4b39c1d9cd1e501b5eaf360b2c
<!-- To configure parameters, directly configure it through command line.<br />
## Path Configuration
Example: `python -u transformer_train.py --path PATH`
## Parameter Tuning
All the parameters you can tune are in the argparser section of train*.py or lstm*.py file.


