# Complex Transformer
A deep learning model which incorporates the transformer model as a backbone and develop complex attention and encoder/decoder network operating on complex-valued input.<br />
This model achieves state-of-the-art results on music transcription tasks. The detail is in a paper we will publish soon.<br />

## Dependency
`python=3.6`
## transformer model
<p align="center">
  <img src="https://github.com/martinmamql/dl_signal/blob/master/img/transformer.png" alt="Complex Transformer"/>
</p>
## File parsing
`cd music/`<br />
`wget  https://homes.cs.washington.edu/~thickstn/media/musicnet.npz`<br />
`python3 -u resample.py musicnet.npz musicnet_11khz.npz 44100 11000`<br />
`rm musicnet.npz`<br />
`python3 -u parse_file.py`<br />
`rm musicnet_11khz.npz`<br />
`cd ..`<br />
(This process is quite long)

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
`python -u lstm_music_gen.py`<br />

LSTM for IQ generation: <br />
`python -u lstm_iq_gen.py`<br />

## Path Configuration
Example: `python -u transformer_train.py --path PATH`
## Parameter Tuning
All the parameters you can tune are in the argparser section of train*.py or lstm*.py file.


