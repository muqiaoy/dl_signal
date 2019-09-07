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
`python -u lstm_music_gen.py`<br />

LSTM for IQ generation: <br />
`python -u lstm_iq_gen.py`<br />

## Path Configuration
Example: `python -u transformer_train.py --path PATH`
## Parameter Tuning
All the parameters you can tune are in the argparser section of train*.py or lstm*.py file.


