# Deep Learning Model for Signal Data Prediction

## Instruction
Run `main.py` to start the training and testing.<br />
To configure parameters, open `main.py`.<br />
## Model Selection
If we want to train the dataset using fnn, set <br />
`model = "fnn"`.<br />
if we want to train the dataset using gru, set <br />
`model = "gru"`.<br />

## Path Configuration
Set `path = "PATH OF YOUR DATASET(TA1 or TA2)"`, for example: `path = "data/TA1"`<br />
## Parameter Tuning
In the model parameter part of main.py, we could set the value of parameters directly, for example:<br />
Set<br />
`'batch_size'   : 10` <br />
to <br />
`'batch_size'   : 20` <br />
will change the batch size of the model from 10 to 20.<br />
The following contains all the parameters we have so far for tuning:<br />
```Python
    # Model parameter
    params_model = {
        'batch_size'   : 10,
        
        #for fnn, hidden size will be applied to every hidden layer
        'hidden_size'  : 100,
        'num_layers'   : 2,
        
        #drop out currently not implemented in fnn
        'dropout'      : 0.2,
        'learning rate': 0.05,
        'momentum'     : 0.95,
        'weight_decay' : 0.0,
    }
```
## Files
- main.py: configuration and hyper tuning.<br />
- models.py: including all the architecture of all models.<br />
- utils.py: data loading using pytorch dataloader and dataset
- fnn.py: train and test setup of fnn model
- gru.py: train and test setup of gru model
