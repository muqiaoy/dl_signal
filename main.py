# Hyperparameters are listed in the beginning of main().
# The loss is calculated in NLLLoss.
import os
def main():
    # Choice of model
    model = "gru" # fnn/gru/tcn, all in lower case
    
    # Data path
    path = "/usr0/home/yaohungt/Dropbox/CMU/research/data_RFMLS/pre_processed_TA1"
    
    # Model parameter
    params_model = {
        'batch_size'   : 10,
        
        #for fnn, hidden size will be applied to every hidden layer
        'hidden_size'  : 100,
        'num_layers'   : 3,
        
        #drop out currently not implemented in fnn
        'dropout'      : 0.2,
        'learning rate': 0.05,
        'momentum'     : 0.95,
        'weight_decay' : 0.0,
    }
    
    batch_size     = params_model['batch_size']
    hidden_size    = params_model['hidden_size']
    num_layers     = params_model['num_layers']
    dropout        = params_model['dropout']
    learning_rate  = params_model['learning rate']
    momentum       = params_model['momentum']
    weight_decay   = params_model['weight_decay']
    
    os.system("python "+ model +".py" +
              " --path " + path +
              " --batch_size " + str(batch_size) +
              " --hidden_size " + str(hidden_size) +
              " --num_layers " + str(num_layers) +
              " --dropout " + str(dropout) +
              " --learning_rate " + str(learning_rate) +
              " --momentum " + str(momentum) +
              " --weight_decay " + str(weight_decay))

if __name__ == '__main__':
    main()

