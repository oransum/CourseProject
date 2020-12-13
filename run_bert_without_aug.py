from model import *

if __name__ == "__main__":
    model_type = 'bert'
    model_name = 'bert-base-uncased'
    exp_version = 'bert_without_aug'
    num_train_epochs = 10
    
    train_data = get_train_data()
    val_data = get_val_data()
    
    training(model_type, model_name, exp_version, train_data, val_data, 
             num_train_epochs=num_train_epochs, use_gpu=False)