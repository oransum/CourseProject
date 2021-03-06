from model import *

if __name__ == "__main__":
    model_type = 'bert'
    model_name = 'bert-base-uncased'
    exp_version = 'bert_with_aug_9000'
    num_train_epochs = 10
    
    train_data = get_aug_train_data_9000()
    val_data = get_val_data()
    
    training(model_type, model_name, exp_version, train_data, val_data, 
             num_train_epochs=num_train_epochs, use_gpu=False)