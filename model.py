import pandas as pd
import json
from simpletransformers.classification import ClassificationModel
import datetime


def get_train_data():
    return pd.read_csv('train_data.csv')

def get_val_data():
    return pd.read_csv('val_data.csv')

def get_aug_train_data_4500():
    ratio = 1
    train_data = get_aug_train_data_9000()
    aug_train_data = train_data[train_data['aug']]
    raw_train_data = train_data[~train_data['aug']]
    return pd.concat([
        raw_train_data, aug_train_data.sample(int(len(raw_train_data) * ratio))])

def get_aug_train_data_2250():
    ratio = 0.5
    train_data = get_aug_train_data_9000()
    aug_train_data = train_data[train_data['aug']]
    raw_train_data = train_data[~train_data['aug']]
    return pd.concat([
        raw_train_data, aug_train_data.sample(int(len(raw_train_data) * ratio))])
    
def get_aug_train_data_9000():
    return pd.read_csv('aug_train_data.csv')

def get_aug_train_data_45000():
    return pd.read_csv('aug_train_data_45000.csv')

def get_duration(start):
    seconds = (datetime.datetime.now() - start).total_seconds()
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def write_duration(start, file_path):
    with open(file_path, "w") as f:
        total = get_duration(start)
        f.write(str(total[0]) + ':' + str(total[1]) + ':' + str(total[2]))
        f.close()

def training(model_type, model_name, exp_version, train_data, val_data, 
              num_train_epochs=10, use_gpu=False, use_early_stopping=True):
    args = {
        'num_train_epochs': num_train_epochs,
        'overwrite_output_dir': True,
        'use_early_stopping': use_early_stopping,
        'output_dir': exp_version + '/outputs/',
        'cache_dir': exp_version + '/cache/',
        'best_model_dir': exp_version + '/outputs/best_model/',
    }

    model = ClassificationModel(model_type, model_name, use_cuda=use_gpu, args=args)
    
    args = {
        'evaluate_during_training': True,
        'evaluate_each_epoch': True
    }
    
    start = datetime.datetime.now()
    model.train_model(train_data, eval_df=val_data, args=args)
    duration = get_duration(start)
    write_duration(start, exp_version + '/duration.txt')