import pandas as pd
import json
from simpletransformers.classification import ClassificationModel

from model import *

def get_data(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()

    test_raw_data = pd.DataFrame(lines)
    test_raw_data.columns = ['json_element']
    
    return pd.json_normalize(test_raw_data['json_element'].apply(json.loads))

def get_model(model_type, file_path):
    model = ClassificationModel(
        model_type, file_path, use_cuda=False, args={}
    )
    
    return model

def predict():
    data = get_data('test.jsonl.txt')
    model = get_model('roberta', 'roberta_with_aug_45000/outputs/checkpoint-18000/')
    
    y_pred, y_prob = model.predict(data['response'].tolist())
    data['y_pred'] = ['SARCASM' if y == 1 else 'NOT_SARCASM' for y in y_pred]
    data[['id', 'y_pred']].to_csv('answer.txt', index=False, header=False)

if __name__ == "__main__":
    predict()