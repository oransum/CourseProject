import pandas as pd
import json
import torch
import datetime
import nlpaug.augmenter.word as naw


def get_raw_data():
    with open('train.jsonl.txt') as f:
        lines = f.read().splitlines()

    train_raw_data = pd.DataFrame(lines)
    train_raw_data.columns = ['json_element']
    train_raw_data = pd.json_normalize(train_raw_data['json_element'].apply(json.loads))
    
    sarcasm_data = train_raw_data[train_raw_data['label'] == 'SARCASM'].reset_index()
    not_sarcasm_data = train_raw_data[train_raw_data['label'] == 'NOT_SARCASM'].reset_index()
    
    val_size = int(len(sarcasm_data) *0.1)
    val_sarcasm_data = sarcasm_data.sample(val_size)
    train_sarcasm_data = sarcasm_data[~sarcasm_data['index'].isin(val_sarcasm_data['index'].tolist())]
    print(train_sarcasm_data.shape, val_sarcasm_data.shape)
    
    val_not_sarcasm_data = not_sarcasm_data.sample(val_size)
    train_not_sarcasm_data = not_sarcasm_data[~not_sarcasm_data['index'].isin(val_not_sarcasm_data['index'].tolist())]
    print(train_not_sarcasm_data.shape, val_not_sarcasm_data.shape)
    val_not_sarcasm_data
    
    train_data = pd.concat([train_sarcasm_data, train_not_sarcasm_data]).drop('index', axis=1)
    train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'NOT_SARCASM' else 1)
    train_data = train_data[['response', 'label']]
    
    val_data = pd.concat([val_sarcasm_data, val_not_sarcasm_data]).drop('index', axis=1)
    val_data['label'] = val_data['label'].apply(lambda x: 0 if x == 'NOT_SARCASM' else 1)
    val_data = val_data[['response', 'label']]

#     train_data.to_csv('train_data.csv', index=False)
#     val_data.to_csv('val_data.csv', index=False)
    
    return train_data
    
def generate(aug_ratio=1, total_aug_ratio=1):
    print('Before:', torch.get_num_threads())
    torch.set_num_threads(31)
    print('After:', torch.get_num_threads())
    
    train_data = get_raw_data()
    
    sarcasm_data = train_data[train_data['label'] == 1]
    sarcasm_data = sarcasm_data.sample(int(len(sarcasm_data) * aug_ratio))
    sarcasm_data = sarcasm_data['response'].tolist()
    not_sarcasm_data = train_data[train_data['label'] == 0]
    not_sarcasm_data = not_sarcasm_data.sample(int(len(not_sarcasm_data) * aug_ratio))
    not_sarcasm_data = not_sarcasm_data['response'].tolist()
    print(len(sarcasm_data), len(not_sarcasm_data))
    
    total_generate_size = (len(sarcasm_data) + len(not_sarcasm_data)) * total_aug_ratio

    
    bert_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", stopwords=['@', 'user'])
    roberta_aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", stopwords=['@', 'user'])

    aug_sarcasm_data = []
    for _ in range(total_aug_ratio):
        print(datetime.datetime.now(), 'total_aug_ratio:', _)
        for aug in [bert_aug, roberta_aug]:
            print(datetime.datetime.now(), aug)
            aug_sarcasm_data.extend(aug.augment(sarcasm_data))

    aug_sarcasm_data = [a.replace('@ user', '@user') for a in aug_sarcasm_data]
    aug_sarcasm_data = [a.replace('[UNK]', '').strip() for a in aug_sarcasm_data]
    print(len(aug_sarcasm_data))
    print(aug_sarcasm_data[:3])

    aug_not_sarcasm_data = []
    for _ in range(total_aug_ratio):
        print(datetime.datetime.now(), 'total_aug_ratio:', _)
        for aug in [bert_aug, roberta_aug]:
            print(datetime.datetime.now(), aug)
            aug_not_sarcasm_data.extend(aug.augment(not_sarcasm_data))

    aug_not_sarcasm_data = [a.replace('@ user', '@user') for a in aug_not_sarcasm_data]
    aug_not_sarcasm_data = [a.replace('[UNK]', '').strip() for a in aug_not_sarcasm_data]
    print(len(aug_not_sarcasm_data))
    print(aug_not_sarcasm_data[:3])

    aug_sarcasm_df = pd.DataFrame(aug_sarcasm_data)
    aug_sarcasm_df.columns = ['response']
    aug_sarcasm_df['label'] = 1
    aug_sarcasm_df['aug'] = True

    not_aug_sarcasm_df = pd.DataFrame(aug_not_sarcasm_data)
    not_aug_sarcasm_df.columns = ['response']
    not_aug_sarcasm_df['label'] = 0
    not_aug_sarcasm_df['aug'] = True

    train_data['aug'] = False
    aug_train_data = pd.concat([train_data, aug_sarcasm_df, not_aug_sarcasm_df])
    print(aug_train_data.shape)

    aug_train_data = pd.concat([train_data, aug_sarcasm_df, not_aug_sarcasm_df])
    print(aug_train_data.shape)
    
    aug_train_data.to_csv('aug_train_data_{}.csv'.format(total_generate_size), index=False)

    
if __name__ == '__main__':
    generate(aug_ratio=1, total_aug_ratio=10)