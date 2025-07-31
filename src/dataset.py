import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)


def prepare_data(df, min_user_inter=5, min_item_inter=5):
    """
    Загружает и фильтрует данные
    :param df: pd.Dataframe со столбцами ['user_id', 'item_id', 'datetime']
    :param min_user_inter: минимальное кол-во интеракций на user_id
    :param min_item_inter: минимальное кол-во интеракций для item_id
    """

    # Фильтрация: удаляем пользователей и фильмы с слишком малым числом взаимодействий
    logger.info(f'Filter users with interactions < {min_user_inter} and items with interactions < {min_item_inter}')
    while True:
        before = len(df)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_user_inter].index)]
        df = df[df['item_id'].isin(item_counts[item_counts >= min_item_inter].index)]
        after = len(df)
        if before == after:
            break

    # Энкодинг
    logger.info('Encoding')
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])

    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    logger.info(f'N users: {num_users}, N items: {num_items}')

    # Группировка: user_id -> [item_id1, item_id2, ...]
    logger.info('Groupping')
    df = df.sort_values(by=['user_id', 'datetime'])
    user_seqs = df.groupby('user_id')['item_id'].apply(list).to_dict()

    return user_seqs, num_users, num_items, user_encoder, item_encoder

def split_sequence(seq, min_len=3):
    """
    Делит последовательность на train / valid / test
    Последний item — test, предпоследний — valid, остальные — train
    :param df:
    """
    if len(seq) < min_len:
        return [], [], []
    return seq[:-2], [seq[-2]], [seq[-1]]

def build_dataset(user_seqs, max_len=50):
    """
    Строит train/valid/test списки для каждого пользователя
    :param max_len: максимальная длина последовательности
    """
    train_data = []
    valid_data = []
    test_data = []

    for user, seq in user_seqs.items():
        train_seq, valid_item, test_item = split_sequence(seq)
        if not train_seq:
            continue

        train_data.append((user, train_seq))
        if valid_item:
            valid_data.append((user, train_seq + valid_item))
        if test_item:
            test_data.append((user, train_seq + valid_item + test_item))

    return train_data, valid_data, test_data

class SASRecDataset(Dataset):
    def __init__(self, data, max_len=50, num_items=10000):
        self.data = data
        self.max_len = max_len
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, seq = self.data[index]
        seq = seq[-self.max_len:]

        # Padding
        seq_padded = [0] * (self.max_len - len(seq)) + seq[:-1]
        target = seq[-1]

        return (
            torch.tensor(user_id, dtype=torch.long),       # user_id
            torch.tensor(seq_padded, dtype=torch.long),    # вход
            torch.tensor(target, dtype=torch.long)         # правильный item
                 
        )
