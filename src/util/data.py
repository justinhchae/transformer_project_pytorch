from src.util import constants
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

pd.set_option('display.max_columns', None)


def get_data(train_filename='train.csv'
             , test_filename='test.csv'
             ):
    test_df = None

    train_filepath = os.sep.join([constants.DATA_PATH, train_filename])
    train_df = pd.read_csv(train_filepath)

    if test_filename is not None:
        test_filepath = os.sep.join([constants.DATA_PATH, test_filename])
        test_df = pd.read_csv(test_filepath)

    return train_df, test_df


class TextOnlyDataset(Dataset):
    def __init__(self
                 , df
                 , text_col
                 , label_col
                 , padding=True
                 , tensor_type="pt"
                 , sample_size=None
                 ):
        self.df = df if sample_size is None else df.sample(sample_size)
        self.text_col = text_col
        self.label_col = label_col
        # self.tokenizer = tokenizer
        self.label = None
        self.tensor_type = tensor_type
        self.padding = padding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_col]
        return text


def make_tokenizer(bert_case_type):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers'
                               , 'tokenizer'
                               , f'bert-base-{bert_case_type}')
    return tokenizer

