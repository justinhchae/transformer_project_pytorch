from src.util import constants
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

pd.set_option('display.max_columns', None)


def get_data(data_full_path, header=1):
    train_df = pd.read_csv(data_full_path, header=header)

    return train_df


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
        self.label = None
        self.tensor_type = tensor_type
        self.padding = padding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_col]
        label = self.df.iloc[idx][self.label_col]
        return text


def make_tokenizer(bert_name, bert_case_type):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers'
                               , 'tokenizer'
                               , f'{bert_name}-base-{bert_case_type}')
    return tokenizer


def collate_batch(batch, tokenizer):
    labels = []
    batch_texts = []

    for (_label, batch_text) in batch:
        labels.append(_label)
        batch_texts.append(batch_text)

    labels = torch.tensor(labels, dtype=torch.long)
    encoded_batch = tokenizer(batch_texts)

    return labels, encoded_batch


