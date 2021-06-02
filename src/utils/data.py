from src import utils

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets.ag_news import AG_NEWS
from torchtext.datasets.dbpedia import DBpedia
from torchtext.datasets.imdb import IMDB
from torchtext.datasets.amazonreviewpolarity import AmazonReviewPolarity
from torchtext.datasets.yelpreviewpolarity import YelpReviewPolarity
from torchtext.datasets.yelpreviewfull import YelpReviewFull
from torchtext.datasets.sogounews import SogouNews
from torchtext.datasets.yahooanswers import YahooAnswers
from functools import partial
from pprint import pprint

pd.set_option('display.max_columns', None)


def get_torch_corpora(torch_corpora_name):
    corpora_path = os.sep.join([utils.constants.DATA_PATH, torch_corpora_name])

    if "ag_news" in torch_corpora_name:
        train_iter, test_iter = AG_NEWS(root=corpora_path)
    elif "dbpedia" in torch_corpora_name:
        train_iter, test_iter = DBpedia(root=corpora_path)
    elif "imdb" in torch_corpora_name:
        train_iter, test_iter = IMDB(root=corpora_path)
    elif "amazon_polarity" in torch_corpora_name:
        train_iter, test_iter = AmazonReviewPolarity(root=corpora_path)
    elif "yelp_polarity" in torch_corpora_name:
        train_iter, test_iter = YelpReviewPolarity(root=corpora_path)
    elif "yelp_review" in torch_corpora_name:
        train_iter, test_iter = YelpReviewFull(root=corpora_path)
    elif "sogou_news" in torch_corpora_name:
        train_iter, test_iter = SogouNews(root=corpora_path)
    elif "yahoo_answers" in torch_corpora_name:
        train_iter, test_iter = YahooAnswers(root=corpora_path)
    else:
        return None, None

    return train_iter, test_iter


def get_corpora(tokenizer, batch_size, shuffle_dataloader, split_train_data=False, torch_corpora_name="ag_news"
                , tensor_type="pt"):
    # use a simple, pre-canned dataset for multi-class text classification
    train_iter, test_iter = get_torch_corpora(torch_corpora_name=torch_corpora_name)
    train_data, test_data = list(train_iter), list(test_iter)

    # freeze partial function signature
    tokenizer = partial(tokenizer, padding=True, truncation=True, add_special_tokens=True, return_tensors=tensor_type)
    # make data loader objects and apply tokenizer

    # for batching, apply collate function with tokenizer in data loader objects
    collate_fn = partial(utils.collate_batch, tokenizer=tokenizer)

    # count number of labels to pass to classification model
    labels_list = set([label for (label, text) in train_data])
    num_labels = len(labels_list)

    if split_train_data:
        # count training/validation split from a single train set
        num_train = int(len(train_data) * 0.95)
        split_train_, split_valid_ = random_split(train_data, [num_train, len(train_data) - num_train])
        train_loader = DataLoader(split_train_, batch_size=batch_size, shuffle=shuffle_dataloader, collate_fn=collate_fn)
        valid_loader = DataLoader(split_valid_, batch_size=batch_size, shuffle=shuffle_dataloader, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_dataloader, collate_fn=collate_fn)
        valid_loader = None

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_dataloader, collate_fn=collate_fn)

    data = {'train_loader': train_loader, 'test_loader': test_loader, 'num_labels': num_labels, 'labels_list': labels_list}

    if valid_loader is not None:
        data.update({'valid_loader': valid_loader})

    return data


def make_tokenizer(bert_name, bert_case_type, path='huggingface/pytorch-transformers'):
    tokenizer = torch.hub.load(path, 'tokenizer', f'{bert_name}-base-{bert_case_type}')
    return tokenizer


def label_pipeline(sentiment_map, x):
    if isinstance(x, str):
        x_prime = sentiment_map[x]
    else:
        x_prime = x

    return x_prime - 1


def collate_batch(batch, tokenizer):
    labels = []
    batch_texts = []
    sentiment_map = {"pos": 2,
                     "neg": 1}

    # https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    # label_pipeline = lambda x: int(x) - 1

    for (_label, batch_text) in batch:
        labels.append(label_pipeline(sentiment_map, _label))
        batch_texts.append(batch_text)

    labels = torch.tensor(labels, dtype=torch.long)
    encoded_batch = tokenizer(batch_texts)

    return labels, encoded_batch


def demo_encoder_decoder(data_loader, tokenizer, torch_corpora_name):
    print("=" * 20, f"Corpora Name: {torch_corpora_name}")
    for labels, batch in data_loader:
        print('Example of decoding encoded text with bert tokenizer:')
        pprint(tokenizer.decode(batch['input_ids'][0]))
        print("=" * 40)
        break

