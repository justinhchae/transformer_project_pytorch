import torch

from src import util, network

import os
from torchtext.datasets.ag_news import AG_NEWS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from functools import partial
from transformers import AdamW
from transformers import DistilBertConfig, DistilBertModel


def run_network(device):
    bert_name = 'distilbert'
    bert_case_type = 'cased'
    # set batch size (could be as high as 32 or so)
    batch_size = 4
    tensor_type = "pt"

    # make a tokenizer from HF library
    tokenizer = util.make_tokenizer(bert_name, bert_case_type)
    tokenizer_ = partial(tokenizer, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt")

    # download a simple dataset and read it from disk to simulate reading from a custom dataset
    ag_news_path = os.sep.join([util.constants.DATA_PATH, 'ag_news'])
    train_iter, test_iter = AG_NEWS(root=ag_news_path)
    train_data, test_data = list(train_iter), list(test_iter)

    num_labels = len(set([label for (label, text) in train_data]))
    num_train = int(len(train_data) * 0.8)

    split_train_, split_valid_ = random_split(train_data, [num_train, len(train_data) - num_train])

    collate_fn = partial(util.collate_batch, tokenizer=tokenizer_)

    train_loader = DataLoader(split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = network.bert_models.Model()
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-5)

    # testing out network with model config from distil bert
    for epoch in range(1):
        model.train()
        for labels, encoded_batch in train_loader:
            optim.zero_grad()
            input_ids = encoded_batch['input_ids'].to(device)
            attention_mask = encoded_batch['attention_mask'].to(device)

            labels = labels.to(device)
            outputs = model(x=input_ids, att=attention_mask)
            print(outputs.size())

            break
        break

    #
