import torch

from src import util, network

import os
from torchtext.datasets.ag_news import AG_NEWS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from functools import partial
from transformers import AdamW
from pprint import pprint


def run_network(device):
    bert_name = 'bert'
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

    # check how the encoder/decoder works on a single input after encoding and batching
    for labels, batch in train_loader:
        print('Example of decoding encoded text with bert toknizer:')
        pprint(tokenizer.decode(batch['input_ids'][0]))
        break

    model = network.bert_models.Model(num_labels=num_labels)
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-5)

    # demonstrating a single pass through a network with distil berg configs

    model.eval()
    with torch.no_grad():
        for labels, encoded_batch in train_loader:
            optim.zero_grad()
            input_ids = encoded_batch['input_ids'].to(device)
            attention_mask = encoded_batch['attention_mask'].to(device)
            token_type_ids = encoded_batch['token_type_ids'].to(device)
            # TODO Incorporate labels into model ouput and loss function
            labels = labels.to(device)
            output = model(input_ids=input_ids
                            , attention_mask=attention_mask
                            , token_type_ids=token_type_ids
                            , labels=labels
                            )
            print(output.size())

            break

    # add additional things here