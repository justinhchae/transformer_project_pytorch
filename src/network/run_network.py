

from src import util, network

import torch
from transformers import get_linear_schedule_with_warmup

import os
from torchtext.datasets.ag_news import AG_NEWS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from functools import partial
from transformers import AdamW
from pprint import pprint
import numpy as np
import time
import datetime
import random
import progressbar


def run_network(device):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    bert_name = 'bert'
    bert_case_type = 'uncased'
    # set batch size (could be as high as 32 or so)
    batch_size = 32
    tensor_type = "pt"

    # make a tokenizer from HF library
    tokenizer = util.make_tokenizer(bert_name, bert_case_type)
    tokenizer_ = partial(tokenizer
                         , padding=True
                         , truncation=True
                         , add_special_tokens=True
                         , return_tensors=tensor_type
                         )

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
    print("=" * 40)
    for labels, batch in train_loader:
        print('Example of decoding encoded text with bert toknizer:')
        pprint(tokenizer.decode(batch['input_ids'][0]))
        print("=" * 40)
        break

    model = network.bert_models.Model(num_labels=num_labels)
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-5)
    epochs = 2
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optim
                                                , num_warmup_steps=0
                                                , num_training_steps=total_steps)

    """
    the following section is based on: 
    https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # validation accuracy, and timings.
    training_stats = []
    curr_step = 0
    print("=" * 10, 'Starting Training\n')

    with progressbar.ProgressBar(max_value=total_steps) as progress:

        for epoch_i in range(0, epochs):
            t0 = time.time()
            total_train_loss = 0

            model.train()

            for labels, encoded_batch in train_loader:

                input_ids = encoded_batch['input_ids'].to(device)
                attention_mask = encoded_batch['attention_mask'].to(device)
                token_type_ids = encoded_batch['token_type_ids'].to(device)
                labels = labels.to(device)

                model.zero_grad()
                output = model(input_ids=input_ids
                                , attention_mask=attention_mask
                                , token_type_ids=token_type_ids
                                , labels=labels
                                )
                loss = output.loss
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optim.step()
                scheduler.step()
                progress.update(curr_step)
                curr_step += 1

            ave_train_loss = total_train_loss / len(train_loader)

            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(ave_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
