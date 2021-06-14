from src import utils, bert_classifier
from src.bert_classifier.train_epoch import train_epoch
from src.bert_classifier.test_epoch import test_epoch

import torch
from transformers import get_linear_schedule_with_warmup

from functools import partial
from transformers import AdamW
from pprint import pprint
import numpy as np
import time
import random
import progressbar
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import copy


def fit(torch_corpora_name
        , do_break_testing=False
        , epochs=1
        , learning_rate=1e-5
        , warmup_steps=100
        , device=None
        , batch_size=16
        , shuffle_data_loader=True
        , tensor_type="pt"
        , tokenizer_max_len=128
        , tokenizer_kws=None
        , loss_function_type="mse_loss"
        , pre_trained_str="roberta-base"):
    # TODO: structure/package network and training parameters
    device = torch.device('cpu') if device is None else device

    _tokenizer = RobertaTokenizer.from_pretrained(pre_trained_str)
    _tokenizer_kws = {'padding': 'max_length'
                      , "max_length": tokenizer_max_len
                      , "truncation": True
                      , 'add_special_tokens': True
                      , 'return_tensors': tensor_type}
    tokenizer_kws = _tokenizer_kws if tokenizer_kws is None else tokenizer_kws
    tokenizer = partial(_tokenizer, **tokenizer_kws)

    data = utils.get_corpora(tokenizer=tokenizer, torch_corpora_name=torch_corpora_name)

    model = bert_classifier.Model(num_labels=data['num_labels'])
    model.to(device)

    optim = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(data["train_loader"]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optim
                                                , num_warmup_steps=warmup_steps
                                                , num_training_steps=total_steps)

    if "mse_loss" == loss_function_type:
        loss_func = F.mse_loss
    elif "l1_loss" == loss_function_type:
        loss_func = F.l1_loss
    elif "smooth_l1_loss" == loss_function_type:
        loss_func = F.smooth_l1_loss

    training_stats = []
    curr_step = 0
    total_t0 = time.time()

    print("=" * 10, f'Starting Training for {torch_corpora_name}', "=" * 10, "\n")

    with progressbar.ProgressBar(max_value=total_steps) as progress:

        for epoch_i in range(0, epochs):
            t0 = time.time()
            # run the train loop
            avg_train_loss, curr_step = train_epoch(model
                                                    , data['train_loader']
                                                    , optim
                                                    , progress
                                                    , scheduler
                                                    , curr_step
                                                    , device
                                                    , break_test=do_break_testing)
            print("")
            print("=" * 20, f"Epoch: {epoch_i + 1} | Corpora: {torch_corpora_name}", "=" * 20)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            # run the test loop
            avg_test_accuracy, avg_test_loss = test_epoch(model
                                                          , data['test_loader']
                                                          , device
                                                          , break_test=do_break_testing)
            print("  Test Loss: {0:.2f}".format(avg_test_loss))
            print("  Test Accuracy: {0:.2f}".format(avg_test_accuracy))
            print("=" * 20)
            epoch_time = utils.train_helpers.format_time(time.time() - t0)

            training_stats.append(
                {
                    'Corpora': torch_corpora_name,
                    'Epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Test Accuracy': avg_test_accuracy,
                    'Epoch Time': epoch_time,
                    'Number of Classes': data['num_labels'],
                    'Batch Size': batch_size,
                    'Bert Model': 'RoBERTa for Sequence Classification',
                    'Bert Type': pre_trained_str,

                }
            )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(utils.train_helpers.format_time(time.time() - total_t0)))
    pprint(training_stats)
    print("=" * 40, "\n")

    model_save = utils.data.save_model(model, f"{torch_corpora_name}_{pre_trained_str}")
    print(f"Saving model state dict to {model_save}")
    return training_stats
