
from src import utils, bert_sequence
from src.bert_sequence.train_epoch import train_epoch
from src.bert_sequence.test_epoch import test_epoch

import torch
from transformers import get_linear_schedule_with_warmup

from functools import partial
from transformers import AdamW
from pprint import pprint
import numpy as np
import time
import random
import progressbar


def network(device, use_seed=False):
    if use_seed:
        random.seed(utils.constants.SEED_VAL)
        np.random.seed(utils.constants.SEED_VAL)
        torch.manual_seed(utils.constants.SEED_VAL)
        torch.cuda.manual_seed_all(utils.constants.SEED_VAL)

    # TODO: structure/package network and training parameters
    bert_name = 'bert'
    bert_case_type = 'uncased'
    bert_type = f"{bert_name}-base-{bert_case_type}"
    bert_variation = "modelForSequenceClassification"
    batch_size = 16
    tensor_type = "pt"
    shuffle_dataloader = True
    epochs = 3
    learning_rate = 5e-5
    # set tdo_break_testing o true to run a single train loop in each epoch for debugging
    do_break_testing = False

    # make a tokenizer from HF library
    tokenizer = utils.make_tokenizer(bert_name, bert_case_type)
    # freeze partial function signature
    tokenizer_ = partial(tokenizer, padding=True, truncation=True, add_special_tokens=True, return_tensors=tensor_type)
    # make data loader objects and apply tokenizer
    data = utils.data.get_corpora(tokenizer=tokenizer_, batch_size=batch_size, shuffle_dataloader=shuffle_dataloader)

    train_loader = data['train_loader']
    test_loader = data['test_loader']
    num_labels = data['num_labels']

    # check how the encoder/decoder works on a single input after encoding and batching
    utils.data.demo_encoder_decoder(train_loader, tokenizer)

    model = bert_sequence.model.Model(num_labels=num_labels, bert_type=bert_type, bert_variation=bert_variation)
    model.to(device)

    optim = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optim
                                                , num_warmup_steps=0
                                                , num_training_steps=total_steps)

    """
    the following section is based on: 
    https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """

    # validation accuracy, and timings.
    training_stats = []
    curr_step = 0
    total_t0 = time.time()

    print("=" * 10, 'Starting Training\n')

    with progressbar.ProgressBar(max_value=total_steps) as progress:

        for epoch_i in range(0, epochs):
            t0 = time.time()
            avg_train_loss, curr_step = train_epoch(model, train_loader, optim, progress, scheduler, curr_step, device
                                         , break_test=do_break_testing)
            print("")
            print("=" * 20, f"Epoch: {epoch_i}", "=" * 20)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            avg_test_accuracy = test_epoch(model, test_loader, device
                                       , break_test=do_break_testing)

            print("")
            print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
            print("=" * 20)


        # TODO:Clean up training structures and metrics
        #
        #     training_time = utils.train_helpers.format_time(time.time() - t0)
        #
        #     print("")
        #     print("  Average training loss: {0:.2f}".format(ave_train_loss))
        #     print("  Training epcoh took: {:}".format(training_time))
        #
        #     print("")
        #     print("Running Test Set...")
        #
        #     t0 = time.time()
        #
        #     # Put the model in evaluation mode--the dropout layers behave differently
        #     # during evaluation.
        #     model.eval()
        #
        #     # Tracking variables
        #     total_eval_accuracy = 0
        #     total_eval_loss = 0
        #     nb_eval_steps = 0
        #
        #     for labels, encoded_batch in test_loader:
        #
        #         input_ids = encoded_batch['input_ids'].to(device)
        #         attention_mask = encoded_batch['attention_mask'].to(device)
        #         token_type_ids = encoded_batch['token_type_ids'].to(device)
        #         labels = labels.to(device)
        #
        #         with torch.no_grad():
        #             output = model(input_ids=input_ids
        #                             , attention_mask=attention_mask
        #                             , token_type_ids=token_type_ids
        #                             , labels=labels
        #                             )
        #         loss = output.loss
        #         total_train_loss += loss.item()
        #
        #         logits = output.logits.detach().cpu().numpy()
        #         labels = labels.to('cpu').numpy()
        #
        #         total_eval_accuracy += utils.train_helpers.flat_accuracy(logits, labels)
        #
        #     # Report the final accuracy for this validation run.
        #     avg_test_accuracy = total_eval_accuracy / len(test_loader)
        #     print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
        #
        #     # Calculate the average loss over all of the batches.
        #     avg_test_loss = total_eval_loss / len(test_loader)
        #
        #     # Measure how long the validation run took.
        #     validation_time = utils.train_helpers.format_time(time.time() - t0)
        #
        #     print("  Test Loss: {0:.2f}".format(avg_test_loss))
        #     print("  Test took: {:}".format(validation_time))
        #
        #     # Record all statistics from this epoch.
        #     training_stats.append(
        #         {
        #             'epoch': epoch_i + 1,
        #             'Training Loss': ave_train_loss,
        #             'Test. Loss': avg_test_loss,
        #             'TEst. Accur.': avg_test_accuracy,
        #             'Training Time': training_time,
        #             'Test Time': validation_time
        #         }
        #     )
        #
        # print("")
        # print("Training complete!")
        #
        # print("Total training took {:} (h:mm:ss)".format(utils.train_helpers.format_time(time.time() - total_t0)))
        # pprint(training_stats)
