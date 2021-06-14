"""
    the following section is based on:
    https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""

from src import utils
import torch


def test_epoch(model, data_loader, device, break_test=False):
    model.eval()

    # Tracking variables
    total_test_accuracy = 0
    total_test_loss = 0

    for counter, (labels, encoded_batch) in enumerate(data_loader):

        input_ids = encoded_batch['input_ids'].to(device)
        attention_mask = encoded_batch['attention_mask'].to(device)
        # token_type_ids = encoded_batch['token_type_ids'].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        total_test_loss += loss.item()

        logits = output.logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        total_test_accuracy += utils.train_helpers.flat_accuracy(logits, labels)

        if break_test:
            if counter == 2:
                break

    # Report the final accuracy for this test run.
    avg_test_accuracy = total_test_accuracy / len(data_loader)
    avg_test_loss = total_test_loss / len(data_loader)

    return avg_test_accuracy, avg_test_loss
