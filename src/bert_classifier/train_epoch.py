"""
    the following section is based on:
    https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""

import torch


def train_epoch(model, data_loader, optimizer, progress, scheduler, curr_step, device, break_test=False):
    model.train()
    total_train_loss = 0

    for counter, (labels, encoded_batch) in enumerate(data_loader):
        input_ids = encoded_batch['input_ids'].to(device)
        attention_mask = encoded_batch['attention_mask'].to(device)
        # token_type_ids = encoded_batch['token_type_ids'].to(device)
        labels = labels.unsqueeze(1).to(device)

        model.zero_grad()

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = output.loss
        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        progress.update(curr_step)
        curr_step += 1

        if break_test:
            if counter == 2:
                break

    avg_train_loss = total_train_loss / len(data_loader)

    return avg_train_loss, curr_step
