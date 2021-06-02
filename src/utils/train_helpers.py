import time
import datetime
import random
import progressbar
import numpy as np


def flat_accuracy(preds, labels):
    """
        the following section is based on:
        https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
        the following section is based on:
        https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
