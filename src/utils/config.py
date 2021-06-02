from src.utils import constants

import os
import torch


def run():
    make_dirs()
    device = set_device()
    return device


def set_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def make_dirs():
    if not os.path.exists(constants.DATA_PATH):
        os.makedirs(constants.DATA_PATH)

    if not os.path.exists(constants.CACHE_PATH):
        os.makedirs(constants.CACHE_PATH)

    if not os.path.exists(constants.FIG_PATH):
        os.makedirs(constants.FIG_PATH)

    if not os.path.exists(constants.MODEL_PATH):
        os.makedirs(constants.MODEL_PATH)
