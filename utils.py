import logging
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
