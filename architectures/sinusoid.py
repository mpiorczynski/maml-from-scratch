import torch.nn as nn


def get_sinusoid_model(_args):
    model = nn.Sequential(
        nn.Linear(1, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 1),
    )
    return model
