from collections import OrderedDict

import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, 3)
        self.bn = nn.BatchNorm2d(n_out, momentum=1.0, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        return x


def get_omniglot_model(args):
    model = nn.Sequential(
        OrderedDict(
            [
                ("block0", CNNBlock(1, 64)),
                ("block1", CNNBlock(64, 64)),
                ("block2", CNNBlock(64, 64)),
                ("flatten", nn.Flatten()),
                ("linear", nn.Linear(64, args.n_way)),
            ]
        )
    )
    return model
