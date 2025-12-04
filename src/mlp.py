import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
