import torch
from torch import nn


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()

    def forward(self, x):
        return x*100-50

test = Test()
x = torch.tensor(1.0)
output = test(x)
print(output)
