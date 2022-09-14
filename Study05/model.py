import torch
from torch import nn


# 搭建神经网络
class CIFAR10_NN(nn.Module):
    def __init__(self):
        super(CIFAR10_NN, self).__init__()
        self.Model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        output = self.Model(x)
        return output


if __name__ == '__main__':
    input = torch.randn(size=(64, 3, 32, 32))
    model = CIFAR10_NN()
    output = model(input)
    print(output.shape)