import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

datasets = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=False, transform=torchvision.transforms.ToTensor(),download=True)
load_data = DataLoader(datasets, batch_size=64, drop_last= True)    # 如果没有这行，则出来的imgs是torch.Size([3, 32, 32])

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear1 = nn.Linear(in_features=196608, out_features=10, bias=True)

    def forward(self, x):
        output = self.linear1(x)
        return output

linear = Linear()

for data in load_data:
    imgs, targets = data
    print(imgs.shape)
    # imgs = torch.reshape(imgs, (1, 1, 1, -1))
    imgs = torch.flatten(imgs) # 根据文档，其功能就是把Tensor摊平到1维，例如此时的batch_size=64,channel=3的32*32的Tensor摊平后时1966008的一维Tensor
    print(imgs.shape)
    imgs = linear(imgs)
    print(imgs.shape)