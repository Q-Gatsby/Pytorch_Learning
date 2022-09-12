import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

datasets = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)
load_data = DataLoader(dataset=datasets, batch_size=1)


class CIF(nn.Module):
    def __init__(self):
        super(CIF, self).__init__()
        self.model1 = nn.Sequential(
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
        x = self.model1(x)
        return x


encropy_loss = nn.CrossEntropyLoss()
cif = CIF()
optim = torch.optim.SGD(cif.parameters(), lr=0.01)  # 定义一个优化器

for epoch in range(20):
    res_loss = 0
    for input, target in load_data:
        output = cif(input)
        loss = encropy_loss(output, target)
        optim.zero_grad()   # 网络中的参数清零
        for name, parms in cif.named_parameters():
            print('-->name:', name)
            print('-->para:', parms)
            print('-->grad_requirs:', parms.requires_grad)
            print('-->grad_value:', parms.grad)
        loss.backward()  # 反向传播，算出所有网络中的梯度grad
        optim.step()  # 对grad进行优化
        for name, parms in cif.named_parameters():
            print('更新')
            print('-->name:', name)
            print('-->para:', parms)
            print('-->grad_requirs:', parms.requires_grad)
            print('-->grad_value:', parms.grad)
        res_loss += loss
    print(res_loss)
