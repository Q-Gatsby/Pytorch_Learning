import torchvision
from torch import nn
from torch.utils.data import DataLoader

datasets = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=False, transform= torchvision.transforms.ToTensor(), download=True)
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

loss = nn.CrossEntropyLoss()
cif = CIF()
for data in load_data:
    imgs, targets = data
    output = cif(imgs)
    # print(imgs)
    # print(targets)
    # print(output.shape)
    result_loss = loss(output, targets)
    # 这里之所以能够利用loss函数，也可以输出的out_features不是10个，如果改成11个，20个，50个都可以正常运行，但参与计算的只有前面10个数，
    # 同时由于out_features变多了，预测正确的概率变低了，因而到CrossEntropyLoss计算的时候，log后的数值会负的更多。
    print(result_loss)
    result_loss.backward()
    # print('OK')