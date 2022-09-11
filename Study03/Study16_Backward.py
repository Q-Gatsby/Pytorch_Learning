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
    print(result_loss)
    result_loss.backward()
    # print('OK')