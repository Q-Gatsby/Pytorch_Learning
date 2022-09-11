import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class CIF(nn.Module):
    def __init__(self):
        super(CIF, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten1 = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)    # 这里如果不清楚最后的in_features是多少，可以print Flatten之后的shape
        # self.linear2 = nn.Linear(in_features=64, out_features=10)

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten1(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

cif = CIF()
input = torch.ones((64, 3, 32, 32))   # 64代表batch_size，意味着每一个channel里面有64个需要计算
output = cif(input)
print(output.shape)
writer = SummaryWriter(log_dir='Seq_logs')
writer.add_graph(model=cif, input_to_model=input)
writer.close()
