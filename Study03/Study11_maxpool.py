import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

Tensor = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype= torch.float)    # 如果不对tensor增加dtype为float，则会出现RuntimeError: "max_pool2d" not implemented for 'Long'

Tensor = torch.reshape(Tensor, (1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train= True, transform= torchvision.transforms.ToTensor(), download=True)
load_data = DataLoader(dataset, batch_size=64)

class Maxpool(nn.Module):
    def __init__(self):
        super(Maxpool, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        output1 = self.maxpool1(x)
        output2 = self.maxpool2(x)
        return output1, output2

maxpool = Maxpool()
output1, output2 = maxpool(Tensor)
print(output1, '\n',output2)

writer = SummaryWriter(log_dir='maxpool_dir')

step = 0
for data in load_data:
    imgs, targets = data
    writer.add_images(tag='Original', img_tensor=imgs, global_step=step)
    imgs1, imgs2 = maxpool(imgs)
    writer.add_images(tag='After_True', img_tensor=imgs1, global_step=step)
    writer.add_images(tag='After_False', img_tensor=imgs2, global_step=step)
    step += 1

writer.close()