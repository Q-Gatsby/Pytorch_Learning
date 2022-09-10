import torch
import torchvision
from torch import nn

# input = torch.tensor([
#     [1, -1, -3],
#     [3, 5, -3],
#     [11, 22, 3]
# ])
#
# input = torch.reshape(input, (-1, 1, 3, 3))
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ReLu(nn.Module):
    def __init__(self):
        super(ReLu, self).__init__()
        self.Relu1 = nn.Sigmoid()    # 这里写ReLu()的区别不大，因为一般的channel值都为正

    def forward(self, input):
        output = self.Relu1(input)
        return output

# relu = ReLu()
# output = relu(input)
# print(output)
# tensor([[[[ 1,  0,  0],
#           [ 3,  5,  0],
#           [11, 22,  3]]]])

datasets = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=False, transform=torchvision.transforms.ToTensor(),download=True)
load_data = DataLoader(datasets, batch_size=36)


writer = SummaryWriter(log_dir='ReLu_logs')
relu = ReLu()
step = 0
for data in load_data:
    imgs, targets = data
    writer.add_images(tag='Before', img_tensor=imgs, global_step=step)
    imgs = relu(imgs)
    writer.add_images(tag='After', img_tensor=imgs, global_step=step)
    step += 1

writer.close()

