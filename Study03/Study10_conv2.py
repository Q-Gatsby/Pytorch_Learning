import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

torch_data = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(), download=True)
writer = SummaryWriter(log_dir='conv2_logs')
load_data = DataLoader(dataset=torch_data, batch_size=64)  # batch_size如果属于一个数相当于默认取x*x；自定义可以batch_size=(10,3)这种


class Trail(nn.Module):
    def __init__(self):
        super(Trail, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


trail = Trail()
# print(trail)
# Trail(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))  stride代表的是H和W，第一个是高度，第二个是宽度
#   同理这里的padding也可以是（4，2）类型的，代表的是高度上下各拓宽4，左右宽度各拓宽2
#   dilation可以取（3，2）这种，代表的是高度方向3个取一个，宽度方向2个取一个
# )
step = 0
for data in load_data:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images(tag='Before_NN', img_tensor=imgs, global_step=step)
    imgs = trail(imgs)
    imgs1 = torch.reshape(imgs, (-1, 3, 30, 30))
    # imgs2 = torch.reshape(imgs, (-1, 3, 32, 32)) 这里-1代表batch_size随着后面的channel3，height30和width30进行调整，不需要自己计算好，类似于numpy的
    print(imgs1.shape)
    writer.add_images(tag='After_NN1', img_tensor=imgs1, global_step=step)
    # writer.add_images(tag='After_NN2', img_tensor=imgs1, global_step=step)
    step += 1

writer.close()