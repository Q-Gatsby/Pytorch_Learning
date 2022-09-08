import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=True, transform=img_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=True, transform=img_transforms, download=True)
# 这里的download尽量都设置为True，是因为可以看到下载的URL地址，如果没有URL地址，也可以通过Command单击数据集名称，向上翻找到URL

print(train_set[0])
img, target = train_set[0]
print(img)
writer = SummaryWriter(log_dir='torchvision_logs')    # 这里注意在使用tensorboard的时候注意把logdir改成对应的存储日志文件夹
# writer.add_image(tag='CIFAR10', img_tensor=img)
for i in range(10):
    writer.add_image(tag='CIFAR10', img_tensor=test_set[i][0], global_step=i)

writer.close()