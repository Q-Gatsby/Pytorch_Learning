import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./torchvision_dataset', train=False, transform=torchvision.transforms.ToTensor())
# root (string): Root directory of dataset where directory
# datasets中具有getitem函数，该函数返回的就是img， target两个参数
loader_data = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
loader_data1 = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
loader_data2 = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
writer = SummaryWriter(log_dir='load_logs')

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(target)

# 初始版本 无shuffle，无drop_last
step = 0
for data in loader_data:
    img, target = data
    writer.add_images(tag='load_data', img_tensor=img, global_step=step)    # 多个图片要add_images而不能用add_image
    step += 1

# 无shuffle，有drop_last
step = 0
for data in loader_data2:
    img, target = data
    writer.add_images(tag='drop_data', img_tensor=img, global_step=step)    # 多个图片要add_images而不能用add_image
    step += 1

# 有shuffle
for i in range(2):
    step = 0
    for data in loader_data1:
        img, target = data
        writer.add_images(tag=f'{i}_data', img_tensor=img, global_step=step)    # 多个图片要add_images而不能用add_image
        step += 1

writer.close()

