from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img = Image.open("dataset/train/ants_image/36439863_0bec9f554f.jpg")
print(img)
writer = SummaryWriter('logs')

# ToTensor
trans_tensor = transforms.ToTensor()  # 通过Command单击transforms，发现ToTensor是一个类
# 通过实例化这个类，并且通过直接调用这个函数，运行了它的__call__函数，返回一个Tensor型
img_tensor = trans_tensor(img)
writer.add_image("img_tensor", img_tensor)

# Normalize
print(img_tensor[0, 0, 0])
trans_norm = transforms.Normalize(mean=[2, 3, 6], std=[4, 1, 2])
# 这里之所以是[0.5,0.5,0.5]，是由于mean参数是一个sequence，同时这个序列中的个数，是由设置的通道数决定的，如果由n的通道，则需要n个元素
# 计算公式： ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
img_norm = trans_norm(img_tensor)
print(img_norm[0, 0, 0])
writer.add_image('Norm', img_norm,global_step=3)

# Resize
trans_resize = transforms.Resize(size=(300, 300))
# 从PIL img resize到 PIL resize img
img_resize = trans_resize(img)
# 从PIL resize img ToTensor转化为 Tensor型
img_resize = trans_tensor(img_resize)
writer.add_image('img_resize1', img_resize, global_step=1)

# Resize2
trans_resize2 = transforms.Resize(300)
trans_compose = transforms.Compose([trans_resize2, trans_tensor])
img_resize2 = trans_compose(img)    # 即从img出发，先resize，再ToTensor
writer.add_image('img_resize1', img_resize2, global_step=2)

# RandomCrop
trans_random = transforms.RandomCrop(size=(100, 100))
crop_compose = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    writer.add_image('random_crop', crop_compose(img), global_step=i)

writer.close()