from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'dataset/train/ants/0013035.jpg'
img = Image.open(img_path)
print(img)

# 1.transforms如何使用？
# 利用transforms，选择其中一个class，这里选择的是ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)    #在括号里面按command+P可以看到需要输入的参数
print(tensor_img)

# 2.为什么我们需要tensor型数据
writer = SummaryWriter('logs')

writer.add_image(tag='tensor_img', img_tensor = tensor_img)

writer.close()
