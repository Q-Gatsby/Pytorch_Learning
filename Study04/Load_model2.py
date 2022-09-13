import torch
import torchvision

# vgg16 = torch.load('vgg16_method2.pth')
# print(vgg16)   # 此时单独去加载模型，出来的是字典格式的模型

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)
