import torch
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)
# 方式1：保存模型+模型参数
torch.save(vgg16, 'vgg16_method1.pth')

# 方式2：官方推荐，保存的是字典格式的
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')