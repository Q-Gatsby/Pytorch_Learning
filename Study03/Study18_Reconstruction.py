import torch
import torchvision
from torch import nn

vgg16_False = torchvision.models.vgg16(pretrained=False)
# vgg16_True = torchvision.models.vgg16(pretrained=True)    # 如果运行这一步会下载模型和其优化后的参数

print(vgg16_False)

# 添加一个新的线性层
vgg16_False.add_module(name='New_Linear', module=nn.Linear(in_features=1000, out_features=10))
print(vgg16_False)
# 添加一个新的Sequential
vgg16_False.add_module(name='New_Sequential', module=nn.Sequential(
    torch.nn.ReLU(),
    torch.nn.Sigmoid()
))
print(vgg16_False)
# # 在原来的一个Sequential中的添加新的一层
vgg16_False.classifier.add_module(name='New_layer', module=nn.Linear(in_features=1000, out_features=10))
print(vgg16_False)
# 修改原来Sequential中的某一层
vgg16_False.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_False)