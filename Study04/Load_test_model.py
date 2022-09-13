import torch
from Model import Test  # 如果没有import这一行，则模型无法导入，model行报错
from Model import trail

model = torch.load('test_model.pth')
print(model)
print(trail())
