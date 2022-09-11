import torch
from torch import nn

input1 = torch.tensor([1, 5, 3], dtype=torch.float)
target1 = torch.tensor([3, 5, 2], dtype=torch.float)

MSE_mean = nn.MSELoss()
MSE_sum = nn.MSELoss(reduction='sum')
MSE_none = nn.MSELoss(reduction='none')
mse_mean = MSE_mean(input1, target1)
mse_sum = MSE_sum(input1, target1)
mse_none = MSE_none(input1, target1)
L1_Loss = nn.L1Loss()
l1_loss = L1_Loss(input1, target1) # 不论是L1Loss还是MSELoss，其对应的说明文档中，均没有对input的形状进行区分，因而不需要reshape

print(l1_loss)
print(mse_mean)
print(mse_sum)
print(mse_none)


x = torch.tensor([0.1, 0.2, 0.3])
print(x.shape)
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))   # 这里的reshape代表的是batch_size=1,然后channel=3；同时这里进行了reshape是因为CrossEntropyLoss对输入input有要求
loss_entropy = nn.CrossEntropyLoss()
print(loss_entropy(x, y))
