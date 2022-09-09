import torch
import torch.nn.functional as F

# import numpy as np
#
# print(np.arange(5).reshape(-1))
# print(np.arange(5).reshape(-1,1))
# print(np.arange(5).reshape(1,-1))


input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])
print(input)
print(input.shape)
# 记得赋值
input = torch.reshape(input, [1, 1, 5, 5])
print(input.shape)

kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])
print(kernel)
print(kernel.shape)
kernel = torch.reshape(kernel, [1, 1, 3, 3])
print(kernel.shape)

output = F.conv2d(input=input, weight=kernel, stride=1)
print(output)
output = F.conv2d(input=input, weight=kernel, stride=2)
print(output)
output = F.conv2d(input=input, weight=kernel, stride=1, padding=1)
print(output)
