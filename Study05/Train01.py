import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import CIFAR10_NN

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='/Users/HYQ/Documents/GitHub/Pytorch_Learning/Study03/torchvision_dataset',
    transform=torchvision.transforms.ToTensor(), train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='/Users/HYQ/Documents/GitHub/Pytorch_Learning/Study03/torchvision_dataset',
    transform=torchvision.transforms.ToTensor(), train=False, download=True)

# 输出数据集大小
print(f'训练集的数据大小为: {len(train_dataset)}')
print('测试集的数据大小为: {}'.format(len(test_dataset)))

# 利用DataLoader加载数据
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

# 创建网络模型
Model = CIFAR10_NN()
if torch.cuda.is_available():
    Model = Model.cuda()

# 利用损失函数计算损失
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 选择优化器
# 一般会选择把学习速率单独提出来，因为这样好修改，同时设置成为e为底的也更清晰，避免设置错误
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=Model.parameters(), lr=learning_rate)

# 设置训练网络中的一些参数
# 记录训练的次数
train_step = 0
# 记录测试的次数
test_step = 0
# 训练的轮数
epoch = 10

# 创建writer，将loss展示到tensorboard上
writer = SummaryWriter(log_dir='model_log')

for i in range(epoch):
    print(f'-------开始第 {i + 1} 次训练-------')

    # 训练步骤开始
    total_test_loss = 0
    total_accuracy = 0
    Model.train()  # 这一行可以注释掉，主要针对模型中存在dropout层等特殊层的时候使用
    for data in train_dataloader:
        # print(data) 这里如果不清楚输出的是何格式的数据，可以先尝试print，确认后再去编写，避免出错
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = Model(imgs)
        result_loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        result_loss.backward()  # 生成梯度
        optimizer.step()
        train_step += 1

        # 输出loss日志
        if train_step % 100 == 0:  # 加上这一步的if语句判断是因为如果不加，会有很多无效输出。
            print(f'训练次数: {train_step},  Loss: {result_loss.item()}')
            writer.add_scalar(tag='train_loss', scalar_value=result_loss.item(), global_step=train_step)

    # 测试模型步骤开始
    Model.eval()  # 这一行一般可以注释掉，主要针对模型中存在的dropout层等特殊结构的时候使用，但平时添加上也是可以的
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets.cuda()
            output = Model(imgs)
            # 计算Loss
            test_loss = loss_fn(output, targets)
            total_test_loss += test_loss.item()  # 这里对test_loss加上item()是因为test_loss是Tensor数据类型，加上item()后会变成数字
            # 计算准确率
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'模型整体在数据集上的Loss为： {total_test_loss}')
    print(f'模型整体在数据集上的准确率为： {total_accuracy / len(test_dataset)}')
    writer.add_scalar(tag='test_loss', scalar_value=total_test_loss, global_step=test_step)
    writer.add_scalar(tag='test_accuracy', scalar_value=total_accuracy / len(test_dataset), global_step=test_step)
    test_step += 1

    # 保存模型
    torch.save(Model, f'Model_{i}.pth')
    print('模型已保存，进行下一轮训练')

writer.close()
