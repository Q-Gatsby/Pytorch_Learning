import torchvision

train_dataset = torchvision.datasets.CIFAR10(root='/Users/HYQ/Documents/GitHub/Pytorch_Learning/Study03/torchvision_dataset',
                                             train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='/Users/HYQ/Documents/GitHub/Pytorch_Learning/Study03/torchvision_dataset',
                                             train=False, download=True)
