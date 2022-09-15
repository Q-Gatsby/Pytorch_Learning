from PIL import Image
from Study05.model import CIFAR10_NN
import torch
import torchvision

model = torch.load('Model_29_GPU.pth', map_location=torch.device('cpu'))
image = Image.open('test3.png')
image = image.convert('RGB')    # 加上这一行是由于PNG格式的照片具有4个Channel，需要改成3个channel的RGB格式
print(image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    print(model)
    output = model(image)
print(torch.argmax(output, dim=1))
