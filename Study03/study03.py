from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

writer = SummaryWriter(log_dir='logs')
img_path = 'dataset/train/ants/28847243_e79fe052cd.jpg'
img = Image.open(img_path)
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)

# 通过opencv加载图片到tensorboard
#第一步是加载图片，第二步是把BGR换成RGB即R和B互换
# cv = cv2.imread(filename=img_path)
# cv = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)

# writer.add_image(tag='test', img_tensor=cv, global_step=2, dataformats='HWC')
writer.add_image(tag='test', img_tensor=img_array, global_step=3, dataformats='HWC')      #这里的H是高度Height，宽度是Width，C的意思是Channel，通道
#这里改变step就是把一个tag下分成了各个子项目，可以通过拖动tensorboard中的滑块看每个step下的图片，step3就代表是第三个步骤，即第三个滑块对应的内容。
#可以通过设置从step1到stepn分别看从1到n的样本的变化，也可以通过改变tag名称观察变化

writer.add_image(tag='test1', img_tensor=img_array, global_step=1, dataformats='HWC')

# for i in range(100):
#     writer.add_scalar('y=2x', i * 2, i)
#
writer.close()