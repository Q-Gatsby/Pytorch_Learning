# Pytorch_Learning
# tensorboard语句
tensorboard --logdir = logs
# 改变输出的host
tensorboard --logdir = logs --port = 6007
# 解决无法打开tensorboard的情况在Terminal输入下面语句
alias tensorboard='python3 -m tensorboard.main'
https://www.cnblogs.com/ccfco/p/15174831.html（三种解决方法）
# 成功打开tensorboard但并没有显示相应数据，是因为没有到对应的文件下
cd到与logs相同路径下的文件夹内
# 关于Channel的解释
Channel是加载图片参数：HWC中的C，H是图片的高度Height，W是宽度Width。
具体Channel概念解析在Notion里面

