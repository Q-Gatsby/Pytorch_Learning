from torch.utils.data import Dataset
from PIL import Image
import os


class Study02(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        path = os.listdir(self.path)
        path.sort()
        self.image_path = path

    def __getitem__(self, item):
        img_name = self.image_path[item]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


# study02 = Study02('dataset/train', 'ants')
# print(study02[0])
# print(study02.image_path)

root_dir = 'dataset/train'
label_dir_ants = 'ants'
label_dir_bees = 'bees'
study02_ants = Study02(root_dir, label_dir_ants)
study02_bees = Study02(root_dir, label_dir_bees)
print(study02_ants)
print(study02_bees)

train_dataset = study02_ants + study02_bees
print(len(train_dataset))
print(train_dataset[125])

img, label = train_dataset[230]
img.show()









