from torch.utils.data import Dataset
from PIL import Image
import os


root_dir = "hymenoptera_data\\train"
label_dir = "ants"

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)


ass = MyData(root_dir,label_dir)
len = ass.__len__()
print(len)


img,label = ass[48]
img.show()
