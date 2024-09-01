from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "hymenoptera_data/train/ants_image/1262877379_64fcada201.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print("类型"+str(type(img_array)))

# writer.add_image()
# 添加图片
# 名称，图片，序号，格式
writer.add_image("test",img_array,2,dataformats="HWC")

# y=x
# 添加坐标 名称 f(y) x
for i in range(100):
    writer.add_scalar("y=9x",23*i,i)


writer.close()