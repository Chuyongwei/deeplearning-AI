from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# python的用法 =
# totensor

# 绝对路径
# 
img_path = "hymenoptera_data\\train\\ants_image\\6240338_93729615ec.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trains = transforms.ToTensor()
tensor_img = tensor_trains(img)


writer.add_image("Tensor_img",tensor_img)

writer.close()
print(tensor_img)