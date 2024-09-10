from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img_path = "hymenoptera_data\\train\\ants_image\\6240338_93729615ec.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
tensor_trains = transforms.ToTensor()
img_tensor = tensor_trains(img)

writer.add_image("ToTensor",img_tensor)


print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalization",img_norm)
writer.close()
# print(tensor_img)