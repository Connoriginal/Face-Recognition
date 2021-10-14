import os
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

subject = str(input("input image path"))

img = Image.open(subject)

trans = transforms.Compose([
                            transforms.Resize((200,200)), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])
                           ])
img = trans(img)
img = img.unsqueeze(0)

model = torch.load("./model/CNN_model.pt")
output = model(img)
result = torch.max(output,1)

label_index = int(result.indices[0]) + 1

print("Predicted subject is : subject%d"%(label_index))