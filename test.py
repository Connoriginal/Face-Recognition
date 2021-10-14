import yalefaceDataset as yfd
import CNN 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# data Transform
trans = transforms.Compose([
                            transforms.Resize((200,200)), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])
                           ])

# get train data
test_data = yfd.YalefaceDataset(train=False, transform=trans)

test_loader = DataLoader(dataset=test_data,batch_size=len(test_data),shuffle=True)

model = torch.load("./model/CNN_model.pt")

correct = 0
total = 0
for i,(img,label) in enumerate(test_loader) :
    outputs = model(img)
    _,predicted = torch.max(outputs.data,1)
    total += label.size(0)
    correct += (predicted.cpu() == label).sum()
    print(predicted, label, correct, total)
    print("avg acc : %f"%(100*correct/total))