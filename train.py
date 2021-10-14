import yalefaceDataset as yfd
import CNN 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import os


# data Transform
trans = transforms.Compose([
                            transforms.Resize((200,200)), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5])
                           ])

# get train data
train_data = yfd.YalefaceDataset(train=True, transform=trans)

train_loader = DataLoader(dataset=train_data,batch_size=10,shuffle=True)

model = CNN.CNN()

cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(30) :
    avg_loss = 0
    cnt = 0
    for i,(img,label) in enumerate(train_loader) :
        optimizer.zero_grad() # 배치마다 optimizer 초기화
        outputs = model(img)
        loss = cost(outputs,label)
        avg_loss += loss.data
        cnt +=1
        loss.backward()
        optimizer.step()
    print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
    scheduler.step(avg_loss)

if not os.path.isdir("./model") :
    os.mkdir("./model")

torch.save(model,"./model/CNN_model.pt")