import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

img_path1 = "./data/clean_yalefaces/subject0"
img_path2 = "./data/clean_yalefaces/subject"

def main():
    subject = str(input("input image path\n"))

    img = Image.open(subject)
    origin_img = np.array(img)

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

    if(label_index < 10) :
        path = img_path1
    else :
        path = img_path2

    p_img = Image.open(path + str(label_index)+"_normal.jpg")
    predict_img = np.array(p_img)

    # plot
    fig = plt.figure()

    fst = fig.add_subplot(1,2,1)
    fst.imshow(origin_img,cmap = plt.get_cmap("gray"))
    fst.set_title("Input Subject")
    fst.axis("off")

    snd = fig.add_subplot(1,2,2)
    snd.imshow(predict_img,cmap=plt.get_cmap("gray"))
    snd.set_title("Predicted Subject" + str(label_index))
    snd.axis("off")

    plt.show()




if __name__ == '__main__':
    main()