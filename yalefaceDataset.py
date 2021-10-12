import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset



class YalefaceDataset(Dataset):
    def __init__(self,train = True, transform=None,target_transform=None):
        if train :
            self.img_labels = pd.read_csv("./data/yale_train.csv")
        else :
            self.img_labels = pd.read_csv("./data/yale_test.csv")
        self.img_dir = "./data/clean_yalefaces/"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path =  os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx,3]
        if self.transform :
            image = self.transform(image)
        if self.target_transform :
            label = self.target_transform(label)
        
        return image, label

    # def get_label_dictionary() :
