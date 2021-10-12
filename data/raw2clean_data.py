import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder

data_path = "./yalefaces"
save_path = "./clean_yalefaces"

# Make directory
if not os.path.isdir(save_path) :
    os.mkdir(save_path)

# Dataframe for the csv file
df = pd.DataFrame(columns=["image name", "subject", "feature"])

i = 0
for file_name in os.listdir(data_path) :
    if (not "gif" in file_name) and (not "txt" in file_name) and (not "DS_Store" in file_name) :
        image = Image.open(data_path+'/'+file_name)
        # label files
        file_name = file_name.split('.')
        subject, feature = file_name[0], file_name[1]
        img_name = subject+"_"+feature+".jpg"
        img_path = save_path + '/' + img_name

        # Add to Dataframe
        df_temp = pd.DataFrame(data=[[img_name,subject,feature]],columns=["image name", "subject", "feature"],index = [i])
        
        df = pd.concat([df,df_temp])
        i += 1
        # Save Image
        if (not os.path.isfile(img_path)) :
            image.save(img_path)

# Save csv file
## add encoded label column Using LabelEncoder()
label_list = df.loc[:,["subject"]]
le = LabelEncoder()
le.fit(label_list)
df["encoded label"] = le.transform(label_list)

## divide in to 8:2 ratio
train_data = df.sample(frac=0.8)
test_data = df.drop(train_data.index)

train_data.to_csv("./yale_train.csv",index=False)
test_data.to_csv("./yale_test.csv",index=False)




