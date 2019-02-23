from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import torch

class NkDataSet(Dataset):

    def __init__(self,csv_path):

        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path,header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):

        single_image_name = self.image_arr[index]

        print(single_image_name)

        img_as_img = cv2.imread(single_image_name)

        img_as_img = cv2.resize(img_as_img, (100,100))


        img_as_tensor = self.to_tensor(img_as_img)

        single_image_label = self.label_arr[index]

        return(img_as_tensor,single_image_label)

    def __len__(self):

        return self.data_len

csv_path = './file/jun.csv'

custom_dataset = NkDataSet(csv_path)

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=1,
                                               shuffle=False)

for i, (images, labels) in enumerate(my_dataset_loader):
    print(images,labels)
