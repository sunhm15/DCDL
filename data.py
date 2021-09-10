import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import csv, json
import cv2
import pdb

class FaceDataset_csv(Dataset):

    def __init__(self, data_path, meta_path, filepath_list, transform=None):
        meta_data = np.loadtxt(meta_path, dtype=str, delimiter=',')
        # print(meta_data)
        all_filepath = meta_data[1:,-1]
        total_labels = meta_data[1:,7].astype(int)
        total_labels_gender = (meta_data[1:, 5]=='M').astype(int)
        labels_race_str = meta_data[1:, 4]
        self.race_set = ['W', 'B']
        # self.race_set = ['W']
        self.images = []
        self.paths = []
        self.labels = []
        self.labels_race = []
        self.labels_gender = []
        for i, filepath in enumerate(all_filepath):
            basename = os.path.basename(filepath)
            datapath = os.path.join(data_path, basename)
            temp_path = os.path.join('Album2', basename)
            if i%10000==0:
                print(i,len(all_filepath))
            if datapath not in filepath_list: # or labels_race_str[i] not in self.race_set:
            	continue
            img = np.array(Image.open(datapath).convert('RGB'))
            self.images.append(img)
            self.paths.append(datapath)
            self.labels_race.append(self.race_set.index(labels_race_str[i]) if labels_race_str[i] in self.race_set else len(self.race_set))
            self.labels_gender.append(total_labels_gender[i])
            self.labels.append(total_labels[i])
        print(len(self.paths))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(self.labels.shape, len(self.labels_gender), np.max(self.labels_race), self.images.shape)
        self.labels -= 16
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):

        img = self.images[index]
        # img = self.images[index].astype(np.float32)
        label = self.labels[index]
        label_gender = self.labels_gender[index]
        label_race = self.labels_race[index]
        if self.transform:
            img = self.transform(img)

        sample = {'path': self.paths[index], 'image': img, 'label': label, 'label_gender': label_gender, 'label_race': label_race}
        return sample   

class FaceDataset_gpath(Dataset):

    def __init__(self, filepath_list, transform=None):
        
        self.images = []
        self.labels = []
        self.labels_gender = []
        for i, filepath in enumerate(filepath_list):
            if i%10000==0:
                print(i, len(filepath_list))
            basename = os.path.basename(filepath)
            self.labels.append(int(basename[-6:-4]))
            # self.images.append(img)
            self.images.append(filepath)
            self.labels_gender.append(int(basename[-7]=='M'))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(np.max(self.labels), np.min(self.labels), np.max(self.labels) - np.min(self.labels))
        self.labels_gender = np.array(self.labels_gender)
        # self.labels -= np.min(self.labels)
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        img = np.array(Image.open(img).convert('RGB'))
        label = self.labels[index]
        labels_gender = self.labels_gender[index]
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'label': label, 'label_gender': labels_gender}
        return sample        
