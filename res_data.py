import torch.utils.data as data
import numpy as np
from os import listdir
import dxchange
import torch
from torch.utils.data import DataLoader
import os


def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min())/(img.max()-img.min())
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff"])

def is_csv_file(filename):
    return any(filename.endswith(extension) for extension in [".csv"])

def load_shu(filepath,h):
    # files = os.listdir(filepath)
    data_1 = np.loadtxt(filepath[0], delimiter=',')
    for i, file in enumerate(filepath[1:]):
        data_2 = np.loadtxt(file, delimiter=',')
        data_1 = np.vstack((data_1,data_2))
    data = torch.tensor(data_1).type(torch.FloatTensor)
    return data

def load_img(filepath):
    #1:p+s,2:p',3:p.
    img1 = dxchange.reader.read_tiff(filepath+ '_1.tiff')
    img1 = nor_data(img1)[np.newaxis,...]
    img2 = dxchange.reader.read_tiff(filepath + '_2.tiff')
    img2 = nor_data(img2)[np.newaxis,...]
    # img3 = dxchange.reader.read_tiff(filepath + '_3.tiff')
    # img3 = nor_data(img3)[np.newaxis, ...]

    img1 = torch.tensor(img1).type(torch.FloatTensor)
    img2 = torch.tensor(img2).type(torch.FloatTensor)
    # img3 = torch.tensor(img3).type(torch.FloatTensor)
    # img = torch.cat([img1,img2], dim=3)
    # img = torch.unsqueeze(img, 0)/ 255.
    return img1,img2

class DatasetFromFolder(data.Dataset):
    def __init__(self, input_image_dir, target_image_dir):
        super(DatasetFromFolder, self).__init__()
        self.input_image_filenames = []
        self.target_image_filenames = []
        for name in listdir(input_image_dir):
            self.tu = list(set(x.split('_')[0] for x in listdir(input_image_dir + name) if is_image_file(x)))
            self.tu.sort()
            self.inputlen = len(self.tu)
            self.input_image_filenames += [os.path.join(input_image_dir +name,x) for x in self.tu]

        for name in listdir(target_image_dir):

            shu = list(set(x.split('_')[0] for x in os.listdir(input_image_dir + name) if is_csv_file(x)))
            self.target_image_filenames += [os.path.join(target_image_dir + name,x) for x in shu]
        self.target_labels = load_shu(self.target_image_filenames,self.inputlen)

    def __getitem__(self, index):
        # a=self.target_image_filenames
        input1,input3= load_img(self.input_image_filenames[index])
        target = self.target_labels[index]

        return input1,input3,target

    def __len__(self):
        return len(self.input_image_filenames)



