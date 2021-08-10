import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import dxchange
import os
from shift_resnet import resnet18
import torch
import cv2
import torch.nn as nn
from res_data import DatasetFromFolder
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff"])
def red_stack_tiff(path):
    # path0 = 'D:/pycharm/pycharm/py/resig/data/shapp3d_160/'
    files = os.listdir(path)
    prj = []
    # prj0 = np.zeros((len(files), size, size))
    for n,file in enumerate(files):
        if is_image_file(file):
            p = dxchange.read_tiff(path + file)
            prj.append(p)
    pr = np.array(prj)
    return pr
def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min())/(img.max()-img.min())
    return img
def tonor_data(img):
    img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
    return img

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])
def get_args():
    parser = argparse.ArgumentParser(description='Test the Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test__path', type=str, default='/path/to/data/predict/')
    parser.add_argument('--model_dir', type=str, default='/path/to/data/test/')
    return parser.parse_args()
if __name__ == "__main__":
    opt = get_args()
    for indx in indxs:
        test__path = opt.test__path
        for batch_size in range(182):
            print('batch_size : ', batch_size)
            test_set = DatasetFromFolder(test__path, test__path)
            testing_data_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
            device = "cuda"
            model = resnet18()
            model.to(device)
            plt_loss_max = []
            plt_loss = []
            plt_mean_loss = []

            ph_model = opt.model_dir
            model.load_state_dict(torch.load(ph_model))
            shifts = torch.tensor([])
            mse = nn.MSELoss(reduction='sum')
            mse.to(device)
            avg_loss_mse_image = 0
            loss_shift = torch.tensor([])
            pred_shift = torch.tensor([])
            for input_fx, input_mov, target in testing_data_loader:
                loss_shift, pred_shift = loss_shift.to(device), pred_shift.to(device)
                input_fx, input_mov = tonor_data(input_fx), tonor_data(input_mov)
                input_fx = input_fx.to(device)
                input_mov = input_mov.to(device)
                target = target.to(device)
                shift = model(input_fx, input_mov)
                loss_mse_image = mse(shift, target)
                avg_loss_mse_image += loss_mse_image
                a = shift - target
                loss_shift = torch.cat([loss_shift, shift - target], dim=0)
                pred_shift = torch.cat([pred_shift, shift], dim=0)

            letest = np.ceil(len(testing_data_loader.dataset)) * 2.
            avg_loss_mse_image = avg_loss_mse_image / letest
            loss_shift = loss_shift.to('cpu').detach().numpy()
            pred_shift = pred_shift.to('cpu').detach().numpy()
            loss_shift_max = np.max(np.fabs(loss_shift))
            loss_shift_mean = np.mean(np.fabs(loss_shift))
            print('batch_size : ', batch_size,
                  "item = %s  avg_loss_mse_image =  %f  loss_max = %f , mean_loss = %f" % (
                      i, avg_loss_mse_image.item(), loss_shift_max, loss_shift_mean))
            plt_mean_loss.append(loss_shift_mean)
            plt_loss_max.append(loss_shift_max)
            plt_loss.append(avg_loss_mse_image.item())
        item_loss = np.argmin(plt_loss)
        item_max_loss = np.argmin(plt_loss_max)
        item_mean = np.argmin(plt_mean_loss)
        print('item', item_loss, 'item_loss_max', plt_loss_max[item_loss], 'item_mse', plt_loss[item_loss],
              'item_mean',
              plt_mean_loss[item_loss])




