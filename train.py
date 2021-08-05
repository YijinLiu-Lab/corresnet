import os
import warnings
from argparse import ArgumentParser
import numpy as np
import torch
from torch.optim import Adam
import dxchange
from model import resnet18
from torch.utils.tensorboard import SummaryWriter
import cv2
import torch.nn as nn
from torch.nn import functional as F
from res_data import DatasetFromFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"
def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min())/(img.max()-img.min())
    return img
def tonor_data(img):
    img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
    return img

def get_args():
    parser = argparse.ArgumentParser(description='Train the Net ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataroot', type=str, default='/path/to/data/train/')
    parser.add_argument('--test_dataroot', type=str, default='/path/to/data/test/')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=44444)
    parser.add_argument('--display_step', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--model_dir', type=str, default='/path/to/your/model/')
    parser.add_argument('--weights_init', type=str, default='False')
    parser.add_argument('--shuffle', type=str, default='True')
    return parser.parse_args()

def train():
    opt = get_args()
    device = "cuda"
    train__path = opt.train_dataroot
    test__path = opt.test_dataroot
    model_dir = opt.model_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_set =DatasetFromFolder(train__path,train__path)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.train_batch_size, shuffle=opt.shuffle)

    test_set = DatasetFromFolder(test__path, test__path)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)
    G = resnet18()
    G.to(device)
    optg = Adam(G.parameters(), lr=opt.learning_rate)
    mse = nn.MSELoss(reduction='sum')
    mse.to(device)
    writer = SummaryWriter()
    i_test = 0

    for i in range(opt.num_steps):
        avg_epoch_Gloss = 0
        for input_fx,input_mov, target in tqdm(training_data_loader):
            input_fx , input_mov = tonor_data(input_fx),tonor_data(input_mov)
            input_fx = input_fx.to(device)
            input_mov = input_mov.to(device)
            target = target.to(device)
            optg.zero_grad()
            shift = G(input_fx, input_mov)
            loss_shift = mse(shift, target)
            G_loss = loss_shift
            G_loss.backward()
            optg.step()
            avg_epoch_Gloss += G_loss
        le = np.ceil(len(training_data_loader.dataset))*2
        avg_epoch_Gloss = avg_epoch_Gloss / le
        writer.add_scalar('train_loss_fn', avg_epoch_Gloss, i)
        print('n_iter%d,train_loss = %f'% (i, avg_epoch_Gloss.item()))
        if i % opt.display_step == 0 :
            avg_loss_mse_image = 0
            for input_fx, input_mov,target in training_data_loader:
                input_fx = input_fx.to(device)
                input_mov = input_mov.to(device)
                shift = G(input_fx, input_mov)
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(G.state_dict(), save_file_name)
            for input_fx, input_mov, target in testing_data_loader:
                input_fx, input_mov = tonor_data(input_fx), tonor_data(input_mov),
                input_fx = input_fx.to(device)
                input_mov = input_mov.to(device)
                target = target.to(device)
                shift = G(input_fx, input_mov)

                loss_mse_image = mse(shift, target)
                avg_loss_mse_image += loss_mse_image
            letest = np.ceil(len(testing_data_loader.dataset))*2
            avg_loss_mse_image = avg_loss_mse_image / letest
            print("test%d , test_losse =  %f" % (i_test, avg_loss_mse_image.item()))
            i_test += 1
    writer.close()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()