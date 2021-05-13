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
def _get_tomolearn_kwargs():
    return {
        'learning_rate': ,
        'num_steps': ,
        'display_step': ,
        'train_batch_size':  ,
        'test_batch_size':,
        'nf': 'vm2',
        'doc_path': '',
        'test_path':'',
        # 'predict_path':'D:/res_ture/predict/',
        # 'vol_size':,
        'weights_init': False,
        'shuffle':True,
        # 'lanmd':40,
        'phat_txt':'',
        'phatest_txt':'',
        'model_dir':''

    }
def make_shift(input_movs,shifts,vol_size):
    input_movs = input_movs.to('cpu').detach().numpy()
    input_movs = input_movs.reshape(len(input_movs),vol_size,vol_size)
    shifts = shifts.to('cpu').detach().numpy()
    a = np.zeros((len(input_movs), vol_size, vol_size))
    for i, (input_mov, shift) in enumerate(zip(input_movs, shifts)):
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        img_s1 = cv2.warpAffine(input_mov, M, (vol_size, vol_size), borderValue=0).astype(np.float32)
        a[i] = img_s1
    a = a.reshape(len(a),1,vol_size,vol_size)
    a_te = torch.tensor(a).type(torch.FloatTensor)



    return a_te
def train(**kwargs):
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step','train_batch_size', 'test_batch_size','nf',
                  'doc_path','test_path','vol_size', 'weights_init','shuffle','lanmd','phat_txt','phatest_txt','model_dir']
    kwargs_defaults = _get_tomolearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    device = "cuda"
    train__path = kwargs['doc_path']
    test__path = kwargs['test_path']
    model_dir = kwargs['model_dir']
    phat_txt = kwargs['phat_txt']
    phatest_txt = kwargs['phatest_txt']
    if not os.path.exists(phat_txt):
        os.makedirs(phat_txt)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(phatest_txt):
        os.makedirs(phatest_txt)
    train_set =DatasetFromFolder(train__path,train__path)
    training_data_loader = DataLoader(dataset=train_set, batch_size=kwargs['train_batch_size'], shuffle=kwargs['shuffle'])

    test_set = DatasetFromFolder(test__path, test__path)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=kwargs['test_batch_size'], shuffle=False)
    G = resnet18()
    G.to(device)
    optg = Adam(G.parameters(), lr=kwargs['learning_rate'])
    mse = nn.MSELoss(reduction='sum')
    mse.to(device)
    writer = SummaryWriter()
    i_test = 0



    for i in range(kwargs['num_steps']):
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
        if i % kwargs['display_step'] == 0 :
            avg_loss_mse_image = 0
            for input_fx, input_mov,target in training_data_loader:
                input_fx = input_fx.to(device)
                input_mov = input_mov.to(device)
                shift = G(input_fx, input_mov)

                f1 = open(phat_txt+'data_%s_eoch.txt' % i, 'a')
                f1.write('shift'+str(shift.data)+'\n'+'target'+str(target.data))
                f1.close()
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
                f2 = open(phatest_txt + 'data_%s_eoch.txt' % i, 'a')
                f2.write('shift' + str(shift.data) + '\n' + 'target' + str(target.data))
                f2.close()
            letest = np.ceil(len(testing_data_loader.dataset))*2
            avg_loss_mse_image = avg_loss_mse_image / letest
            print("test%d , test_losse =  %f" % (i_test, avg_loss_mse_image.item()))
            writer.add_scalar('tes_loss', avg_loss_mse_image, i_test)
            i_test += 1
    writer.close()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()