import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random
import copy

import logging

from config import vit_config, mamba_config

if __name__ == '__main__':

    if len(sys.argv) > 1 \
        and sys.argv[3] in ['attention_block', 'ssm_forward']:
        arg_method = sys.argv[1] # method: MI
        arg_gpu = sys.argv[2] # GPU
        arg_setting = sys.argv[3] # setting: attention_block, ssm_forward
    else:
        raise Exception('1st arg: method [MI],\n 2nd arg: GPU,\n 3rd arg: setting [attention_block, ssm_forward]')

    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'log/log_{arg_setting}_{arg_method}.txt',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'MI_Cal on GPU:{arg_gpu}')

    os.environ['CUDA_VISIBLE_DEVICES'] = arg_gpu

    
    def reset_seed():
        seed = 2024
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    reset_seed()

    logging.info(f'init seed: {torch.initial_seed()}')
    
    # vit
    file_path_attention_block = vit_config.file_path_attention_block
    # mamba
    file_path_ssm_forward = mamba_config.file_path_ssm_forward
    print('<<<Reading Data>>>')
    # vit
    tensor_attention_block = torch.load(file_path_attention_block)
    # mamba
    tensor_ssm_forward = torch.load(file_path_ssm_forward)
    print('<<<Data Read>>>')
    
    MI_setting={
        'attention_block':{
            'num_dim':384 + 384,
            'num_layer':12,
            'num_token':197,
            'input':tensor_attention_block[0,:,:,:,:],
            'output':tensor_attention_block[1,:,:,:,:],
        },
        'ssm_forward': {
            'num_dim': 768 + 768,
            'num_layer': 24,
            'num_token': 197,
            'input': tensor_ssm_forward[0,:,:,:,:],
            'output': tensor_ssm_forward[1,:,:,:,:],
        },
    }

    MI_results = {}
    MI_nonlog_results = {}

    batch_size = 256
    num_iter = 1000
    decay = 0.9

    output_root_data='./data/'
    output_root_img='./img/'

    # num_dim_ssm = 768
    # idx_layer_ssm = 8

    # num_dim_vit = 384
    # idx_layer_vit = 9


    class T_model_Deep(nn.Module):
        def __init__(self, num_dim=768, range=[-1000,1000]):
            super(T_model_Deep, self).__init__()
            self.fc1 = nn.Linear(num_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
            self.output = nn.Linear(256, 1)

            self.dropout = nn.Dropout(p=0.5)
            self.batchnorm1 = nn.BatchNorm1d(1024)
            self.batchnorm2 = nn.BatchNorm1d(512)
            self.batchnorm3 = nn.BatchNorm1d(256)

            self.range = range

        def forward(self, x, y):
            x = torch.cat((x, y), dim=1)
            
            x = F.relu(self.batchnorm1(self.fc1(x)))
            x = self.dropout(x)
            
            x = F.relu(self.batchnorm2(self.fc2(x)))
            x = self.dropout(x)
            
            x = F.relu(self.batchnorm3(self.fc3(x)))
            x = self.dropout(x)
            
            x = self.output(x)

            # x = F.sigmoid(self.output(x))*(max(self.range)-min(self.range))+min(self.range)


            return x

    def MI(X, Y, T_m):
        T = T_m(X, Y)
        rand_idx = torch.randperm(Y.size(0))
        T_bar = T_m(X, Y[rand_idx])
        return T.mean() - torch.log(torch.mean(torch.exp(T_bar))) # changed loss function
    # Func. Cal. MI

    def Calculate_MI(X, Y):
        X, Y = X.cuda(), Y.cuda()
        num_dim = X.shape[-1] + Y.shape[-1]
        T_m = T_model_Deep(num_dim=num_dim, range=[-20,20]).cuda()
        T_m_ema = copy.deepcopy(T_m)
        for param in T_m_ema.parameters():
            param.requires_grad = False

        T_m.train()
        T_m_ema.train()

        optimizer = optim.AdamW(T_m.parameters(), lr=0.0001)

        T_max = 0
        T_max_ema = 0
        for epoch in range(num_iter):
            sampled_indices = random.sample(range(len(X)), batch_size)
            _X, _Y = X[sampled_indices], Y[sampled_indices]
            loss_function = -MI(_X, _Y, T_m)

            loss = loss_function.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                    for ema_param, param in zip(T_m_ema.parameters(), T_m.parameters()):
                        ema_param.copy_(ema_param * decay + param.data * (1 - decay))
            
            # print(f'Epoch {epoch+1}, T:{T_m_ema(X, Y).mean().item()}, Loss: {loss.item()}')
            _T_max = torch.abs(T_m(X, Y)).max()
            T_max = _T_max if _T_max > T_max else T_max
            _T_max_ema = torch.abs(T_m_ema(X, Y)).max()
            T_max_ema = _T_max_ema if _T_max_ema > T_max_ema else T_max_ema
        _MI = 0
        _MI_ema = 0

        with torch.no_grad():
            T_m.eval()
            T_m_ema.eval()
            for i in range(1000):
                _MI+=MI(X, Y, T_m).mean() / 1000
                _MI_ema+=MI(X, Y, T_m_ema).mean() / 1000
            return _MI, _MI_ema, T_max, T_max_ema

        raise NotImplementedError
    
    # Cal. MI by layer

    #
    # inputs:
    #   idx_layer: given layer
    #   setting: key: num_token (per-layer), input, output ([layer, data_batch, time, token])
    # outputs:
    #   

    def Calculate_MI_layer(idx_layer, setting):
        MI_Xt_Yt = []
        for t in range(setting['num_token']):
            X, Y = setting['input'][idx_layer, :, t, :], setting['output'][idx_layer, :, t, :]
            MI, MI_ema, _T_max, _T_max_ema = Calculate_MI(X, Y)
            logging.info(f"t: {t}, MI: {MI},  MI_ema: {MI_ema},\
                        _T_max: {_T_max}, _T_max_ema: {_T_max_ema}")
            MI_Xt_Yt.append((MI, MI_ema))
        return torch.Tensor(MI_Xt_Yt).transpose(0, 1)

    # Calcualte Layer-wise MI between inputs and outputs.

    if arg_method == 'MI':
        print('MI calculating start')
        for key_setting in MI_setting.keys():
            if arg_setting == key_setting:
                setting = MI_setting[key_setting] # key: num_token (per-layer), input, output
            else:
                continue

            print(f'MI calculating: {key_setting}')
            reset_seed()
            MI_layer = []
            for idx_layer in range(setting['num_layer']):
                _MI_layer = Calculate_MI_layer(idx_layer, setting) # [normal/ema, time]
                logging.info(f'Setting: {key_setting}, Layer: {idx_layer},\n \
                            normal: mean: {torch.mean(_MI_layer[0])}, std: {torch.std(_MI_layer[0])}, max: {torch.max(_MI_layer[0])}, median: {torch.median(_MI_layer[0])}, min: {torch.min(_MI_layer[0])},\n \
                            ema: mean: {torch.mean(_MI_layer[1])}, std: {torch.std(_MI_layer[1])}, max: {torch.max(_MI_layer[1])}, median: {torch.median(_MI_layer[1])}, min: {torch.min(_MI_layer[1])}, \
                    ')
                MI_layer.append(_MI_layer)
                torch.save(_MI_layer, output_root_data+f'MI_{key_setting}_{idx_layer}.pt')
            MI_layer_tensor = torch.cat(MI_layer) # [layer, normal/ema, time]
            torch.save(MI_layer_tensor, output_root_data+f'MI_{key_setting}.pt')
            MI_results[key_setting] = MI_layer_tensor