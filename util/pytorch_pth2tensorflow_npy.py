'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
'''

import torch
import numpy as np
import os
import urllib.request
import sys
 
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# for item in model_urls:
#     a =  model_urls[item].split('/')[-1]
#     print("'"+item+"':'/home/test01/sambashare/sdh/resnet_pytorch_pth/"+a+"',")


pth_paths = {
    'resnet18':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnet18-5c106cde.pth',
    'resnet34':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnet34-333f7ec4.pth',
    'resnet50':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnet50-19c8e357.pth',
    'resnet101':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnet101-5d3b4d8f.pth',
    'resnet152':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnet152-b121ed2d.pth',
    'resnext50_32x4d':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':'/home/test01/sambashare/sdh/resnet_pytorch_pth/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':'/home/test01/sambashare/sdh/resnet_pytorch_pth/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2':'/home/test01/sambashare/sdh/resnet_pytorch_pth/wide_resnet101_2-32ee1156.pth'
}




def convert(model_name='resnet18',pth_path=None, save_path=None):
    if pth_path==None:
        pth_path = pth_paths[model_name]

    if os.path.exists(pth_path)==False:
        print('.pth is not existed. Please download the .pth. The commands down here can help you download related .pth files(For Linux Users):\n')
        for item in model_urls:
            a =  model_urls[item].split('/')[-1]
            #eg: wget -O /home/test01/sambashare/sdh/resnet_pytorch_pth/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth
            print("wget -O /home/test01/sambashare/sdh/resnet_pytorch_pth/"+a+' '+model_urls[item]+'&')
        
        print('\nOr you can retype the pth_pth:')
        pth_path = input()  
    if os.path.exists(pth_path)==False:
        print('Not Found. Please download it at first. See you~')
        sys.exit(0)

    dictionary = {}
    for var_name, value in torch.load(pth_path).items():
        numpy_value = value.data.numpy().astype(np.float32)
        # print(type(value))

        if 'downsample.1' in var_name or 'bn' in var_name:
            var_name = var_name.replace('.running_mean', '/moving_mean').replace('.running_var', '/moving_variance')
            var_name = var_name.replace('.weight', '/gamma').replace('.bias', '/beta')

        if len(numpy_value.shape)==4:
            numpy_value = np.transpose(numpy_value, [2,3,1,0])

        if var_name=='fc.weight':
            numpy_value = numpy_value.T

        dictionary[var_name] = numpy_value

    if save_path!=None:
        np.save(model_name+'.npy', dictionary)

    return dictionary




'''test code'''
# d = convert('resnet50')
# for item in d:
#     print(item, d[item].shape)



