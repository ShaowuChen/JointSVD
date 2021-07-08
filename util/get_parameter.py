'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
'''

import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
import numpy as np 
import copy
import sys
import time
flags = tf.flags
FLAGS=flags.FLAGS



cifar10_Path = {
    'resnet18': '/home/test01/sambashare/sdh/CoupledSVD/ckpt/orig/cifar10/resnet18/resnet18_top1_0.9479999989271164-299',
    'resnet34': '/home/test01/sambashare/sdd/Coupled_Pruning/cifar10/orig_ckpt/resnet34_std0.01_l2/resnet34_top1_0.9510999977588653-299',
    'resnet50': None,
    'resnet101': '/home/test01/sambashare/sdh/CoupledSVD/ckpt/orig/cifar10/resnet101/resnet101_top1_0.9496000051498413-299',
    'resnet152': None
}

cifar100_Path = {
    'resnet18': '/home/test01/sambashare/sdh/CoupledSVD/ckpt/orig/cifar100/resnet18/resnet18_top1_0.7072999954223633-299',
    'resnet34': '/home/test01/sambashare/sdh/CoupledSVD/ckpt/orig/cifar100/resnet34/resnet34_top1_0.7580999940633774-299',
    'resnet50': '/home/test01/sambashare/sdh/CoupledSVD/ckpt/orig/cifar100/resnet50/resnet50_top1_0.756700000166893-299',
    'resnet101': None,
    'resnet152': None
}

imagenet_Path = {
    'resnet18': None,
    'resnet34':'/home/test01/sambashare/sdd/Coupled_Pruning/imagenet/orig_ckpt/resnet34/resnet34_top1_0.7102599966526032-5',
    'resnet50': None,
    'resnet101': None,
    'resnet152': None
}
Path = {'cifar10':cifar10_Path, 'cifar100':cifar100_Path, 'imagenet':imagenet_Path}

#include layer1
expansion = 4

orig_Repeat_list = {
    'resnet18': [2,2,2,2],
    'resnet34': [3,4,6,3],
    'resnet50': [3,4,6,3],
    'resnet101': [3,4,23,3],
    'resnet152': [3,8,36,3]
}
orig_Conv_list = {
    'resnet18': [1,2],
    'resnet34': [1,2],
    'resnet50': [1,2,3],
    'resnet101': [1,2,3],
    'resnet152': [1,2,3]
}

#exclude layer1
expansion = 4

Repeat_list = {
    'resnet18': [2,2,2],
    'resnet34': [4,6,3],
    'resnet50': [4,6,3],
    'resnet101': [4,23,3],
    'resnet152': [8,36,3]
}
#BasicBlock: decompose both conv1 and conv2, i.e., the 3x3 kernels 
#BottleBlock: only decompose conv2, i.e., the 3x3 kernel
Conv_list = {
    'resnet18': [1,2],
    'resnet34': [1,2],
    'resnet50': [2],
    'resnet101': [2],
    'resnet152': [2]
}





'''
============================================================
                    API
============================================================
'''

def get_parameter():

    assert(FLAGS.method in ['SVD', 'TT', 'NC_CTD', 'PCSVD', 'rJSVD_1',  'rJSVD_2', 'lJSVD', 'Bi_JSVD'])

    if FLAGS.model in ['resnet18', 'resnet34']:
        conv_list = [1,2]
    else:
        assert(FLAGS.method!='TT') #Resnet50, Resnet101 contains lots of 1*1, TT is not appropriate
        conv_list = [1,2,3]

    path = Path[FLAGS.dataset][FLAGS.model]
    layer_list = [2,3,4]
    repeat_list = Repeat_list[FLAGS.model]
    layers_out_channels = [128,256,512] 

    npy_dict = ckpt2npy(path)  
    function = FLAGS.method+'_Parameter'
    decomposed_dict, record_dict = eval(function)(copy.deepcopy(npy_dict), layer_list, repeat_list, conv_list, layers_out_channels, eval(FLAGS.rank_rate_SVD))

    weight_dict = {**npy_dict, **decomposed_dict}

    return weight_dict, record_dict



def ckpt2npy(path):
    
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var = reader.get_variable_to_shape_map()
    npy_dict = {}

    for key in var:
        # print(key)
        value = reader.get_tensor(key)
        npy_dict[key] = value

    return npy_dict.copy()



'''
============================================================
                    SVD
============================================================

'''
def SVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):

    size_SVD_decompose = 0
    SVD_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat_th in range(repeat_list[i]):
            for conv in conv_list:

                weight_name = 'layer'+str(layer_name)+'.'+str(repeat_th)+'.conv'+str(conv)+'.weight'
                # print(weight_name)
                weight = npy_dict[weight_name] 
                U, V = decompose_SVD_4d(copy.deepcopy(weight), rank_rate_SVD, '4d')
                size_SVD_decompose += U.size + V.size
                SVD_dict[weight_name.replace('weight', 'U')] = U
                SVD_dict[weight_name.replace('weight', 'V')] = V
                # print('\n')
    CR = num_orig_all() / (num_undecomposed_parameter() + size_SVD_decompose)
    return SVD_dict, {'num_decomposed':size_SVD_decompose, 'num_undecomposed':num_undecomposed_parameter(), 'CR':CR}


def decompose_SVD_4d(weight, rank_rate_SVD, shape_option:'2d/4d/mix'='4d'):
    [F1, F2, I, O] = weight.shape  # weight[F1,F2,I,O]
    # print(weight.shape)
    #maximize rank_rate_SVD==0.5
    rank=int(O*rank_rate_SVD) if F1==F2==3 else int(min(I,O)*rank_rate_SVD)
    # print('rank',rank)

    W = np.reshape(np.transpose(weight,(0,2,1,3)), [F1*I, -1])
    # print(W.shape)
    U_2d, S_2d, V_2d = np.linalg.svd(W, full_matrices=True)
    # print('U_2d.shape',U_2d.shape)
    U_2d = np.dot(U_2d[:, 0:rank].copy(),np.diag(S_2d)[0:rank, 0:rank].copy())
    # print('rank',rank)
    
    # print('np.diag(S_2d)[0:rank, 0:rank].shape',np.diag(S_2d)[0:rank, 0:rank].shape)
    U_4d = np.transpose(np.reshape(U_2d, [F1, I, 1, rank]),(0,2,1,3))

    V_2d = V_2d[0:rank, :].copy()
    V_4d = np.transpose(np.reshape(V_2d,[1,rank,F2,O]),(0,2,1,3))

    if shape_option=='4d':
        return U_4d, V_4d
    elif shape_option=='2d':
        return U_2d, V_2d
    elif shape_option=='mix':
        return U_2d, V_4d
    else:
        assert(0)

def decompose_SVD_2d(weight_2d, rank):

    U_2d, S_2d, V_2d = np.linalg.svd(weight_2d, full_matrices=True)

    U_2d = np.dot(U_2d[:, 0:rank].copy(),np.diag(S_2d)[0:rank, 0:rank].copy())
    V_2d = V_2d[0:rank, :].copy()

    return U_2d, V_2d

def chagne_2D_to_4d(weight_2d, shape, rank, U_or_V=None):
    [F1, F2, I, O] = shape
    if U_or_V=='U':
        weight_4d = np.transpose(np.reshape(weight_2d, [F1, I, 1, rank]),(0,2,1,3))
    elif U_or_V=='V':
        weight_4d = np.transpose(np.reshape(weight_2d, [1,rank,F2,O]),(0,2,1,3))
    else:
        assert(0)
    return weight_4d



'''
============================================================
                    TT
============================================================
'''
def TT_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):

    rank_rate_TT = calculate_tt_rate(npy_dict,rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    num_tt = 0
    TT_dict = {}
    for i, layer_name in enumerate(layer_list):
        for repeat_th in range(repeat_list[i]):
            for conv in conv_list:

                weight_name = 'layer'+str(layer_name)+'.'+str(repeat_th)+'.conv'+str(conv)+'.weight'
                weight = npy_dict[weight_name] 

                G1, G2, G3 = decompose_TT(copy.deepcopy(weight), rank_rate_TT)

                TT_dict[weight_name.replace('weight', '1')] = G1
                TT_dict[weight_name.replace('weight', '2')] = G2
                TT_dict[weight_name.replace('weight', '3')] = G3

                num_tt += G1.size +G2.size + G3.size

    # num_tt = num_TT(rank_rate_TT, layer_list, repeat_list, conv_list, layers_out_channels)
    CR = num_orig_all() / (num_undecomposed_parameter() + num_tt)
    return TT_dict, {'num_decomposed':num_tt, 'num_undecomposed':num_undecomposed_parameter(), 'CR':CR}

def decompose_TT(value, rank_rate_TT, round_or_int=int):
    shape = value.shape
    (h,w,i,o) = shape

    assert (len(shape) == 4)
    assert (o >= i)

    rank_i=round_or_int(rank_rate_TT*i)
    rank_o=round_or_int(rank_rate_TT*o)


    # [H,W,I,O]
    value_2d = np.reshape(value, [h * w * i, o])
    U2, S2, V2 = np.linalg.svd(value_2d)

    # max rank: o
    V2 = np.matmul(np.diag(S2)[:rank_o, :rank_o], V2[:rank_o, :])
    U2_cut = U2[:, :rank_o]

    # to [i,h,w,rank2] and then [i, hw*rank2]
    U2_cut = np.transpose(np.reshape(U2_cut, [h, w, i, rank_o]), [2, 0, 1, 3])
    U2_cut = np.reshape(U2_cut, [i, h * w * rank_o])

    U1, S1, V1 = np.linalg.svd(U2_cut)
    # max rank: i
    assert(rank_i<=len(S1))
    V1 = np.matmul(np.diag(S1)[:rank_i, :rank_i], V1[:rank_i, :])
    U1_cut = U1[:, :rank_i]

    G3 = np.reshape(V2, [1, 1, rank_o, o])
    G2 = np.transpose(np.reshape(V1, [rank_i, h, w, rank_o]), [1, 2, 0, 3])
    G1 = np.reshape(U1_cut, [1, 1, i, rank_i])

    return G1,G2,G3


def calculate_tt_rate(npy_dict,rank_rate_svd, layer_list, repeat_list, conv_list, layers_out_channels):
    #r_1 is based on 'I'; r_2 is based on 'O'
    #i.e., r_1 = int(I*rank_rate_tt)
    #r_2 = int(O*rank_rate_tt)

    #rr=rank_rate_tt which is need to solve from function.
    # num_tt = sum( 1*1*i * (i * rr)  + 9 * (i*rr) * (o*rr) + 1*1*o*(o*rr)=sum(9*i*o) * (rr^2) + sum((i*i+o*o))*rr) = num_svd
    #solve function: a = sum(9*i*o), b = sum((i*i+o*o)), c=-num_svd
    # i_or_o = 1

    a = 0
    b = 0    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        
        for i in range(len(layers_out_channels)): #I==O
            a += (9*layers_out_channels[i]*layers_out_channels[i])*(2*repeat_list[i]-1)#sum of 3*3*I*O
            #each BasicBlock contains 2 3x3conv
            b += (layers_out_channels[i]*layers_out_channels[i] + layers_out_channels[i]*layers_out_channels[i])*(2*repeat_list[i]-1)#sum of I*I+O*O

        #parameters of decomposed conv1 of repeat0     
        # if FLAGS.decom_conv1==True:#I=O/2
        for i in range(len(layers_out_channels)):
            a += 9*layers_out_channels[i]/2*layers_out_channels[i]#3*3*I*O
            b += layers_out_channels[i]/2*layers_out_channels[i]/2 + layers_out_channels[i]*layers_out_channels[i]#I*I+O*O

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        assert(0)
        for i in range(len(layers_out_channels)):#I==O
            #each one contains 1 3x3conv
            a += (9*layers_out_channels[i]*layers_out_channels[i])*repeat_list[i]
            b += (layers_out_channels[i]*layers_out_channels[i] + layers_out_channels[i]*layers_out_channels[i])*repeat_list[i]

    _, dic = SVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_svd)
    num_svd = dic['num_decomposed']
    c = -num_svd

    delta=b*b-4*a*c
    rank_rate_tt1 = (-b + np.sqrt(delta))/(2*a)
    rank_rate_tt2 = (-b - np.sqrt(delta))/(2*a)
    if delta<0 or rank_rate_tt1<=0:
        print('no')
        sys.exit(0)
    else:
        return rank_rate_tt1

#num of parameters in the TT based decomposed layers under rank_rate_tt
def num_TT(rank_rate_tt, layer_list, repeat_list, conv_list, layers_out_channels):
    #tt resnet34 分解层总参数量
    if rank_rate_tt<=0:
        assert(0)

    num = 0
    # don't decompose layer1; conv1 of repeat0 is up to FLAGS.decom_conv1
    if FLAGS.model in ['resnet18', 'resnet34']:
        #parameters of decomposed conv except conv1 of repeat0     
        for i in range(len(repeat_list)):
            num += (layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt)*2 +\
                   9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]*rank_rate_tt)) * (2*repeat_list[i]-1)#each BasicBlock contains 2 3x3conv

        #parameters of decomposed conv1 of repeat0     
        # if FLAGS.decom_conv1==True:
        for i in range(len(repeat_list)):
            num += layers_out_channels[i]/2 * int(layers_out_channels[i]/2*rank_rate_tt)+\
                   layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt) +\
                   9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]/2*rank_rate_tt)

    #decompose all layers; only decompose conv2, i.e., 3x3 kernels
    else: 
        assert(0)
        for i in range(len(repeat_list)):
            num += (layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt)+\
                   layers_out_channels[i] * int(layers_out_channels[i]*rank_rate_tt) +\
                   9*int(layers_out_channels[i]*rank_rate_tt)*int(layers_out_channels[i]*rank_rate_tt))*repeat_list[i]#each one contains 1 3x3conv

    return num


'''
===============================================================================================================
            NC_CTD (calculate rank_i for each con$i$_x seperately instead for the whole network)
===============================================================================================================
HID layers would be compressed by navie TT.

formula to caluculate rank_rate_indep, rank_rate_shared:

Given rank_rate_svd=r_svd, 
get rank_rate_tt (for navie tt);
Given numbers of convolution weights=num, rank_indep_over_rank_shared=n,  rank_rate_independent=r_i, rank_rate_shared=r_s,

for Resnet18, Resnet34
    for conv$i$_x:
        r_i * num + r_s = (n*num+1) r_s = rank_rate_tt * num
        then:
        r_s = num/(n*num+1) * rank_rate_tt
        r_i = n*num/(n*num+1) * rank_rate_tt
 '''

def NC_CTD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):
    # print('\n', rank_rate_SVD,'\n')

    rank_rate_TT = calculate_tt_rate(npy_dict,rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)

    num_parameter_count=0
    parameter_dict = {}
    rank_rate_shared_list = []
    rank_rate_independent_list = []


    '============================Navie TT============================================='

    #layerx.repeat0.conv1 (HID layers) would be compressed by navie TT
    #Note that in ResNet18, layer+str(layer_name)+'.1.conv1.weight' is alone,
    #Since the Repeat_list is [2,2,2,2];Thus layer+str(layer_name)+'.1.conv1.weight' should be compressed by navie TT as well
    if FLAGS.model=='resnet18':
        naive_tt_repeat_list = ['0','1']
    else:
        naive_tt_repeat_list = ['0']
    for i, layer_name in enumerate(layer_list):
        for naive_repeat in naive_tt_repeat_list:
            weight_name = 'layer'+str(layer_name)+'.'+naive_repeat+'.conv1.weight'
            weight = npy_dict[weight_name] 
            G1, G2, G3 = decompose_TT(copy.deepcopy(weight), rank_rate_TT, round_or_int=int)

            parameter_dict[weight_name.replace('weight', '1')] = G1
            parameter_dict[weight_name.replace('weight', '2')] = G2
            parameter_dict[weight_name.replace('weight', '3')] = G3
            num_parameter_count += G1.size + G2.size + G3.size


    '==========================NC-CTD==============================================='

    Iter_times = FLAGS.Iter_times if FLAGS.from_scratch==False else 1

    for i, layer_name in enumerate(layer_list):
        for j, conv in enumerate(conv_list):
            if FLAGS.model=='resnet18' and str(conv)=='1':
                continue
            else:
                if str(conv)=='1':
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
                else:
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

                # name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]
                part_of_shared_name = 'layer'+str(layer_name)+'.conv'+str(conv)
                parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]

                num = len(name_list)
                n = eval(FLAGS.rank_indep_over_rank_shared)
                rank_rate_shared = num/(n*num+1) * rank_rate_TT
                rank_rate_independent = n*num/(n*num+1) * rank_rate_TT

                rank_rate_independent_list.append(rank_rate_independent)
                rank_rate_shared_list.append(rank_rate_shared)

                shared_component = 0 #Initialized
                for iter_time in range(Iter_times): #Iteration
                    sub_network_independent_dict, independent_component_list, size_independent = Independent_component_CTD(npy_dict, name_list, shared_component, rank_rate_independent)
                    sub_network_shared_dict, shared_component, size_shared = shared_component_CTD(npy_dict, parameter_list, independent_component_list, part_of_shared_name, rank_rate_shared)

                    if iter_time==0:
                        num_parameter_count += size_independent + size_shared

                #end of interation
                assert(parameter_dict.keys()&sub_network_independent_dict.keys()&sub_network_shared_dict.keys()==set())
                parameter_dict = {**parameter_dict, **sub_network_independent_dict, **sub_network_shared_dict}



    CR = num_orig_all() / (num_undecomposed_parameter() + num_parameter_count)
    record_dict = {'num_decomposed':num_parameter_count,
                    'num_undecomposed':num_undecomposed_parameter(),
                   'rank_rate_shared_list':rank_rate_shared_list,
                   'rank_rate_independent_list':rank_rate_independent_list,
                   'rank_rate_independent/rank_rate_shared': FLAGS.rank_indep_over_rank_shared,
                   'CR':CR }

    return parameter_dict, record_dict


def Independent_component_CTD(npy_dict, name_list, shared_component, rank_rate_independent):
    independent_dict = {}
    independent_component_list = []
    size_independent = 0

    for item in name_list:
        weight_name = item + '.weight'
        I1,I2,I3 = decompose_TT( npy_dict[weight_name]-shared_component, rank_rate_independent, round_or_int=round)
        independent_dict[item+'.I1'] = I1
        independent_dict[item+'.I2'] = I2
        independent_dict[item+'.I3'] = I3
        independent_component_list.append(recover_TT(I1,I2,I3))
        size_independent += I1.size + I2.size + I3.size

    return  independent_dict, independent_component_list, size_independent


def shared_component_CTD(npy_dict, parameter_list, independent_component_list, part_of_shared_name, rank_rate_shared):
    shared_dict = {}
    shared_recover_dict = {}
    size_shared = 0

    assert(len(parameter_list)==len(independent_component_list))
    shared_component = sum([parameter_list[i]-independent_component_list[i] for i in range(len(parameter_list))])/len(parameter_list)
    S1,S2,S3 = decompose_TT(shared_component, rank_rate_shared, round_or_int=round)
    shared_dict[part_of_shared_name+'.S1'] = S1
    shared_dict[part_of_shared_name+'.S2'] = S2
    shared_dict[part_of_shared_name+'.S3'] = S3
    size_shared += S1.size + S2.size + S3.size

    return shared_dict, shared_component, size_shared


def recover_TT(G1, G2, G3):
    assert(G2.shape[0]==G2.shape[1]==3)
    H = 3
    W = 3
    I = G1.shape[2]
    O = G3.shape[3]
    r_input = G1.shape[3]
    r_output = G3.shape[2]

    G12 = np.matmul(np.squeeze(G1), np.reshape(np.transpose(G2,[2,0,1,3]), [r_input, H*W*r_output]))  #(I, HWr_output)
    G12 = np.reshape(np.transpose(np.reshape(G12, [I,H,W,r_output]), [1,2,0,3]), [H*W*I, r_output])  #(HWI, r_output)
    G123 = np.reshape(np.matmul(G12, np.squeeze(G3)), [H,W,I,O])

    return G123



'''
============================================================
                    rJSVD_1  (RJSVD-1 in the paper)
============================================================
formula to caluculate rank_rate_JSVD:

Give rank_rate_svd=r_svd, numbers of convolution weights=num, 

for Resnet18, Resnet34
    conv1 ( one (and only one) of the weight have half of input dimension:)
        (2*num-1/2)r_svd = (num+1/2)r_jsvd  
        then: 
        rank_rate_jsvd = (2*num-1/2)/(num+1/2) * r_svd

    conv2 (all the weights are of the same dimensions:) 
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num U, and one identical or shared V, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

for Resnet50, Resnet101
    conv1( 1*1*(o*2)*o for repeat0, 1*1*(o*4)*o for other repeat_i where 4 is the expansion factor):
        o*2*r_svd+r_svd*o + (o*4*r+r*o)*r_svd*(num-1)==(5*num-2)o*r_svd = o*2*r_jsvd+o*4*r_jsvd*(num-1) + r_jsvd*o==(4num-1)o*r_jsvd
        then
        r_jsvd = (5*num-2)/(4*num-1)*r_svd 

    conv2(3*3*o*o):
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num U, and one identical or shared V, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

    conv3( 1*1*o*(o*4) where 4 is the expansion factor):
        [o*r_svd+r_svd*o*4]*num==(5*num)*o*r_svd = o*r_jsvd*num + r_jsvd*o*4==(4+num)*o*r_jsvd
        then
        r_jsvd = 5*num/(4+num)*r_svd 
 '''


def rJSVD_1_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):
    assert(len(layer_list)==len(repeat_list)==len(layers_out_channels))
    # rank_rate_JSVD = calculate_JSVD_rate(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    num_parameter_count=0
    JSVD_dict = {}
    JSVD_matmul_dict = {}
    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:

            # if str(conv)=='1' and FLAGS.decom_conv1==False:
            #     name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
            # else:
            name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]            

            num = len(parameter_list)
            if FLAGS.model in ['resnet18', 'resnet34']:
                if conv==1:
                    rank_rate_JSVD = (2*num-1/2)/(num+1/2) * rank_rate_SVD
                elif conv==2:
                    rank_rate_JSVD = 2*num/(num+1) * rank_rate_SVD
                else:
                    assert(0)
            elif FLAGS.model in ['resnet50', 'resnet101']:
                if conv==1:
                    rank_rate_JSVD = (5*num-2)/(4*num-1) * rank_rate_SVD 
                elif conv==2:
                    rank_rate_JSVD =  2*num/(num+1) * rank_rate_SVD
                elif conv==3:
                    rank_rate_JSVD = 5*num/(4+num) * rank_rate_SVD 
                else:
                    assert(0)


            #rank = rank_rate_JSVD * O
            (H, W, I, O) = parameter_list[-1].shape
            rank_JSVD=int(O*rank_rate_JSVD) if H==W==3 else int(min(I,O)*rank_rate_JSVD)

            parameter_stacked = np.vstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(parameter_list)])            
            U_2d_stacked, V_share_2d = decompose_SVD_2d(parameter_stacked, rank_JSVD)
            num_parameter_count += U_2d_stacked.size + V_share_2d.size

            V_share_4d = chagne_2D_to_4d(V_share_2d, parameter_list[-1].shape, rank_JSVD, 'V')
            JSVD_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_share_4d

            
            width = H*I
            for j, item in enumerate(name_list):
                # if str(conv)=='1' and FLAGS.decom_conv1==True:
                if str(conv)=='1':
                    if j==0:
                        start = 0 
                        end = int(width/2)
                    else:
                        start = int(width/2) + width * (j-1)
                        end = int(width/2) + width * j
                else:
                    start = width*j
                    end =  width*(j+1)

                JSVD_dict[item+'.U_partly_shared'] = chagne_2D_to_4d(U_2d_stacked[start:end].copy(), parameter_list[j].shape, rank_JSVD, 'U')


    CR = num_orig_all() / (num_undecomposed_parameter() + num_parameter_count)
    # print('num_orig_all',num_orig_all())
    # print('num_decomposed',num_parameter_count)
    # print('num_undecomposed',num_undecomposed_parameter())
    # print('CR',CR)
    # input()

    return JSVD_dict, {'num_decomposed':num_parameter_count,'num_undecomposed':num_undecomposed_parameter(),'rank_rate_calculate':rank_rate_JSVD, 'CR':CR}



'''
============================================================
                    rJSVD_2  (RJSVD-2 in the paper)
============================================================
formula to caluculate rank_rate_JSVD:
!!conv1 with 1/2 size of input dimension will not be jointly decomposed but svd!!

Give rank_rate_svd=r_svd, numbers of convolution weights=num, 

for Resnet18, Resnet34
    conv1/conv2 (all the weights are of the same dimensions:) 
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num U, and one identical or shared V, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

for Resnet50, Resnet101
    conv1( 1*1*(o*2)*o for repeat0, 1*1*(o*4)*o for other repeat_i where 4 is the expansion factor):
        (o*4+o)*r_svd*(num)==(5*num)*o*r_svd = o*4*r_jsvd*(num) + r_jsvd*o==(4num+1)*o*r_jsvd
        then
        r_jsvd = (5*num)/(4*num+1)*r_svd 

    conv2(3*3*o*o):
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num U, and one identical or shared V, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

    conv3( 1*1*o*(o*4) where 4 is the expansion factor):
        [o*r_svd+r_svd*o*4]*num==(5*num)*o*r_svd = o*r_jsvd*num + r_jsvd*o*4==(4+num)*o*r_jsvd
        then
        r_jsvd = 5*num/(4+num)*r_svd 
 '''


def rJSVD_2_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):
    assert(len(layer_list)==len(repeat_list)==len(layers_out_channels))
    # rank_rate_JSVD = calculate_JSVD_rate(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    num_parameter_count=0
    JSVD_dict = {}
    JSVD_matmul_dict = {}

    if FLAGS.model=='resnet18':
        naive_tt_repeat_list = ['0','1']
    else:
        naive_tt_repeat_list = ['0']
    for i, layer_name in enumerate(layer_list):
        for naive_repeat in naive_tt_repeat_list:
            weight_name = 'layer'+str(layer_name)+'.'+naive_repeat+'.conv1.weight'
            weight = npy_dict[weight_name] 
            U, V = decompose_SVD_4d(copy.deepcopy(weight), rank_rate_SVD, '4d')
            num_parameter_count += U.size + V.size
            JSVD_dict[weight_name.replace('weight', 'U')] = U
            JSVD_dict[weight_name.replace('weight', 'V')] = V


    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:

            if FLAGS.model=='resnet18' and str(conv)=='1':
                continue
            else:
                if str(conv)=='1':
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
                else:
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]            

            num = len(parameter_list)
            if FLAGS.model in ['resnet18', 'resnet34']:
                # if conv==1:
                #     rank_rate_JSVD = (2*num-1/2)/(num+1/2) * rank_rate_SVD
                # elif conv==2:
                rank_rate_JSVD = 2*num/(num+1) * rank_rate_SVD
                # else:
                #     assert(0)
            elif FLAGS.model in ['resnet50', 'resnet101']:
                if conv==1:
                    rank_rate_JSVD = (5*num)/(4*num+1) * rank_rate_SVD 
                elif conv==2:
                    rank_rate_JSVD =  2*num/(num+1) * rank_rate_SVD
                elif conv==3:
                    rank_rate_JSVD = 5*num/(4+num) * rank_rate_SVD 
                else:
                    assert(0)


            #rank = rank_rate_JSVD * O
            (H, W, I, O) = parameter_list[-1].shape
            rank_JSVD=int(O*rank_rate_JSVD) if H==W==3 else int(min(I,O)*rank_rate_JSVD)

            parameter_stacked = np.vstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(parameter_list)])            
            U_2d_stacked, V_share_2d = decompose_SVD_2d(parameter_stacked, rank_JSVD)
            num_parameter_count += U_2d_stacked.size + V_share_2d.size

            V_share_4d = chagne_2D_to_4d(V_share_2d, parameter_list[-1].shape, rank_JSVD, 'V')
            JSVD_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_share_4d

            
            width = H*I
            for j, item in enumerate(name_list):
                # if str(conv)=='1' and FLAGS.decom_conv1==True:
                # if str(conv)=='1':
                #     if j==0:
                #         start = 0 
                #         end = int(width/2)
                #     else:
                #         start = int(width/2) + width * (j-1)
                #         end = int(width/2) + width * j
                # else:
                start = width*j
                end =  width*(j+1)

                JSVD_dict[item+'.U_partly_shared'] = chagne_2D_to_4d(U_2d_stacked[start:end].copy(), parameter_list[j].shape, rank_JSVD, 'U')


    CR = num_orig_all() / (num_undecomposed_parameter() + num_parameter_count)
    return JSVD_dict, {'num_decomposed':num_parameter_count,'num_undecomposed':num_undecomposed_parameter(),'rank_rate_calculate':rank_rate_JSVD, 'CR':CR}



'''
============================================================
                    lJSVD  (LJSVD)
============================================================
formula to caluculate rank_rate_JSVD:
!!conv1 with 1/2 size of input dimension will not be jointly decomposed but svd!!

Give rank_rate_svd=r_svd, numbers of convolution weights=num, 

for Resnet18, Resnet34
    conv1/conv2 (all the weights are of the same dimensions:) 
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num V, and one identical or shared U, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

for Resnet50, Resnet101
    conv1( 1*1*(o*2)*o for repeat0, 1*1*(o*4)*o for other repeat_i where 4 is the expansion factor):
        (o*4+o)*r_svd*(num)==(5*num)*o*r_svd = o*4*r_jsvd + r_jsvd*o*(num)==(4+num)*o*r_jsvd
        then
        r_jsvd = (5*num)/(4+num)*r_svd 

    conv2(3*3*o*o):
        (2*num)r_svd = (num+1)r_jsvd  (for example, in r_jsvd, there are num U, and one identical or shared V, which make num+1 )
        then: 
        rank_rate_jsvd = 2*num/(num+1) * r_svd

    conv3( 1*1*o*(o*4) where 4 is the expansion factor):
        [o*r_svd+r_svd*o*4]*num==(5*num)*o*r_svd = o*r_jsvd + r_jsvd*o*4*num ==(4*num+1)*o*r_jsvd
        then
        r_jsvd = (5*num)/(4*num+1)*r_svd 
 '''


def lJSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):
    assert(len(layer_list)==len(repeat_list)==len(layers_out_channels))
    # rank_rate_JSVD = calculate_JSVD_rate(rank_rate_SVD, layer_list, repeat_list, conv_list, layers_out_channels)
    num_parameter_count=0
    JSVD_dict = {}
    JSVD_matmul_dict = {}

    if FLAGS.model=='resnet18':
        naive_tt_repeat_list = ['0','1']
    else:
        naive_tt_repeat_list = ['0']
    for i, layer_name in enumerate(layer_list):
        for naive_repeat in naive_tt_repeat_list:
            weight_name = 'layer'+str(layer_name)+'.'+naive_repeat+'.conv1.weight'
            weight = npy_dict[weight_name] 
            U, V = decompose_SVD_4d(copy.deepcopy(weight), rank_rate_SVD, '4d')
            num_parameter_count += U.size + V.size
            JSVD_dict[weight_name.replace('weight', 'U')] = U
            JSVD_dict[weight_name.replace('weight', 'V')] = V


    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:

            if FLAGS.model=='resnet18' and str(conv)=='1':
                continue
            else:
                if str(conv)=='1':
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
                else:
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]

            parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]            

            num = len(parameter_list)
            if FLAGS.model in ['resnet18', 'resnet34']:
                # if conv==1:
                #     rank_rate_JSVD = (2*num-1/2)/(num+1/2) * rank_rate_SVD
                # elif conv==2:
                rank_rate_JSVD = 2*num/(num+1) * rank_rate_SVD
                # else:
                #     assert(0)
            elif FLAGS.model in ['resnet50', 'resnet101']:
                if conv==1:
                    rank_rate_JSVD = 5*num/(4+num) * rank_rate_SVD 
                elif conv==2:
                    rank_rate_JSVD =  2*num/(num+1) * rank_rate_SVD
                elif conv==3:
                    rank_rate_JSVD = (5*num)/(4*num+1) * rank_rate_SVD 
                else:
                    assert(0)


            #rank = rank_rate_JSVD * O
            (H, W, I, O) = parameter_list[-1].shape
            rank_JSVD=int(O*rank_rate_JSVD) if H==W==3 else int(min(I,O)*rank_rate_JSVD)

            #hstack
            parameter_stacked = np.hstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(parameter_list)])      
            U_share_2d, V_2d_stacked = decompose_SVD_2d(parameter_stacked, rank_JSVD)

            num_parameter_count += U_share_2d.size + V_2d_stacked.size

            U_share_4d = chagne_2D_to_4d(U_share_2d, parameter_list[-1].shape, rank_JSVD, 'U')
            JSVD_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.U_shared'] = U_share_4d

            
            width = W*O
            for j, item in enumerate(name_list):
                # if str(conv)=='1' and FLAGS.decom_conv1==True:
                # if str(conv)=='1':
                #     if j==0:
                #         start = 0 
                #         end = int(width/2)
                #     else:
                #         start = int(width/2) + width * (j-1)
                #         end = int(width/2) + width * j
                # else:
                start = width*j
                end =  width*(j+1)

                JSVD_dict[item+'.V_partly_shared'] = chagne_2D_to_4d(V_2d_stacked[:,start:end].copy(), parameter_list[j].shape, rank_JSVD, 'V')


    CR = num_orig_all() / (num_undecomposed_parameter() + num_parameter_count)
    return JSVD_dict, {'num_decomposed':num_parameter_count,'num_undecomposed':num_undecomposed_parameter(),'rank_rate_calculate':rank_rate_JSVD, 'CR':CR}


'''
============================================================
                    Bi-JSVD
============================================================
'''

def Bi_JSVD_Parameter(npy_dict, layer_list, repeat_list, conv_list, layers_out_channels, rank_rate_SVD, *useless_vars):

    num_parameter_count=0
    Bi_JSVD_dict = {}
    half_rank_JSVD_list = []


    if FLAGS.model=='resnet18':
        naive_tt_repeat_list = ['0','1']
    else:
        naive_tt_repeat_list = ['0']
    for i, layer_name in enumerate(layer_list):
        for naive_repeat in naive_tt_repeat_list:
            weight_name = 'layer'+str(layer_name)+'.'+naive_repeat+'.conv1.weight'
            weight = npy_dict[weight_name] 
            U, V = decompose_SVD_4d(copy.deepcopy(weight), rank_rate_SVD, '4d')
            num_parameter_count += U.size + V.size
            Bi_JSVD_dict[weight_name.replace('weight', 'U')] = U
            Bi_JSVD_dict[weight_name.replace('weight', 'V')] = V

    for i, layer_name in enumerate(layer_list):
        for conv in conv_list:

            if FLAGS.model=='resnet18' and str(conv)=='1':
                continue
            else:
                if str(conv)=='1':
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
                else:
                    name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]
            # if str(conv)=='1':
            #     name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i]) if not repeat==0]
            # else:
            #     name_list = [ 'layer'+str(layer_name)+'.'+str(repeat)+'.conv'+str(conv) for repeat in range(repeat_list[i])]
            
            parameter_list = [ copy.deepcopy(npy_dict[name +'.weight']) for name in name_list]            

            num = len(parameter_list)
            if FLAGS.model in ['resnet18', 'resnet34']:

                rank_rate_JSVD = 2*num/(num+1) * rank_rate_SVD
                rank_rate_lJSVD = 2*num/(num+1) * rank_rate_SVD * eval(FLAGS.ratio_lJSVD)
                rank_rate_rJSVD =  2*num/(num+1) * rank_rate_SVD * (1-eval(FLAGS.ratio_lJSVD))

            elif FLAGS.model in ['resnet50', 'resnet101']:
                if conv==1:
                    rank_rate_lJSVD = 5*num/(4+num) * rank_rate_SVD*eval(FLAGS.ratio_lJSVD)
                    rank_rate_rJSVD = (5*num)/(4*num+1) * rank_rate_SVD*(1-eval(FLAGS.ratio_lJSVD))
                elif conv==2:
                    rank_rate_lJSVD =  2*num/(num+1) * rank_rate_SVD*eval(FLAGS.ratio_lJSVD)
                    rank_rate_rJSVD =  2*num/(num+1) * rank_rate_SVD*(1-eval(FLAGS.ratio_lJSVD))
                elif conv==3:
                    rank_rate_lJSVD = (5*num)/(4*num+1) * rank_rate_SVD*eval(FLAGS.ratio_lJSVD)
                    rank_rate_rJSVD = 5*num/(4+num) * rank_rate_SVD*(1-eval(FLAGS.ratio_lJSVD))
                else:
                    assert(0)

            (H, W, I, O) = parameter_list[-1].shape
            rank_lJSVD = int(O*rank_rate_lJSVD) if H==W==3 else int(min(I,O)*rank_rate_lJSVD) 
            rank_rJSVD = int(O*rank_rate_rJSVD) if H==W==3 else int(min(I,O)*rank_rate_rJSVD) 

            #Iteratively
            Iter_times = FLAGS.Iter_times if FLAGS.from_scratch==False else 1
            for j in range(Iter_times):
                if j==0:            
                    rJSVD_matmul_list = 0 #Initialized
                lJSVD_matmul_list, lJSVD_dict, num_parameter_count_lJSVD = lJSVD_component(parameter_list, name_list, layer_name, conv, rJSVD_matmul_list, rank_lJSVD)
                rJSVD_matmul_list, rJSVD_dict, num_parameter_count_rJSVD = rJSVD_component(parameter_list, name_list, layer_name, conv, lJSVD_matmul_list, rank_rJSVD)

            num_parameter_count += num_parameter_count_lJSVD + num_parameter_count_rJSVD
            assert(lJSVD_dict.keys()&rJSVD_dict.keys()==set())
            Bi_JSVD_dict = {**Bi_JSVD_dict, **lJSVD_dict, **rJSVD_dict}

    CR = num_orig_all() / (num_undecomposed_parameter() + num_parameter_count)

    return Bi_JSVD_dict, {'num_decomposed':num_parameter_count,'num_undecomposed':num_undecomposed_parameter(),'half_rank_JSVD_list':half_rank_JSVD_list, 'CR':CR}


def lJSVD_component(parameter_list, name_list, layer_name, conv, rJSVD_matmul_list, rank_lJSVD, *useless_vars):

    if rJSVD_matmul_list==0:
        res_parameter_list = parameter_list
    else:
        res_parameter_list = [parameter_list[index]-rJSVD_matmul_list[index] for index in range(len(parameter_list))]

    lJSVD_dict = {}
    #hstack
    parameter_stacked = np.hstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(res_parameter_list)])      
    U_share_2d, V_2d_stacked = decompose_SVD_2d(parameter_stacked, rank_lJSVD)
    num_parameter_count_lJSVD = U_share_2d.size + V_2d_stacked.size

    U_share_4d = chagne_2D_to_4d(U_share_2d, parameter_list[-1].shape, rank_lJSVD, 'U')
    lJSVD_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.U_shared'] = U_share_4d

    lJSVD_matmul_list = []
    (H, W, I, O) = parameter_list[-1].shape   
    width = W*O
    for j, item in enumerate(name_list):

        start = width*j
        end =  width*(j+1)

        lJSVD_dict[item+'.V_partly_shared'] = chagne_2D_to_4d(V_2d_stacked[:,start:end].copy(), parameter_list[j].shape, rank_lJSVD, 'V')

        lJSVD_matmul_list.append(np.transpose(np.reshape(np.dot(U_share_2d,V_2d_stacked[:,start:end]), [H,I,W,O]), [0,2,1,3])) 


    return lJSVD_matmul_list, lJSVD_dict, num_parameter_count_lJSVD

def rJSVD_component(parameter_list, name_list, layer_name, conv, lJSVD_matmul_list, rank_rJSVD, *useless_vars):
    if lJSVD_matmul_list==0:
        res_parameter_list = parameter_list
    else:
        res_parameter_list = [parameter_list[index]-lJSVD_matmul_list[index] for index in range(len(parameter_list))]

    rJSVD_dict = {}
    #vstack
    parameter_stacked = np.vstack([np.reshape(np.transpose(item, [0,2,1,3]), (item.shape[0]*item.shape[2],-1)) for item in copy.deepcopy(res_parameter_list)])      
    U_2d_stacked, V_share_2d = decompose_SVD_2d(parameter_stacked, rank_rJSVD)
    num_parameter_count_rJSVD = U_2d_stacked.size + V_share_2d.size

    V_share_4d = chagne_2D_to_4d(V_share_2d, parameter_list[-1].shape, rank_rJSVD, 'V')
    rJSVD_dict['layer'+str(layer_name)+'.conv'+str(conv)+'.V_shared'] = V_share_4d

    rJSVD_matmul_list = []
    (H, W, I, O) = parameter_list[-1].shape   
    width = H*I
    for j, item in enumerate(name_list):

        start = width*j
        end =  width*(j+1)

        rJSVD_dict[item+'.U_partly_shared'] = chagne_2D_to_4d(U_2d_stacked[start:end,:].copy(), parameter_list[j].shape, rank_rJSVD, 'U')

        rJSVD_matmul_list.append(np.transpose(np.reshape(np.dot(U_2d_stacked[start:end,:],V_share_2d), [H,I,W,O]), [0,2,1,3])) 


    return rJSVD_matmul_list, rJSVD_dict, num_parameter_count_rJSVD



'''
============================================================
                Functions about CR calculation
============================================================
'''
def num_orig_all():

    orig_repeat_list = orig_Repeat_list[FLAGS.model]
    layers_out_channels = [64,128,256,512]

    kernal_size_first_conv = 3 if FLAGS.dataset in ['cifar10', 'cifar100'] else 7
    classes = 10 if FLAGS.dataset=='cifar10' else 100 if FLAGS.dataset=='cifar100' else 1000
    out_dimension = 512 if FLAGS.model in ['resnet18', 'resnet34'] else 2048

    #parameters of first conv / fc weight, fc bias
    num = kernal_size_first_conv*kernal_size_first_conv*3*64 + out_dimension*classes + classes

    #BasicBlock
    if FLAGS.model in ['resnet18', 'resnet34']:
        num += 64*128 + 128*256 + 256*512 #shortcut
        num += 3*3*64*64 + 3*3*64*128 + 3*3*128*256 + 3*3*256*512 #repeat0.conv1 of each layer
        for i in range(len(orig_repeat_list)):#3x3conv
            num += 3*3*layers_out_channels[i]*layers_out_channels[i]*(2*orig_repeat_list[i]-1)

    #BottleNeck:
    else:
        num += 64*256 + 256*512 + 512*1024 + 1024*2048 #shortcut
        num += 64*64 + 3*3*64*64 + 3*3*64*64 + 64*256 + (256*64 + 3*3*64*64 + 64*256)*(orig_repeat_list[0]-1)#layer1
        num += 256*128 + 3*3*128*128 + 128*512 + (512*128 + 3*3*128*128 + 128*512)*(orig_repeat_list[1]-1)#layer2
        num += 512*256 + 3*3*256*256 + 256*1024 + (1024*256 + 3*3*256*256 + 256*1024)*(orig_repeat_list[2]-1)#layer3
        num += 1024*512 + 3*3*512*512 + 512*2048 + (2048*512 + 3*3*512*512 + 512*2048)*(orig_repeat_list[3]-1)#layer4

    return num


def num_undecomposed_parameter():

    orig_repeat_list = orig_Repeat_list[FLAGS.model]
    layers_out_channels = [64,128,256,512]

    kernal_size_first_conv = 3 if FLAGS.dataset in ['cifar10', 'cifar100'] else 7
    classes = 10 if FLAGS.dataset=='cifar10' else 100 if FLAGS.dataset=='cifar100' else 1000
    out_dimension = 512 if FLAGS.model in ['resnet18', 'resnet34'] else 2048

    #parameters of first conv / fc weight, fc bias
    num = kernal_size_first_conv*kernal_size_first_conv*3*64 + out_dimension*classes + classes

    #BasicBlock
    if FLAGS.model in ['resnet18', 'resnet34']:
        num += 64*128 + 128*256 + 256*512 #shortcut
        num += (3*3*64*64*2) * orig_repeat_list[0] #layer1
        # if FLAGS.decom_conv1==False:
            # num += 3*3*64*128 + 3*3*128*256 +3*3*256*512 #conv1 of repeat0

    #BottleNeck: don't decompose 1x1 layer
    else:
        num += 64*256 + 256*512 + 512*1024 + 1024*2048 #shortcut
        num += 64*64 + 64*256 + (256*64 + 64*256)*(orig_repeat_list[0]-1) #layer1
        # num += 256*128 + 128*512 + (512*128 + 128*512)*(orig_repeat_list[1]-1) #layer2
        # num += 512*256 + 256*1024 + (1024*256 + 256*1024)*(orig_repeat_list[2]-1) #layer3
        # num += 1024*512 + 512*2048 + (2048*512 + 512*2048)*(orig_repeat_list[3]-1) #layer4

    return num




