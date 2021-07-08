
'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
Time: 2020/12/13

Describe:
    Network: 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    Dataset: imagenet(224x224x3)/cifar10(32x32x3)/cifar100(32x32x3);
    Train: 
        1、from scratch(random initializer)： pass the parameter 'weight_dict=None', 'initializer=tf.xxx.xxxinitializer()';
        2、with a dic: pass the parameter 'weight_dict=the_parameter_ditionary'; For example, download the Pytorch .pth and transpose it to be .npy which can be read as a dict, then build the network in tf with this dict.  

Platform: Ubuntu; Tensorflow1.15


Reference: 
    Paper: https://arxiv.org/pdf/1512.03385.pdf (Batchsize=145, 128epoch(60*10^4iterations), momentum0.9, weight decay=0.0001;learning rate starts from 0.1 and is divided by 10 when the error plateaus)
    Pytorch_Implement:https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

'''

import tensorflow as tf
import numpy as np 

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# global initializer ,regularizer

# bm
# eps=1e-5  momentum=0.1  affine=True、
#Note: momentum_in_bn_tensorflow = 1 - momentum_in_bn_pytorch
def bn(in_tensor, layer_num, repeat_num, conv_num, is_training, weight_dict=None, name_bn_scope=None, downsample=False):
    #Note: momentum_in_bn_tensorflow = 1 - momentum_in_bn_pytorch
    #Used by conv1; conv3x3; conv1x1; shortcut downsample;
    
    if name_bn_scope==None:
        if downsample:#for shortcut downsample
            name_bn_scope = 'layer' + str(layer_num) + '.'+str(repeat_num) + '.downsample.1'
        else:#for conv3x3; conv1x1 in bottleneck/basicblock
            name_bn_scope = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'bn' + str(conv_num)
    else:#for conv1
        pass

    if weight_dict!=None:
        #gamma=scale/weight;beta=offset;
        tensor_go_next = tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5,
                                                beta_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/moving_variance"]),
                                                training=is_training, name=name_bn_scope)
    else:
        tensor_go_next = tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5, training=is_training, name=name_bn_scope)

    return tensor_go_next
 

def conv3x3(in_tensor, out_channels:int, layer_num, repeat_num, conv_num, stride, weight_dict=None, initializer=None, regularizer=None):
    # if name_weight==None:
    name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.weight'

    # with tf.device('/cpu:0'):
    if weight_dict!=None:
        weight = tf.get_variable(name=name_weight, initializer=tf.constant(weight_dict[name_weight]), regularizer=regularizer) 
        # print(name_weight, weight_dict[name_weight].shape)      
    else:
        weight = tf.get_variable(shape=[3, 3, in_tensor.shape[3], out_channels], name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer)

    tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides=[1,stride, stride,1], padding='SAME')

    return tensor_go_next


def conv1x1(in_tensor, out_channels:int, layer_num, repeat_num, conv_num, stride, weight_dict=None, initializer=None, regularizer=None, downsample=False):
    # with tf.device('/cpu:0'):
    if downsample==False:#botteneck
        name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.weight'
    else:#shorcut weight
        name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'downsample.0.weight'

    if weight_dict!=None:
        weight = tf.get_variable(name=name_weight, initializer=tf.constant(weight_dict[name_weight]), regularizer=regularizer)        
    else:
        weight = tf.get_variable(shape=[1, 1, in_tensor.shape[3], out_channels], name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer)

    tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides=[1,stride, stride,1], padding='SAME')

    return tensor_go_next


def BasicBlock(in_tensor, layer_num, repeat_num, is_training, weight_dict=None, initializer=None, regularizer=None):
    layers_out_channels = [64,128,256,512]

    # in_tensor.shape[3]!=out_channels  <==>  stride = 2 <==> downsample    
    if in_tensor.shape[3]!=layers_out_channels[layer_num-1]:
        stride = 2
        downsample = True
    else:
        stride = 1
        downsample = False

    tensor_go_next = conv3x3(in_tensor, layers_out_channels[layer_num-1], layer_num, repeat_num, 1, stride, weight_dict, initializer, regularizer)
    tensor_go_next = bn(tensor_go_next, layer_num, repeat_num, 1, is_training, weight_dict, name_bn_scope=None)
    tensor_go_next = tf.nn.relu(tensor_go_next)

    tensor_go_next = conv3x3(tensor_go_next, layers_out_channels[layer_num-1], layer_num, repeat_num, 2, 1, weight_dict, initializer, regularizer)
    tensor_go_next = bn(tensor_go_next, layer_num, repeat_num, 2, is_training, weight_dict, name_bn_scope=None)
    
    if downsample:
        identity = conv1x1(in_tensor, layers_out_channels[layer_num-1], layer_num, repeat_num, None, stride, weight_dict, initializer, regularizer, True)
        identity = bn(identity, layer_num, repeat_num, None, is_training, weight_dict, name_bn_scope=None, downsample=True)

    else:
        identity = in_tensor

    tensor_go_next += identity
    tensor_go_next = tf.nn.relu(tensor_go_next)

    return tensor_go_next


def Bottleneck(in_tensor, layer_num, repeat_num, is_training, weight_dict=None, initializer=None, regularizer=None):#expansion
    expansion = 4
    layers_out_channels = [64,128,256,512]
    expanded_layers_out_channels = [item*4 for item in layers_out_channels]

    stride=2 if (layer_num!=1 and repeat_num==0) else 1    

    tensor_go_next = conv1x1(in_tensor, layers_out_channels[layer_num-1], layer_num, repeat_num, 1, 1, weight_dict, initializer, regularizer, downsample=False)
    tensor_go_next = bn(tensor_go_next, layer_num, repeat_num, 1, is_training, weight_dict, name_bn_scope=None)
    tensor_go_next = tf.nn.relu(tensor_go_next)

    # Both conv2 of Bottleneck and downsample layers downsample the input when stride != 1
    tensor_go_next = conv3x3(tensor_go_next, layers_out_channels[layer_num-1], layer_num, repeat_num, 2, stride, weight_dict, initializer, regularizer)
    tensor_go_next = bn(tensor_go_next, layer_num, repeat_num, 2, is_training, weight_dict, name_bn_scope=None)
    tensor_go_next = tf.nn.relu(tensor_go_next)

    tensor_go_next = conv1x1(tensor_go_next, expanded_layers_out_channels[layer_num-1], layer_num, repeat_num, 3, 1, weight_dict, initializer, regularizer, downsample=False)
    tensor_go_next = bn(tensor_go_next, layer_num, repeat_num, 3, is_training, weight_dict, name_bn_scope=None)

    
    if repeat_num==0:
        identity = conv1x1(in_tensor, expanded_layers_out_channels[layer_num-1], layer_num, repeat_num, None, stride, weight_dict, initializer, regularizer, downsample=True)
        identity = bn(identity, layer_num, repeat_num, None, is_training, weight_dict, name_bn_scope=None, downsample=True)
    else:
        identity = in_tensor

    tensor_go_next += identity
    tensor_go_next = tf.nn.relu(tensor_go_next)

    return tensor_go_next


def set_variable_other_layer(dataset, block_function, weight_dict, initializer, regularizer):
    kernel_size_conv1=7 if dataset in ['imagenet','uc'] else 3
    stride_size_conv1=2 if dataset in ['imagenet','uc'] else 1
    output_dimension=1000 if dataset=='imagenet' else 10 if dataset=='cifar10' else 100 if dataset=='cifar100' else 21 if dataset=='uc' else 0
    size_fc_input=512 if block_function==BasicBlock else 2048

    # with tf.device('/cpu:0'):
    if weight_dict==None:
        weight_conv1 = tf.get_variable(shape=[kernel_size_conv1, kernel_size_conv1, 3, 64], name='conv1.weight', dtype=tf.float32, initializer=initializer, regularizer=regularizer)

        weight_fc =  tf.get_variable(shape=[size_fc_input, output_dimension],  dtype=tf.float32, name='fc.weight',  initializer=initializer,regularizer=regularizer)
        bias_fc = tf.get_variable(shape=[output_dimension],  dtype=tf.float32, name='fc.bias', initializer=initializer,regularizer=regularizer)        
    else:
        weight_conv1 = tf.get_variable(name='conv1.weight', initializer=tf.constant(weight_dict['conv1.weight']), regularizer=regularizer)

        weight_fc =  tf.get_variable(name='fc.weight', initializer=tf.constant(weight_dict['fc.weight']), regularizer=regularizer)
        bias_fc = tf.get_variable(name='fc.bias', initializer=tf.constant(weight_dict['fc.bias']), regularizer=regularizer)  

    return weight_conv1, weight_fc, bias_fc    
    


def ResNet(x_palceholder, block_function, dataset, layer_list, is_training, weight_dict, initializer, regularizer):

    def _make_layer(in_tensor, block_function, layer_num, repeat_times, is_training, weight_dict=None, initializer=None, regularizer=None):
        tensor_go_next =  in_tensor
        for i in range(repeat_times):
            tensor_go_next = block_function(tensor_go_next, layer_num, i, is_training, weight_dict, initializer, regularizer)
        return tensor_go_next

    weight_conv1, weight_fc, bias_fc =  set_variable_other_layer(dataset, block_function, weight_dict, initializer, regularizer)

    stride_size_conv1=2 if dataset in ['imagenet','uc'] else 1
    tensor_go_next = tf.nn.conv2d(x_palceholder, weight_conv1, strides=[1,stride_size_conv1,stride_size_conv1,1], padding='SAME')
    tensor_go_next = bn(tensor_go_next, None, None, None, is_training, weight_dict=weight_dict, name_bn_scope='bn1')
    tensor_go_next = tf.nn.relu(tensor_go_next)

    if dataset in ['imagenet','uc']:
        tensor_go_next = tf.nn.max_pool(tensor_go_next, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

    tensor_go_next = _make_layer(tensor_go_next, block_function, 1, layer_list[0], is_training, weight_dict, initializer, regularizer)
    tensor_go_next = _make_layer(tensor_go_next, block_function, 2, layer_list[1], is_training, weight_dict, initializer, regularizer)
    tensor_go_next = _make_layer(tensor_go_next, block_function, 3, layer_list[2], is_training, weight_dict, initializer, regularizer)
    tensor_go_next = _make_layer(tensor_go_next, block_function, 4, layer_list[3], is_training, weight_dict, initializer, regularizer)

    tensor_go_next = tf.layers.average_pooling2d(tensor_go_next, pool_size=7 if dataset in ['imagenet','uc'] else 4, strides=1)
    assert(tensor_go_next.shape[3] in [512,2048])
    print(tensor_go_next)
    assert(tensor_go_next.shape[1]==tensor_go_next.shape[1]==1)
    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1, tensor_go_next.shape[3]], name='tensor_go_next_flatten') #[b,512] or [b,2048]
    logits = tf.add(tf.matmul(tensor_go_next_flatten, weight_fc), bias_fc, name='logits')

    softmax = tf.nn.softmax(logits, name='softmax')

    return logits, softmax





def resnet18(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
    if weight_dict==None and initializer==None:
        initializer = tf.truncated_normal_initializer(stddev=0.01)

    layer_list = [2,2,2,2]
    logits, softmax = ResNet(x_palceholder, BasicBlock, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet34(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
    if weight_dict==None and initializer==None:
        initializer = tf.truncated_normal_initializer(stddev=0.01)

    layer_list = [3,4,6,3]
    logits, softmax = ResNet(x_palceholder, BasicBlock, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet50(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
    if weight_dict==None and initializer==None:
        initializer = tf.truncated_normal_initializer(stddev=0.01)
   
    layer_list = [3,4,6,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax


def resnet101(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
    if weight_dict==None and initializer==None:
        initializer = tf.truncated_normal_initializer(stddev=0.01)
   
    layer_list = [3,4,23,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet152(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
    if weight_dict==None and initializer==None:
        initializer = tf.truncated_normal_initializer(stddev=0.01)
   
    layer_list = [3,8,36,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax


'''test_code'''

# tf.reset_default_graph()
# a = tf.get_variable(name='aa', initializer=tf.constant(np.ones(shape=[10,32,32,3]).astype(np.float32)))
# logits, softmax = resnet18(a,'cifar100',False)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     logits, softmax = sess.run([logits,softmax])
#     print(logits.shape)
