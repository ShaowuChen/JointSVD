import tensorflow as tf
import numpy as np 

flags = tf.flags
FLAGS=flags.FLAGS

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com

Description (method name in code: method name in paper):
SVD: SVD
TT: Tensor Train Decomposition==Tucker Decompositon (Note that for unfoled 3D tensor Tucker=TT)
rJSVD_1: RJSVD_1
rJSVD_2  RJSVD_2
lJSVD: LJSVD
Bi_JSVD: Bi-JSVD. 
'''

# global initializer ,regularizer

# bm
# eps=1e-5  momentum=0.1  affine=True、
#Note: momentum_in_bn_tensorflow = 1 - momentum_in_bn_pytorch
def bn(in_tensor, layer_num, repeat_num, conv_num, is_training, weight_dict, name_bn_scope=None, downsample=False):
    #Note: momentum_in_bn_tensorflow = 1 - momentum_in_bn_pytorch
    #Used by conv1; conv3x3; conv1x1; shortcut downsample;
    
    if name_bn_scope==None:
        if downsample:#for shortcut downsample
            name_bn_scope = 'layer' + str(layer_num) + '.'+str(repeat_num) + '.downsample.1'
        else:#for conv3x3; conv1x1 in bottleneck/basicblock
            name_bn_scope = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'bn' + str(conv_num)
    else:#for conv1
        pass


    if FLAGS.from_scratch==True:
        tensor_go_next = tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5, training=is_training, name=name_bn_scope)
    else:
        #gamma=scale/weight;beta=offset;
        tensor_go_next = tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5,
                                                beta_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/beta"]),
                                                gamma_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/gamma"]),
                                                moving_mean_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/moving_mean"]),
                                                moving_variance_initializer=tf.constant_initializer(weight_dict[name_bn_scope+"/moving_variance"]),
                                                training=is_training, name=name_bn_scope)
    return tensor_go_next


def Shared_Component(in_tensor, layer_num, conv_num, stride, weight_dict, initializer, regularizer):
    #Shared Components
    name_weight1 = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.U_shared'
    name_weight2 = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.V_shared'

    if FLAGS.from_scratch==True:
        weight1 = tf.get_variable(shape=weight_dict[name_weight1].shape, name=name_weight1, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight2 = tf.get_variable(shape=weight_dict[name_weight2].shape, name=name_weight2, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
    else:
        weight1 = tf.get_variable(name=name_weight1, initializer=tf.constant(weight_dict[name_weight1]), regularizer=regularizer)
        weight2 = tf.get_variable(name=name_weight2, initializer=tf.constant(weight_dict[name_weight2]), regularizer=regularizer)

    tensor_go_next_share = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')
    tensor_go_next = tf.nn.conv2d(tensor_go_next_share, weight2, [1, 1, stride, 1], padding='SAME')

    return tensor_go_next


def Independent_Component(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer):
    #Independent Components
    name_weight = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.' + 'conv' + str(conv_num) + '.'
    name_weight1 = name_weight + 'U'
    name_weight2 = name_weight + 'V'

    if FLAGS.from_scratch==True:
        weight1 = tf.get_variable(shape=weight_dict[name_weight1].shape, name=name_weight1, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight2 = tf.get_variable(shape=weight_dict[name_weight2].shape, name=name_weight2, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
    else:
        weight1 = tf.get_variable(name=name_weight1, initializer=tf.constant(weight_dict[name_weight1]), regularizer=regularizer)
        weight2 = tf.get_variable(name=name_weight2, initializer=tf.constant(weight_dict[name_weight2]), regularizer=regularizer)

    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, stride, 1], padding='SAME')

    return tensor_go_next


def JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option=None):
    assert(option in ['rJSVD_1', 'rJSVD_2', 'lJSVD'])

    if option=='rJSVD_1' or  option=='rJSVD_2':
        #independent U, shared V      
        name_weight1 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.U_partly_shared'
        name_weight2 = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.V_shared'

    elif option=='lJSVD':
        #shared U,  independent V   
        name_weight1 = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.U_shared'   
        name_weight2 = 'layer' + str(layer_num) + '.' + str(repeat_num) + '.conv' + str(conv_num) + '.V_partly_shared'
        

    if FLAGS.from_scratch==True:
        weight1 = tf.get_variable(shape=weight_dict[name_weight1].shape, name=name_weight1, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight2 = tf.get_variable(shape=weight_dict[name_weight2].shape, name=name_weight2, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
    else:
        weight1 = tf.get_variable(name=name_weight1, initializer=tf.constant(weight_dict[name_weight1]), regularizer=regularizer)
        weight2 = tf.get_variable(name=name_weight2, initializer=tf.constant(weight_dict[name_weight2]), regularizer=regularizer)


    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, stride, 1], padding='SAME')

    return tensor_go_next


def TTD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride, end_name=''):
    name_weight1 = name_weight + end_name + '1'
    name_weight2 = name_weight + end_name + '2'
    name_weight3 = name_weight + end_name + '3'
    if FLAGS.from_scratch==True:
        weight1 = tf.get_variable(shape=weight_dict[name_weight1].shape, name=name_weight1, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight2 = tf.get_variable(shape=weight_dict[name_weight2].shape, name=name_weight2, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight3 = tf.get_variable(shape=weight_dict[name_weight3].shape, name=name_weight3, dtype=tf.float32, initializer=initializer, regularizer=regularizer)

    else:
        weight1  = tf.get_variable(name=name_weight1, initializer=tf.constant(weight_dict[name_weight1]), regularizer=regularizer) 
        weight2  = tf.get_variable(name=name_weight2, initializer=tf.constant(weight_dict[name_weight2]), regularizer=regularizer) 
        weight3  = tf.get_variable(name=name_weight3, initializer=tf.constant(weight_dict[name_weight3]), regularizer=regularizer) 

    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, 1, 1, 1], padding='SAME')            
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, stride, stride, 1], padding='SAME')            
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight3, [1, 1, 1, 1], padding='SAME')

    return tensor_go_next

def SVD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride):
    name_weight1 = name_weight + 'U'
    name_weight2 = name_weight + 'V'
    if FLAGS.from_scratch==True:
        weight1 = tf.get_variable(shape=weight_dict[name_weight1].shape, name=name_weight1, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight2 = tf.get_variable(shape=weight_dict[name_weight2].shape, name=name_weight2, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
    else:
        weight1  = tf.get_variable(name=name_weight1, initializer=tf.constant(weight_dict[name_weight1]), regularizer=regularizer) 
        weight2  = tf.get_variable(name=name_weight2, initializer=tf.constant(weight_dict[name_weight2]), regularizer=regularizer)  

    tensor_go_next = tf.nn.conv2d(in_tensor, weight1, [1, stride, 1, 1], padding='SAME')
    tensor_go_next = tf.nn.conv2d(tensor_go_next, weight2, [1, 1, stride, 1], padding='SAME')

    return tensor_go_next

def conv3x3(in_tensor, out_channels:int, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer):

    assert(FLAGS.method in ['SVD', 'TT', 'NC_CTD', 'PCSVD', 'rJSVD_1',  'rJSVD_2', 'lJSVD', 'Bi_JSVD'])

    if  layer_num==1:#keep it the same
        name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.weight'
        if FLAGS.from_scratch==True:
            weight = tf.get_variable(shape=weight_dict[name_weight].shape, name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer) 
        else:
            weight = tf.get_variable(name=name_weight, initializer=tf.constant(weight_dict[name_weight]), regularizer=regularizer) 

        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides=[1,stride, stride,1], padding='SAME')

    else:
        name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.'

        if FLAGS.method=='SVD':
            tensor_go_next = SVD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)

        elif FLAGS.method=='rJSVD_1':
            tensor_go_next = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_1')

        elif FLAGS.method in ['rJSVD_2', 'lJSVD', 'Bi_JSVD']:
            if (repeat_num==0 and conv_num==1) or (FLAGS.model=='resnet18' and conv_num==1):  #conv1.weight of layer 2to4 | repeat.x are NOT jointly deocmposed(but svd)
                tensor_go_next = SVD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)            
            else:
                if FLAGS.method in ['rJSVD_2', 'lJSVD']:
                    tensor_go_next = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option=FLAGS.method)
                
                elif FLAGS.method=='Bi_JSVD':
                    tensor_go_next1 = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_2')
                    tensor_go_next2 = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='lJSVD')
                    tensor_go_next = tensor_go_next1 + tensor_go_next2
        
        elif FLAGS.method=='PCSVD':
            tensor_go_next_Partly_share = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_2')
            tensor_go_next_independent = Independent_Component(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer)
            tensor_go_next = tensor_go_next_Partly_share + tensor_go_next_independent

        elif FLAGS.method=='TT':
            tensor_go_next = TTD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)

        elif FLAGS.method=='NC_CTD':
            if (repeat_num==0 and conv_num==1) or (FLAGS.model=='resnet18' and conv_num==1): #conv1.weight of layer 2to4 | repeat.x are NOT jointly deocmposed(but svd)
                tensor_go_next = TTD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)
            else:
                part_of_shared_name = 'layer' + str(layer_num) + '.conv' + str(conv_num) + '.'
                tensor_go_next_s = TTD(in_tensor, weight_dict, part_of_shared_name, initializer, regularizer, stride, 'S')
                tensor_go_next_i = TTD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride, 'I')
                tensor_go_next = tensor_go_next_i + tensor_go_next_s

        else:
            assert(0)

    return tensor_go_next

def conv1x1(in_tensor, out_channels:int, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, downsample=False):
    # with tf.device('/cpu:0'):
    if downsample==False:#botteneck
     
        if layer_num==1: #keep it the same
            name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.weight'
            if FLAGS.from_scratch==True:
                weight = tf.get_variable(shape=weight_dict[name_weight].shape, name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
            else:
                weight = tf.get_variable(name=name_weight, initializer=tf.constant(weight_dict[name_weight]), regularizer=regularizer)                

            tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides=[1,stride, stride,1], padding='SAME')

        else:#decompose it 
            name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'conv' + str(conv_num) + '.'

            if FLAGS.method=='SVD':
                tensor_go_next = SVD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)

            elif FLAGS.method=='TT':
                assert(0)
            elif FLAGS.method=='NC_CTD':
                assert(0)

            elif FLAGS.method=='rJSVD_1':
                tensor_go_next = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_1')

            elif FLAGS.method in ['rJSVD_2', 'lJSVD', 'Bi_JSVD']:
                if (repeat_num==0 and conv_num==1) or (FLAGS.model=='resnet18' and conv_num==1):  #conv1.weight of layer 2to4 | repeat.x are NOT jointly deocmposed(but svd)
                    tensor_go_next = SVD(in_tensor, weight_dict, name_weight, initializer, regularizer, stride)          
                
                else:
                    if FLAGS.method in ['rJSVD_2', 'lJSVD']:
                        tensor_go_next = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option=FLAGS.method)
                    
                    elif FLAGS.method=='Bi_JSVD':
                        tensor_go_next1 = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_2')
                        tensor_go_next2 = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='lJSVD')
                        tensor_go_next = tensor_go_next1 + tensor_go_next2
            
            elif FLAGS.method=='PCSVD':
                tensor_go_next_Partly_share = JSVD(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer, option='rJSVD_2')
                tensor_go_next_independent = Independent_Component(in_tensor, layer_num, repeat_num, conv_num, stride, weight_dict, initializer, regularizer)
                tensor_go_next = tensor_go_next_Partly_share + tensor_go_next_independent

            else:
                assert(0)

    else:#shorcut weight, keep it the same

        name_weight = 'layer'+ str(layer_num) + '.'+str(repeat_num) + '.' + 'downsample.0.weight'
        if FLAGS.from_scratch==True:
            weight = tf.get_variable(shape=weight_dict[name_weight].shape, name=name_weight, dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        else:
            weight = tf.get_variable(name=name_weight, initializer=tf.constant(weight_dict[name_weight]), regularizer=regularizer)                

        tensor_go_next = tf.nn.conv2d(in_tensor, weight, strides=[1,stride, stride,1], padding='SAME')

    return tensor_go_next



def BasicBlock(in_tensor, layer_num, repeat_num, is_training, weight_dict, initializer, regularizer):
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


def Bottleneck(in_tensor, layer_num, repeat_num, is_training, weight_dict, initializer, regularizer):#expansion
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

    if FLAGS.from_scratch==True:
        weight_conv1 = tf.get_variable(shape=[kernel_size_conv1, kernel_size_conv1, 3, 64], name='conv1.weight', dtype=tf.float32, initializer=initializer, regularizer=regularizer)
        weight_fc =  tf.get_variable(shape=[size_fc_input, output_dimension],  dtype=tf.float32, name='fc.weight',  initializer=initializer,regularizer=regularizer)
        bias_fc = tf.get_variable(shape=[output_dimension],  dtype=tf.float32, name='fc.bias', initializer=initializer,regularizer=regularizer)     
    else:
        weight_conv1 = tf.get_variable(name='conv1.weight', initializer=tf.constant(weight_dict['conv1.weight']), regularizer=regularizer)
        weight_fc =  tf.get_variable(name='fc.weight', initializer=tf.constant(weight_dict['fc.weight']), regularizer=regularizer)
        bias_fc = tf.get_variable(name='fc.bias', initializer=tf.constant(weight_dict['fc.bias']), regularizer=regularizer)  
     
    return weight_conv1, weight_fc, bias_fc    
    


def ResNet(x_palceholder, block_function, dataset, layer_list, is_training, weight_dict, initializer, regularizer):

    def _make_layer(in_tensor, block_function, layer_num, repeat_times, is_training, weight_dict, initializer, regularizer):
        tensor_go_next =  in_tensor
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
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
    assert(tensor_go_next.shape[1]==tensor_go_next.shape[1]==1)
    tensor_go_next_flatten = tf.reshape(tensor_go_next, [-1, tensor_go_next.shape[3]], name='tensor_go_next_flatten') #[b,512] or [b,2048]
    logits = tf.add(tf.matmul(tensor_go_next_flatten, weight_fc), bias_fc, name='logits')

    softmax = tf.nn.softmax(logits, name='softmax')

    return logits, softmax



def resnet18(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):

    layer_list = [2,2,2,2]
    logits, softmax = ResNet(x_palceholder, BasicBlock, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet34(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):

    layer_list = [3,4,6,3]
    logits, softmax = ResNet(x_palceholder, BasicBlock, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet50(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
   
    layer_list = [3,4,6,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax


def resnet101(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
   
    layer_list = [3,4,23,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax



def resnet152(x_palceholder, dataset, is_training, gpu, weight_dict=None, initializer=None, regularizer=None):
   
    layer_list = [3,8,36,3]
    logits, softmax = ResNet(x_palceholder, Bottleneck, dataset, layer_list, is_training, weight_dict, initializer, regularizer)
    return logits, softmax


