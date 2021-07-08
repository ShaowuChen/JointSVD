#coding:utf-8

'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
'''


import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
import os
import sys
import time
import resnet_decom_revising
import resnet_decom
import resnet


flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('model', 'resnet34', 'model name')
flags.DEFINE_string('method', 'lJSVD', "orig/lJSVD/Bi-JSVD")
flags.DEFINE_string('rank_rate_SVD', '0.08', 'rank_rate_SVD')
flags.DEFINE_string('ratio_lJSVD', '0.5', 'for Bi_JSVD')

flags.DEFINE_bool('from_scratch', False, 'for Bi_JSVD')
flags.DEFINE_string('dataset', 'cifar10', 'imagenet/cifar10/cifar100')
flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )

flags.DEFINE_integer('batch_size', 128, 'Training batch_size' )
flags.DEFINE_integer('batch_size_eval', 500, 'Evaluating batch_size' )


flags.DEFINE_string('root_path', './ckpt', 'root_path' )
flags.DEFINE_string('time_path', 'inference_evaluating_time', 'time evaluation')
flags.DEFINE_string('exp_path', 'exp1', 'exp1/2/3')



os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


is_training = tf.placeholder(tf.bool, name = 'is_training')

if FLAGS.dataset=='imagenet':
    from util import imagenet_data, image_processing
    num_train_images = 1281167
    num_evalu_images = 50000

    imagenet_data_val = imagenet_data.ImagenetData('validation')
    imagenet_data_train = imagenet_data.ImagenetData('train')
    val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=FLAGS.batch_size_eval, num_preprocess_threads=4)
    # train_images, train_labels =  image_processing.distorted_inputs(imagenet_data_train, batch_size=FLAGS.batch_size, num_preprocess_threads=4)

    # https://blog.csdn.net/dongjbstrong/article/details/81128835
    # x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    # y = tf.cond(is_training, lambda: train_labels-1, lambda: val_labels-1, name='y')

elif FLAGS.dataset=='cifar10':
    num_train_images = 50000
    num_evalu_images = 10000

    from util import cifar10_input
    with tf.device('/cpu:0'):
        # train_images, train_labels = cifar10_input.distorted_inputs(data_dir='/home/test01/sambashare/sdd/cifar-10-batches-bin', batch_size= FLAGS.batch_size)
        val_images, val_labels = cifar10_input.inputs(data_dir='/home/test01/sambashare/sdd/cifar-10-batches-bin', eval_data=True, batch_size=FLAGS.batch_size_eval)

    # x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    # y = tf.cond(is_training, lambda: train_labels, lambda: val_labels, name='y')

elif FLAGS.dataset=='cifar100':
    num_train_images = 50000
    num_evalu_images = 10000
    from util import cifar100_input
    with tf.device('/cpu:0'):
        # train_images, train_labels = cifar100_input.distorted_inputs(data_dir='/home/test01/sambashare/sdd/cifar-100-binary', batch_size= FLAGS.batch_size)
        val_images, val_labels = cifar100_input.inputs(data_dir='/home/test01/sambashare/sdd/cifar-100-binary', eval_data=True, batch_size=FLAGS.batch_size_eval)

    # x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    # y = tf.cond(is_training, lambda: train_labels, lambda: val_labels, name='y')

else:
    print('Unsupported dataset: ', FLAGS.dataset)
    sys.exit(0)

x=val_images
y=val_labels


print('===============================Start building network===============================')

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
Path = 

name_dic = {
    'lJSVD':'lJSVD',
    'JSVD':'rJSVD_1',

}


if FLAGS.method=='orig':
    if FLAGS.dataset=='cifar10':
        checkpoint_path=cifar10_Path[FLAGS.model]
    elif FLAGS.dataset=='cifar100':
        checkpoint_path=cifar100_Path[FLAGS.model]
else:
    
    if FLAGS.method in ['lJSVD','rJSVD','JSVD']:
        checkpoint_path ='/home/test01/sambashare/sdh/CoupledSVD/ckpt/20210315/cifar10_300epoch/'+FLAGS.model+'/'+FLAGS.method
        checkpoint_path = checkpoint_path + '/'+ FLAGS.rank_rate_SVD + '_i_over_sNone'
    elif FLAGS.method=='Bi_JSVD':
        time_p = '20210612' if FLAGS.model=='resnet18' else '20210315'
        method_p = 'Bi_JSVD_old' if FLAGS.model=='resnet34' else FLAGS.method
        last_name =  FLAGS.rank_rate_SVD + '_i_over_sNone' if FLAGS.model=='resnet34' else FLAGS.rank_rate_SVD + '_left_ratio0.5'
        checkpoint_path ='/home/test01/sambashare/sdh/CoupledSVD/ckpt/'+time_p+'/cifar10_300epoch/'+FLAGS.model+'/'+method_p + '/' +last_name
        # checkpoint_path = checkpoint_path + '/'+ FLAGS.rank_rate_SVD + '_left_ratio0.5'
    else:  
        assert(0)


    file_list = os.listdir(checkpoint_path)
    if file_list==[]:
        print('empty')
        assert(0)

    mark = False
    for item in file_list:
        if '.meta' in item:
            mark = True
            checkpoint_path = checkpoint_path + '/' + item[:-5]
            break
    if not mark:
        assert(0)

# log_path = '/home/test01/sambashare/sdh/CoupledSVD/ckpt/20210315/'+FLAGS.time_path
# os.makedirs(log_path, exist_ok=True)
# log_path = '/home/test01/sambashare/sdh/CoupledSVD/ckpt/20210315/'+FLAGS.time_path+'/inference_evaluating_time.log'

log_path = '/home/test01/sambashare/sdh/CoupledSVD/ckpt/'+FLAGS.time_path
os.makedirs(log_path, exist_ok=True)
log_path = '/home/test01/sambashare/sdh/CoupledSVD/ckpt/'+FLAGS.time_path+'/inference_evaluating_time.log'

if FLAGS.gpu=='0':
    log_path2 = '/home/test01/sambashare/sdh/CoupledSVD/ckpt/'+FLAGS.time_path+'/gpu_time.log'

else:
    assert(0)

with open(log_path, 'a+') as f:
    f.write('checkpoint_path: '+checkpoint_path+'\n')
    
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    var_to_shape_map[key] = reader.get_tensor(key)

if FLAGS.method=='orig':
    logits, prediction = eval('resnet.'+FLAGS.model)(x, FLAGS.dataset, False, 0, weight_dict=var_to_shape_map, initializer=None, regularizer=None)
    Top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, y, 1), tf.float32), name='Top1')
elif FLAGS.method=='Bi_JSVD' and FLAGS.model=='resnet18':
    logits, prediction = eval('resnet_decom_revising.'+FLAGS.model)(x, FLAGS.dataset, False, 0, weight_dict=var_to_shape_map, initializer=None, regularizer=None)
    Top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, y, 1), tf.float32), name='Top1')
else:
    logits, prediction = eval('resnet_decom.'+FLAGS.model)(x, FLAGS.dataset, False, 0, weight_dict=var_to_shape_map, initializer=None, regularizer=None)
    Top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, y, 1), tf.float32), name='Top1')


with tf.Session(config=config) as sess:

    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    sess.run(tf.global_variables_initializer())

    try:

        print('\n...validation...')
        #Warm up the gpu
        top1 = 0
        num_iterations = int(num_evalu_images // FLAGS.batch_size_eval)
        for i in range(num_iterations):
            top_1 = sess.run(Top1, feed_dict={is_training: False})
            top1 += top_1
        top1 /= num_iterations


        repeat_times = 5
        start = time.time()
        for i in range(num_iterations*repeat_times):
            logit = sess.run(logits, feed_dict={is_training: False})
        end = time.time()
        eval_time_one_figure_ms = (end - start)/repeat_times/num_evalu_images *1000 #average time per figure (ms)

        with open(log_path, 'a+') as f:
            f.write('acc: '+str(top1)+'\n')
            f.write('eval_time_one_figure_ms: '+str(eval_time_one_figure_ms)+' ms'+'\n\n\n')

        with open(log_path2, 'a+') as f:
            f.write(', '+str(eval_time_one_figure_ms))

    finally:
        coord.request_stop()
        coord.join(threads)


