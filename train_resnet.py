'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
'''


import tensorflow as tf
import numpy as np
from util import pytorch_pth2tensorflow_npy
import os
import sys
import time



flags = tf.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('model', 'resnet34', 'model name')
flags.DEFINE_string('dataset', 'imagenet', 'imagenet/cifar10/cifar100')
flags.DEFINE_boolean('from_scratch', False, 'from_scratch' )

flags.DEFINE_string('initializer', 'None', 'initializer')
flags.DEFINE_bool('bool_regularizer', False, 'regularizer')

flags.DEFINE_boolean('multi_gpus', False, 'gpu choosed to used' )
flags.DEFINE_string('gpu', '0', 'gpu choosed to used' )

flags.DEFINE_integer('batch_size', 128, 'Training batch_size' )
flags.DEFINE_integer('batch_size_eval', 500, 'Evaluating batch_size' )

flags.DEFINE_integer('epoch', 6, 'Training epoch')
flags.DEFINE_string('num_lr', '1e-3', 'initial learning_rate')
flags.DEFINE_string('change_lr', '[2,4]', 'epoch time to change num_lr')
flags.DEFINE_integer('lr_decay', 10, 'lr_decay')   

flags.DEFINE_integer('max_to_keep', 2, 'CKPT MAX TO SAVE' )
flags.DEFINE_string('root_path', './ckpt/orig', 'root_path' )
flags.DEFINE_string('ckpt_path', 'orig_ckpt', 'ckpt path to restore')
flags.DEFINE_string('extra_mark_path', '', 'extra_mark_path')


if FLAGS.multi_gpus==True:
    assert(0)
    import resnet_multi_gpus as resnet 
else:
    import resnet


os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ckpt_path = FLAGS.root_path +'/'+FLAGS.dataset+'/'+FLAGS.model+FLAGS.extra_mark_path
os.makedirs(ckpt_path, exist_ok=True)
log_path = ckpt_path+ '/'+FLAGS.dataset+'_'+FLAGS.model+'.log'


is_training = tf.placeholder(tf.bool, name = 'is_training')
lr = tf.placeholder(tf.float32, name='learning_rate')

if FLAGS.from_scratch:
    value_dict=None

if FLAGS.dataset=='imagenet':
    from util import imagenet_data, image_processing
    num_train_images = 1281167
    num_evalu_images = 50000

    imagenet_data_val = imagenet_data.ImagenetData('validation')
    imagenet_data_train = imagenet_data.ImagenetData('train')
    val_images, val_labels = image_processing.inputs(imagenet_data_val, batch_size=FLAGS.batch_size_eval, num_preprocess_threads=16)
    train_images, train_labels =  image_processing.distorted_inputs(imagenet_data_train, batch_size=FLAGS.batch_size, num_preprocess_threads=16)

    # https://blog.csdn.net/dongjbstrong/article/details/81128835
    x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    y = tf.cond(is_training, lambda: train_labels-1, lambda: val_labels-1, name='y')

    value_dict = pytorch_pth2tensorflow_npy.convert(FLAGS.model) if FLAGS.from_scratch==False else None


elif FLAGS.dataset=='cifar10':
    num_train_images = 50000
    num_evalu_images = 10000

    from util import cifar10_input
    with tf.device('/cpu:0'):
        train_images, train_labels = cifar10_input.distorted_inputs(data_dir='/home/test01/sambashare/sdd/cifar-10-batches-bin', batch_size= FLAGS.batch_size)
        val_images, val_labels = cifar10_input.inputs(data_dir='/home/test01/sambashare/sdd/cifar-10-batches-bin', eval_data=True, batch_size=FLAGS.batch_size_eval)

    x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    y = tf.cond(is_training, lambda: train_labels, lambda: val_labels, name='y')

    value_dict = pytorch_pth2tensorflow_npy.convert(FLAGS.model) if FLAGS.from_scratch==False else None

elif FLAGS.dataset=='cifar100':
    num_train_images = 50000
    num_evalu_images = 10000
    from util import cifar100_input
    # with tf.device('/cpu:0'):
    train_images, train_labels = cifar100_input.distorted_inputs(data_dir='/home/test01/sambashare/sdd/cifar-100-binary', batch_size= FLAGS.batch_size)
    val_images, val_labels = cifar100_input.inputs(data_dir='/home/test01/sambashare/sdd/cifar-100-binary', eval_data=True, batch_size=FLAGS.batch_size_eval)

    x = tf.cond(is_training, lambda: train_images, lambda: val_images, name='x')
    y = tf.cond(is_training, lambda: train_labels, lambda: val_labels, name='y')

else:
    print('Unsupported dataset: ', FLAGS.dataset)
    sys.exit(0)

print('start building network')
#dynamic calling the function
regularizer=tf.contrib.layers.l2_regularizer(5e-4) if FLAGS.bool_regularizer else None
logits, prediction = eval('resnet.'+ FLAGS.model)(x, FLAGS.dataset, is_training, eval(FLAGS.gpu),  weight_dict=value_dict, initializer=eval(FLAGS.initializer), regularizer=regularizer)
print('building network done\n\n')

Top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, y, 1), tf.float32), name='Top1')
Top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prediction, y, 5), tf.float32), name='Top5')

if FLAGS.bool_regularizer==True:
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #L2 Regularizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss') + tf.add_n(keys)
else:
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y), name='loss')

optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.9, name='Momentum' )

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, name = 'train_op')

tf.summary.histogram('prediction', prediction)
tf.summary.scalar('loss', loss)
tf.summary.scalar('Top1', Top1)
tf.summary.scalar('Top5', Top5)

def validate(sess):
    print('\n...validation...')
    top1 = 0
    top5 = 0

    start = time.time()
    num_iterations = int(num_evalu_images // FLAGS.batch_size_eval)
    for i in range(num_iterations):
        top_1, top_5 = sess.run([Top1, Top5], feed_dict={is_training: False})
        top1 += top_1
        top5 += top_5
    end = time.time()
    eval_time_one_epoch = end - start
    top1 /= num_iterations
    top5 /= num_iterations

    return top1, top5, eval_time_one_epoch


def log(informations):
    with open(log_path,'a') as f:
        for i in informations:
            print(i)
            f.write(str(i)+'\n')


def fine_tune(sess, saver, merged_summary_op, summary_wirter):
    print('raw validation')
    raw_top1, raw_top5, eval_time_one_epoch = validate(sess)
    informations = ["raw Top1:"+str(raw_top1)+' Top5:'+str(raw_top5), 'eval_time_one_epoch='+str(eval_time_one_epoch)+'\n']
    log(informations)

    best_top1 = -99999
    num_lr = eval(FLAGS.num_lr)

    epoch = FLAGS.epoch
    for j in range(epoch):
        print('\n...Training...')
        print('epoch=',epoch)
        print('j=',j)
        
        change_lr = eval(FLAGS.change_lr)
        if type(change_lr)==int:
            num_lr = num_lr/FLAGS.lr_decay if j%change_lr==0 else num_lr
        elif type(change_lr)==list:
            num_lr = num_lr/FLAGS.lr_decay if j in change_lr else num_lr
        else:
            print('wrong chang_lr')
            sys.exit(0)

        num_iterations = int(num_train_images // FLAGS.batch_size)
        print('num_iterations=',num_iterations)
        print('num_lr=', num_lr)
        ave_loss = 0

        start = time.time()
        for i in range(num_iterations):

            if i%(max(int(num_iterations//5),1))==0:#print & add summary  6 times
                print(i)
                summary_str, _, loss_eval = sess.run([merged_summary_op, train_op, loss], feed_dict={is_training: True, lr: num_lr})
                summary_wirter.add_summary(summary_str, i+num_iterations*j)

            else:
                 _, loss_eval = sess.run([train_op, loss], feed_dict={is_training: True, lr: num_lr})

            ave_loss += loss_eval/num_iterations


        end=time.time()
        running_time_one_epoch = end - start

        top1, top5, eval_time_one_epoch = validate(sess)

        informations = ["Epoch"+str(j)+" Top1:"+str(top1)+' Top5:'+str(top5),'num_lr='+str(num_lr), 'loss='+str(ave_loss),\
                         'Runing_Time_1epoch:%f m, eval_time_one_epoch:%f s\n'%(running_time_one_epoch/60,eval_time_one_epoch)]
        log(informations)

        # if top1>best_top1:
            # saver.save(sess, ckpt_path+'/'+FLAGS.model+'_top1_'+ str(top1), global_step=j)
        best_top1 = max(best_top1, top1)
    saver.save(sess, ckpt_path+'/'+FLAGS.model+'_top1_'+ str(top1), write_meta_graph=True)

def train(sess, saver, merged_summary_op, summary_wirter):
    best_top1 = -99999
    num_lr = eval(FLAGS.num_lr)

    epoch = FLAGS.epoch
    for j in range(epoch):
        print('\n...Training...')
        print('epoch=',epoch)
        print('j=',j)
        
        change_lr = eval(FLAGS.change_lr)
        if type(change_lr)==int:
            num_lr = num_lr/FLAGS.lr_decay if j%change_lr==0 else num_lr
        elif type(change_lr)==list:
            num_lr = num_lr/FLAGS.lr_decay if j in change_lr else num_lr
        else:
            print('wrong chang_lr')
            sys.exit(0)
        print('num_lr',num_lr)

        num_iterations = int(num_train_images // FLAGS.batch_size)
        print('num_iterations=',num_iterations)
        print('num_lr=', num_lr)
        ave_loss = 0

        start = time.time()
        for i in range(num_iterations):

            if i%(max(int(num_iterations//5),1))==0:#print & add summary  5 times
                print(i)
                summary_str, _, loss_eval = sess.run([merged_summary_op, train_op, loss], feed_dict={is_training: True, lr: num_lr})
                summary_wirter.add_summary(summary_str, i+num_iterations*j)
                print('loss=',loss_eval)

            else:
                _, loss_eval = sess.run([train_op, loss], feed_dict={is_training: True, lr: num_lr})


            ave_loss += loss_eval/num_iterations
        print(ave_loss)
        end=time.time()

        running_time_one_epoch = end - start

        top1, top5, eval_time_one_epoch = validate(sess)

        informations = ["Epoch"+str(j)+" Top1:"+str(top1)+' Top5:'+str(top5),'num_lr='+str(num_lr), 'loss='+str(ave_loss),\
                         'Runing_Time_1epoch:%f m, eval_time_one_epoch:%f s\n'%(running_time_one_epoch/60,eval_time_one_epoch)]
        log(informations)

        if j>epoch-11:
            saver.save(sess, ckpt_path+'/'+FLAGS.model+'_top1_'+ str(top1), global_step=j)
        # best_top1 = max(best_top1, top1)


def main():
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        merged_summary_op = tf.summary.merge_all()
        summary_wirter = tf.summary.FileWriter(ckpt_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars += [g for g in g_list if 'Momentum' in g.name]
        var_list += bn_moving_vars

        saver = tf.train.Saver(var_list=var_list, max_to_keep=FLAGS.max_to_keep)
        saver.save(sess, ckpt_path+'/'+FLAGS.model+'_raw')
        try:
            
            with open(log_path,'a') as f:
                for _key in FLAGS:
                    print(_key+': '+str(FLAGS[_key].value)+'\n')
                    f.write(_key+': '+str(FLAGS[_key].value)+'\n')
            if FLAGS.from_scratch==False:
                fine_tune(sess, saver, merged_summary_op,summary_wirter)
            else:
                train(sess, saver, merged_summary_op,summary_wirter)


        finally:
            coord.request_stop()
            coord.join(threads)

main()





