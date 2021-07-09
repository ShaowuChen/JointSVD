Welcome!

This repository contains codes for the paper :

**Joint Matrix Decomposition for Deep Convolutional Neural Networks compression**

Feel free to ask me questions, and please cite our work if it help.

(Currently summited to the *Neurocomputing*, and we are considering  to upload it to arxiv later )

***
# 1.Environment:
python3.6.12 ; Tensorflow1.15; MATLAB R2021a
***
# 2.File description:

--**resnet.py**: Define the original networks for CIFAR-10/CIFAR-100/ImageNet

--**train_resnet.py**: Use this code to train the original networks from scratch  for CIFAR-10/CIFAR- 100 or fine-tune the pre-trained networks transferred from Pytorch for ImageNet

--**resnet_decom.py**: Define the decomposed networks for CIFAR-10/CIFAR-100/ImageNet

--**train_resnet_decom.py**: Use this code to fine-tune or train the decomposed networks from scratch.

--**inference_evaluating_time.py**: Test the time for inference/Realistic Acceleartion.
______________________________________________________________________________________________________
--util

----**get_parameter.py**: obtain the factorized weights using decomposition methods

----**pytorch_pth2tensorflow_npy.py**: create a dictionary contains pre-trained Pytorch weights for ResNet34 on ImageNet to transfer the model to TensorFlow

----**cifar10_input.py**: load CIFAR-10

----**cifar100_input.py**: load CIFAR-100

----**dataset.py&image_processing.py&imagenet_data.py**：load ImageNet (TF records format)

______________________________________________________________________________________________________

--draw_pictures

----**draw_Bi_JSVD_curves.m**: (Matlab) draw the Figure 3 and 4

----**draw_time_figure.py**: draw the Figure 5 (a) to (d)

***
# 3.Demo (How to run)
#### 1. train a original network from scratch like this:
```c
python train_resnet.py --model=resnet34 --dataset=cifar10 --from_scratch=True  --bool_regularizer=True --gpu=0 --batch_size=128 --epoch=300 --num_lr=1e-1 change_lr=[140,200,250]  --lr_decay=10
```
#### 2、set a rank rate, choose a method to compress network follow the 'pre-train->decompose->fine-tune' criteria:

```c
python train_resnet_decom.py  --method=lJSVD --model=resnet34 --dataset=cifar100  --repeat_exp_times=3  --batch_size=128 --bool_regularizer=True --exp_path=cifar10_300epoch --from_scratch=False--epoch=300 --num_lr=1e-1 --change_lr="[140,200,250]" --max_to_keep=10 --rank_rate_SVD=0.04
```
#### or train it from scratch:
```c
python train_resnet_decom.py  --method=lJSVD --model=resnet34 --dataset=cifar100  --repeat_exp_times=3  --batch_size=128 --bool_regularizer=True --exp_path=cifar10/from_scratch --from_scratch=True--epoch=300 --num_lr=1e-1 --change_lr="[140,200,250]" --max_to_keep=10 --rank_rate_SVD=0.04
```
***
# 4.Resources

 1. [CIFAR10/CIFAR100 data sets](http://www.cs.toronto.edu/~kriz/cifar.html)
 2. [Pre-trained weights for ResNet34 on ImageNet](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
