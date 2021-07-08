import matplotlib.pyplot as plt
import numpy as np 
# from google.colab import files

'''
Author: Shaowu Chen
Paper: Joint Matrix Decomposition for Deep Convolutional Neural Networks Compression
Email: shaowu-chen@foxmail.com
'''

x3 = [1,5,9]#orig
x2 = [2,6,10]#CF_small
x1 = [3,7,11]#CF_large


def draw(x1,x2,x3,y1,y2,y3,name):

  plt.bar(x3, y3, width=1, color = '#EDB120', alpha=0.7, label='original')
  for x,y in zip(x3, y3):
      plt.text(x,y+0.007,np.round(y,4),fontsize=8, ha='center',va='bottom')
  plt.bar(x2, y2, width=1, color = '#D95319', alpha=0.7, label='CFs')
  for x,y in zip(x2, y2):
      plt.text(x,y+0.007,np.round(y,4),fontsize=8, ha='center',va='bottom')

  plt.bar(x1, y1, width=1, color = '#0072BD', alpha=0.5, label='CFl')
  for x,y in zip(x1, y1):
      plt.text(x,y+0.007,np.round(y,4),fontsize=8, ha='center',va='bottom')


  plt.tick_params(labelsize=12)
  plt.ylabel('Time (ms)', fontsize = 15)
  # plt.title(name, fontsize = 18)
  plt.xticks([2,6,10],['ResNet-18', 'ResNet-34','ResNet-50'], fontsize = 15)


  plt.ylim(0,0.40)
  legend = plt.legend(shadow=True, fontsize='x-large')
  # plt.legend().get_title().set_fontsize(fontsize = 20)
  plt.tight_layout()
  # plt.savefig(name+'.pdf',dpi=1000)
  # files.download(name+'.pdf') 
  plt.show()
  plt.cla()

'===========LJSVD================='
orig_gpu= [0.10292566299438477,0.1767024087905884,0.36038771152496335]

y1=[0.062397799491882316, 0.09842421054840088, 0.23763404369354246]
y2=[0.0660504674911499, 0.10680055141448974, 0.24929830074310302]
draw(x1,x2,x3,y1,y2,orig_gpu,'LJSVD')

'==========RJSVD-1================='  
y1=[0.0639296817779541, 0.10209775924682618, 0.24058823108673097]
y2=[0.06772093296051025, 0.10912471771240234, 0.2520550012588501]
draw(x1,x2,x3,y1,y2,orig_gpu,'RJSVD-1')

'==========RJSVD-2================='  
y1=[0.0646796417236328, 0.10005036830902098, 0.24183786869049073]
y2=[0.06579039573669435, 0.10753789901733399, 0.25125780582427976]
draw(x1,x2,x3,y1,y2,orig_gpu,'RJSVD-2')


'==========Bi-JSVD=================' 
y1=[0.07357218265533447, 0.13133941650390626, 0.3242643308639527]
y2=[0.07453553199768065, 0.1377725839614868, 0.3334836149215698]
draw(x1,x2,x3,y1,y2,orig_gpu,'Bi-JSVD0.5')




