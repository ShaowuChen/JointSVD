# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.realpath(__file__+"/../"))
# import image_processing


from dataset import Dataset
import tensorflow as tf

class ImagenetData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset,train_set_number_rate=None):
    super(ImagenetData, self).__init__('ImageNet', subset)
    self.train_set_number_rate = train_set_number_rate

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 1000

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1281167
    if self.subset == 'validation':
      return 50000

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any ImageNet %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the ImageNet data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take several hours depending on the speed of your '
          'computer and network connection\n')
    print('Please see README.md for instructions on how to build '
          'the ImageNet dataset using download_and_preprocess_imagenet.\n')
    print('Note that the raw data size is 300 GB and the processed data size '
          'is 150 GB. Please ensure you have at least 500GB disk space.')

  def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        data_dir = '/home/test01/sambashare/sdd/ImageNet_Dataset_TFRecords'
        tf_record_pattern = os.path.join(data_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if self.subset == 'validation':
            assert (self.train_set_number_rate == None)
        elif self.subset == 'train':
            if self.train_set_number_rate != None:
                if self.train_set_number_rate == 1:
                    pass
                else:
                    data_files = data_files[0:round(len(data_files) * self.train_set_number_rate)]
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                              self.subset,
                                                              data_dir))

            self.download_message()
            exit(-1)
        return data_files


