# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:13:41 2018

@author: murata
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# Variables
shape = (128,128,128)
class_num = 4
x = tf.placeholder("float", [None, list(shape)])
y_ = tf.placeholder("float", [None, class_num])
