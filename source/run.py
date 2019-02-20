#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@project:Explaining-and-Harnessing-Adversarial-Examples
@author: Kevin Leung
@file:run.py
@interpreter: Python3.6
@ide:PyCharm
@time:2019-02-20 16:58:49
"""
import numpy as np
import tensorflow as tf

in_put = np.random.random(100)
in_put = np.reshape(in_put, (5, 5, 4))
out = tf.slice(in_put, [0, 2, 0], [1, -1, -1])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(out))
