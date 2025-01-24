#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import os
import tensorflow as tf
import numpy as np
from npu_bridge.npu_init import *
tf.enable_resource_variables()

#np.allclose比较函数的相对公差参数
ABSOLUTE_TOL = 0.001
#np.allclose比较函数的绝对公差参数
RELATIVE_TOL = 0.001

def main(unused_argv):
    custom_op_lib = tf.load_op_library(os.path.join("./outputs/libcustom_ops.so")) # 加载自定义算子库
    # 定义输入数据
    shape_params = (8, 2048)
    dtype_params = np.float16

    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    y_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)

    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    y = tf.compat.v1.placeholder(dtype_params, shape=shape_params)

    tf_z = tf.math.add(x, y)
    ac_z = custom_op_lib.add_custom(x, y)    # 调用Ascend C AddCustom自定义算子

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf_golden = sess.run(tf_z, feed_dict={x: x_data, y: y_data})

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ac_golden = sess.run(ac_z, feed_dict={x: x_data, y: y_data})

    # 通过np.allclose函数比较TensorFlow和Ascend C的输出是否一致
    np.array(tf_golden).astype(dtype_params)
    np.array(ac_golden).astype(dtype_params)

    cmp_result = np.allclose(tf_golden, ac_golden, atol=ABSOLUTE_TOL, rtol=RELATIVE_TOL)
    if cmp_result:
        print("The result of tf and ac is the same.")
    else:
        print("The result of tf and ac is different.")


if __name__ == '__main__':
    tf.app.run()