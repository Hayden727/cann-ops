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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import logging
import tensorflow as tf
import numpy as np
import npu_device
npu_device.compat.enable_v1()

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.compat.v1.flags.FLAGS
ABSOLUTE_TOL = 0.001
RELATIVE_TOL = 0.001


def config(excute_type):
    if excute_type == 'ai_core':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["min_group_size"].b = 1

    elif excute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    return session_config


def main(unused_argv):
    shape_params = (8, 2048)
    dtype_params = np.float16
    x_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)
    y_data = np.random.uniform(-2, 2, size=shape_params).astype(dtype_params)

    x = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    y = tf.compat.v1.placeholder(dtype_params, shape=shape_params)
    out = tf.math.add(x, y)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        result_cpu = session.run(out, feed_dict={x: x_data, y: y_data})
    with tf.compat.v1.Session(config=config('ai_core')) as session:
        print("run npu")
        result_ai_core = session.run(out, feed_dict={x: x_data, y: y_data})

    np.array(result_ai_core).astype(dtype_params)
    np.array(result_cpu).astype(dtype_params)

    print('====================================')
    cmp_result = np.allclose(result_ai_core, result_cpu, ABSOLUTE_TOL, RELATIVE_TOL)
    print(cmp_result)
    print('====================================')


if __name__ == "__main__":
    tf.compat.v1.app.run()
