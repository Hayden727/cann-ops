import os
import logging
# import torch
import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
import npu_device
from npu_device.compat.v1.npu_init import *

logging.getLogger().setLevel(logging.INFO)

os.environ["DEVICE_ID"] = str(0)
os.environ["ASCEND_DEVICE_ID"] = str(0)
os.environ["JOB_ID"] = "10089"

tf.compat.v1.disable_eager_execution()

npu_device.compat.enable_v1()
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

ROW_LEN = 25
LAST_DIM = 32 * 1024 # 该算子支持 <= 16k 以及 刚好32k 的尾轴维度，且需要256字节对齐

x1__ = np.random.randn(ROW_LEN, LAST_DIM).astype(np.float16)
x2__ = np.random.randn(1, LAST_DIM // 128, 128).astype(np.float16)
t1_ = float(0.3)
t2_ = float(0.1)
t3_ = float(0.8)

x1_ = tf.Variable(x1__, tf.float16)
x2_ = tf.Variable(x2__, tf.float16)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_gold(x1, x2, t1, t2, t3):
    tmp = sigmoid(x1*t1)
    zero = np.zeros_like(tmp)
    sel = np.where(tmp < t2, tmp, 2*tmp)
    sel = sel.reshape(-1, LAST_DIM) *  x2.reshape(1, LAST_DIM)
    res = sel * t3
    return res.reshape(res.shape[0], LAST_DIM // 128, 128).astype(np.float16)

def cosine_distance(x, y):
    x = x.reshape(-1, LAST_DIM)
    y = y.reshape(-1, LAST_DIM)
    return np.sum(x * y, -1) / (np.linalg.norm(x, -1) * np.linalg.norm(y, -1))

tfOpLib = tf.load_op_library("/data00/zilu/tf_ops/op_lib/libxpu_tfops.so")
output = tfOpLib.MulSigmoid(x1=x1_, x2=x2_, t1=t1_, t2=t2_, t3=t3_)

gold = get_gold(x1__, x2__, t1_, t2_, t3_)
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    out= sess.run(output)
    print("allclose: ", np.allclose(out, gold, 1e-2))
    