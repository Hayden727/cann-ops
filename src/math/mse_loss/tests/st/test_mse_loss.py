#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import torch
import numpy as np
import torch.nn.functional as F

def mse_loss_test(predict, label):
    predict_tensor = torch.from_numpy(predict)
    label_etnsor = torch.from_numpy(label)
    reduction = "mean"
    golden = F.mse_loss(predict_tensor, label_etnsor, reduction=reduction)
    return golden.numpy()


def calc_expect_func(predict, label, y):
    res = mse_loss_test(predict["value"], label["value"])
    return [res]