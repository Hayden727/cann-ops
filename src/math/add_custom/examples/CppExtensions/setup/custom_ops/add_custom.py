#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd

import custom_ops_lib


def add_custom(self, other):
    return custom_ops_lib.add_custom(self, other)


def add_custom1(self, other):
    return custom_ops_lib.add_custom1(self, other)
