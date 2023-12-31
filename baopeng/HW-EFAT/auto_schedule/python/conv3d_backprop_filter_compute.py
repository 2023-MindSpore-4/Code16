#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
conv3d backprop filter DSL interface.
"""
import warnings
from tbe.dsl.compute.conv3d_backprop_filter_compute import conv3d_dw as conv3d_dw_tbe


def conv3d_dw(x, out_backprop, filter_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    x : the featuremap data, tvm.placeholder, 6hd shape

    out_backprop : the grads data, tvm.placeholder, 6hd shape

    filter_size : 5-D shape, specifies the filter sizes

    para_dict : dict of parameters
    strides : 3-D shape, specifies in depth, height and width dimension
    pads : 6-D shape, specifies in up/down/left/right dimension
    dilations : 5-D shape, specifies in batch/channel/depth/height/width dimension
    res_dtype : the output data type
    kernel_name : conv3d_backprop_filter_cce by default
    group_dict : group of parameters

    Returns
    -------
    result tensor of conv3d_backprop_filter compute
    """
    warnings.warn("te.lang.cce.te_compute.conv3d_backprop_filter_compute is expired, "
        "please replace it with the func tbe.dsl.compute.conv3d_backprop_filter_compute",
        DeprecationWarning)
    return conv3d_dw_tbe(x, out_backprop, filter_size, para_dict)
