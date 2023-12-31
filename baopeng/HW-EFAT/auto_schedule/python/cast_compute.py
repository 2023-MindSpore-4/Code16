#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
cast compute
"""
# 'pylint: disable=import-error
import warnings
from decorator import decorator
from te import tvm
from te.platform import intrinsic_check_support
from te.utils.error_manager.error_manager_util import get_error_message
from te.utils.shape_util import shape_to_list
from ..api import ceil
from ..api import floor
from ..api import round
from ..api import trunc
from ..api import round_half_up
from .util import auto_cast_tensor
from .util import is_cast_support
from .util import get_cast_type
from .util import check_input_tensor_shape


NAME_INDEX = [0]


def _cast(raw_tensor, dst_dtype, is_auto_cast=True):
    """
    cast tensor from src_type to dst_dtype, only support float32 to float16,
    float16 to float32, float16 to int8, int8 to float16,
    float16 to uint8, uint8 to float16 when shape is dynamic

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    dst_dtype : destinatin type

    Returns
    -------
    wrapped_tensor : casted tensor
    """
    src_dtype = raw_tensor.dtype
    if dst_dtype.lower() == src_dtype.lower():
        return raw_tensor

    if not is_cast_support(src_dtype.lower(), dst_dtype.lower()):
        if is_cast_support(src_dtype.lower(), "float32") and is_cast_support(
                "float32",
                dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float32", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        elif is_cast_support(src_dtype.lower(), "float16") and is_cast_support(
                "float16",
                dst_dtype.lower()):
            raw_tensor = _cast_op(raw_tensor, "float16", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        elif not intrinsic_check_support("Intrinsic_vconv", "deq") and \
                intrinsic_check_support("Intrinsic_vcbd", "s322s16"):
            raw_tensor = _cast_op(raw_tensor, "int16", 'elewise_single_cast')
            src_dtype = raw_tensor.dtype
        else:
            dict_args = {
                "errCode": "E90002",
                "detailed_cause": f"Unsupported cast type! src_dtype is [{src_dtype.lower()}], "
                                  f"dst_dtype is [{dst_dtype.lower()}]"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    # Default cast_type is "cast", while float casting to int32 or int16 is "trunc"
    # float to int8 or uint8 doesn't need to use "trunc" mode
    # Reason: TF uses "trunc" strategy in its cast operator, so maintain consistency with it
    cast_type = "cast"
    if "int" in dst_dtype and "float" in src_dtype and \
            intrinsic_check_support("Intrinsic_vconv",
                                    get_cast_type(src_dtype, dst_dtype) + "z"):
        cast_type = "trunc"
    return _cast_op(raw_tensor, dst_dtype, 'elewise_single_' + cast_type,
                    is_auto_cast=is_auto_cast)


def _cast_op(input_tensor, output_dtype, op_type, is_auto_cast=True):
    """
    factory method of single elewise operations
    """
    tensor = input_tensor
    shape = shape_to_list(tensor.shape)
    check_input_tensor_shape(shape)
    if op_type == "elewise_single_cast":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_round":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_ceil":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_floor":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_trunc":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    elif op_type == "elewise_single_round_d":
        lambda_func = lambda *indice: tensor(*indice).astype(output_dtype)
    else:
        dict_args = {"errCode": "E90003", "detailed_cause": f"operation {op_type} not support yet."}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    name = op_type.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if not is_auto_cast:
        op_type = op_type + "|not_auto_cast"

    with tvm.tag_scope(op_type):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp
