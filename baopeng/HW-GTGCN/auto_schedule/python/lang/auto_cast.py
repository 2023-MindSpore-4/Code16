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
auto_cast
"""
from decorator import decorator

from te import tvm
from te.platform import intrinsic_check_support
from te.utils.error_manager.error_manager_util import get_error_message
from te.platform import get_soc_spec


@decorator
def auto_cast_of_elewise(func, *args, **kwargs):
    """
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    """
    from tbe.dsl.compute.cast import _cast
    from tbe.dsl.compute.util import is_cast_support
    from tbe.dsl.compute.util import judge_var
    from tbe.dsl.compute.util import in_dynamic_and_static_unify
    # dynamic not support auto_cast
    if in_dynamic_and_static_unify():
        return func(*args, **kwargs)

    def _check_args_type(args):
        if len(args) in (1, 2, 3):
            if not isinstance(args[0], tvm.tensor.Tensor):
                dict_args = {
                    "errCode": "E90001",
                    "detailed_cause": f"The first input type must be [tvm.tensor], while type is [{type(args[0])}]"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if len(args) == 3:
                if not isinstance(args[1], tvm.tensor.Tensor):
                    dict_args = {
                        "errCode": "E90001",
                        "detailed_cause":
                            f"The second input type must be [tvm.tensor], while type is [{type(args[0])}]"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))

    _check_args_type(args)

    intr = func.__name__
    intr = _intrinsic_check(intr)

    is_support_fp32 = intrinsic_check_support("Intrinsic_"+intr, "float32")
    if len(args) == 1:
        def _cast_one_input_tensor(args, intr, is_support_fp32):
            temp_tensor = args[0]
            dtype = temp_tensor.dtype
            is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
            if not is_support_dtype:
                if is_support_fp32 and is_cast_support(dtype, "float32"):
                    temp_tensor = _cast(temp_tensor, "float32")
                else:
                    temp_tensor = _cast(temp_tensor, "float16")

            return temp_tensor

        temp_tensor = _cast_one_input_tensor(args, intr, is_support_fp32)
        return func(temp_tensor)
    if len(args) == 2:
        if isinstance(args[1], tvm.tensor.Tensor):
            def _cast_two_input_tensor(args, intr, is_support_fp32):
                lhs = args[0]
                rhs = args[1]
                dtype_l = lhs.dtype
                dtype_r = rhs.dtype

                lhs_t = lhs
                rhs_t = rhs
                is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr,
                                                            dtype_l)
                is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr,
                                                            dtype_r)
                if not is_support_ldtype \
                        or not is_support_rdtype or dtype_l != dtype_r:
                    if is_support_fp32 \
                            and is_cast_support(dtype_l, "float32") \
                            and is_cast_support(dtype_r, "float32"):
                        lhs_t = _cast(lhs, "float32")
                        rhs_t = _cast(rhs, "float32")
                    else:
                        lhs_t = _cast(lhs, "float16")
                        rhs_t = _cast(rhs, "float16")

                return lhs_t, rhs_t

            lhs_t, rhs_t = _cast_two_input_tensor(args, intr, is_support_fp32)
            return func(lhs_t, rhs_t)

        def _cast_tensor_scalar_two_input(args, intr, is_support_fp32):
            temp_tensor = args[0]
            scalar = args[1]
            dtype = temp_tensor.dtype
            is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
            if not is_support_dtype:
                if is_support_fp32 \
                        and is_cast_support(dtype, "float32"):
                    temp_tensor = _cast(temp_tensor, "float32")
                    dtype = "float32"
                else:
                    temp_tensor = _cast(temp_tensor, "float16")
                    dtype = "float16"

            tmp_arg = scalar
            scalar_type = judge_var(scalar)
            if scalar_type == "tvm_const" and scalar.dtype != dtype:
                tmp_arg = tvm.const(scalar.value, dtype=dtype)

            if scalar_type == "python_const":
                tmp_arg = tvm.const(scalar, dtype=dtype)

            return temp_tensor, tmp_arg

        temp_tensor, tmp_arg = _cast_tensor_scalar_two_input(args, intr, is_support_fp32)
        return func(temp_tensor, tmp_arg)
    if len(args) == 3:
        if isinstance(args[2], tvm.tensor.Tensor):
            def _cast_three_input_tensor(args, intr, is_support_fp32):
                tensor_0 = args[0]
                tensor_1 = args[1]
                tensor_2 = args[2]

                dtype_0 = tensor_0.dtype
                dtype_1 = tensor_1.dtype
                dtype_2 = tensor_2.dtype

                tensor_0_t = tensor_0
                tensor_1_t = tensor_1
                tensor_2_t = tensor_2

                if dtype_0 != dtype_1 or dtype_0 != dtype_2 or dtype_2 != dtype_1:
                    dict_args = {
                        "errCode": "E90001",
                        "detailed_cause": f"Input tensors must has same dtype! while dtype_0 is [{dtype_0}], "
                                          f"dtype_1 is [{dtype_1}], dtype_2 is [{dtype_2}]"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))

                is_support_dtype0 = intrinsic_check_support("Intrinsic_"+intr,
                                                            dtype_0)
                if not is_support_dtype0:
                    if is_support_fp32 \
                            and is_cast_support(dtype_0, "float32"):
                        tensor_0_t = _cast(tensor_0, "float32")
                        tensor_1_t = _cast(tensor_1, "float32")
                        tensor_2_t = _cast(tensor_2, "float32")
                    else:
                        tensor_0_t = _cast(tensor_0, "float16")
                        tensor_1_t = _cast(tensor_1, "float16")
                        tensor_2_t = _cast(tensor_2, "float16")

                return tensor_0_t, tensor_1_t, tensor_2_t

            tensor_0_t, tensor_1_t, tensor_2_t = \
                _cast_three_input_tensor(args, intr, is_support_fp32)
            return func(tensor_0_t, tensor_1_t, tensor_2_t)

        def _cast_tensors_scalar_in_three_input(args, intr, is_support_fp32):
            lhs = args[0]
            rhs = args[1]
            scalar = args[2]
            dtype_l = lhs.dtype
            dtype_r = rhs.dtype

            lhs_t = lhs
            rhs_t = rhs
            is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr, dtype_l)
            is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr, dtype_r)
            if not is_support_ldtype \
                    or not is_support_rdtype or dtype_l != dtype_r:
                if is_support_fp32 \
                        and is_cast_support(dtype_l, "float32") \
                        and is_cast_support(dtype_r, "float32"):
                    lhs_t = _cast(lhs, "float32")
                    rhs_t = _cast(rhs, "float32")
                    dtype_l = "float32"
                else:
                    lhs_t = _cast(lhs, "float16")
                    rhs_t = _cast(rhs, "float16")
                    dtype_l = "float16"

            tmp_arg = scalar
            if not isinstance(tmp_arg, str):
                scalar_type = judge_var(scalar)
                if scalar_type == "tvm_const" and scalar.dtype != dtype_l:
                    tmp_arg = tvm.const(scalar.value, dtype=dtype_l)

                if scalar_type == "python_const":
                    tmp_arg = tvm.const(scalar, dtype=dtype_l)

            return lhs_t, rhs_t, tmp_arg

        lhs_t, rhs_t, tmp_arg = \
            _cast_tensors_scalar_in_three_input(args, intr, is_support_fp32)
        return func(lhs_t, rhs_t, tmp_arg)
    return func(*args, **kwargs)


def _intrinsic_check(intr):
    ret_intr = intr
    if not intrinsic_check_support("Intrinsic_" + intr):
        if intr == "vdiv":
            ret_intr = "vrec"
        elif intr == "vsqrt":
            ret_intr = "vrsqrt"
        elif intr == "vlog":
            ret_intr = "vln"
        elif intr == "vmaxs":
            ret_intr = "vmax"
        elif intr == "vmins":
            ret_intr = "vmin"

    return ret_intr


@decorator
def auto_cast_of_reduce(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    Only static shape support auto cast.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    from tbe.dsl.compute.util import reduce_axis_check
    from tbe.dsl.compute.util import auto_cast_tensor
    from tbe.dsl.compute.util import dsl_support_dtype
    from tbe.dsl.compute.util import in_dynamic_and_static_unify
    intr = func.__name__

    if intr == "sum":
        intr = "reduce_sum"

    def _is_last_axis(shape, axis):
        local_axis = []
        for i in axis:
            new_axis = i
            if i < 0:
                new_axis = i + len(shape)
            local_axis.append(new_axis)

        return len(shape) - 1 in local_axis

    def _check_dynamic_dtype(raw_tensor, intr, supported_dtypes, is_last_axis):
        """
        check dtype for dynamic shape
        """
        if not is_last_axis:
            supported_dtypes.append("int32")

        dtype = raw_tensor.dtype

        if dtype not in supported_dtypes:
            soc_ver = get_soc_spec("SOC_VERSION")
            dict_args = {
                "errCode": "E90002",
                "detailed_cause": f"[{intr}] do not support [{dtype}] in [{soc_ver}] !"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    if len(args) == 3 or len(args) == 4:
        if not isinstance(args[0], tvm.tensor.Tensor):
            dict_args = {
                "errCode": "E90001",
                "detailed_cause": f"The first input type must be [tvm.tensor], while type is [{type(args[0])}]"}

        raw_tensor = args[0]
        axis = args[1]
        keepdims = args[2]
        priority_flag = False
        if len(args) == 4:
            priority_flag = args[3]

        if isinstance(axis, (tuple, list)):
            axis = axis
        else:
            axis = [axis]

        shape_len = len(raw_tensor.shape)
        axis = reduce_axis_check(shape_len, axis)

        is_last_axis = _is_last_axis(raw_tensor.shape, axis)

        supported_dtypes = dsl_support_dtype(intr)
        if not supported_dtypes:
            dict_args = {"errCode": "E90002", "detailed_cause": f"[{intr}] is not supported!"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        # dynamic shape do not perform auto cast
        if in_dynamic_and_static_unify():
            _check_dynamic_dtype(raw_tensor, intr, supported_dtypes, is_last_axis)
            return func(raw_tensor, axis, keepdims)

        # 1. reduce_max/min last v100 with priority_flag
        #    or v200, support float32
        vcmax_support_fp32 = intrinsic_check_support("Intrinsic_vcmax",
                                                     "float32")
        support_fp32 = (vcmax_support_fp32 or priority_flag)
        if intr in ("reduce_max", "reduce_min") and \
                is_last_axis and (not support_fp32):
            supported_dtypes = list(set(supported_dtypes) - set(("float32",)))

        # 2. reduce_max/min/sum nlst support int32
        if intr in ("reduce_max", "reduce_min", "reduce_sum") and \
                (not is_last_axis):
            supported_dtypes.append("int32")

        temp_tensor = auto_cast_tensor(raw_tensor, intr, supported_dtypes)

        return func(temp_tensor, axis, keepdims)

    return func(*args, **kwargs)


@decorator
def auto_cast_of_cast(func, *args, **kwargs):
    '''
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    Only static shape support auto cast.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    '''
    from tbe.dsl.compute.util import auto_cast_tensor
    from tbe.dsl.compute.util import dsl_support_dtype
    from tbe.dsl.compute.util import in_dynamic_and_static_unify
    if in_dynamic_and_static_unify():
        return func(*args, **kwargs)
    intr = func.__name__

    if len(args) == 1:
        if not isinstance(args[0], tvm.tensor.Tensor):
            dict_args = {
                "errCode": "E90001",
                "detailed_cause": f"The first input type must be [tvm.tensor], while type is [{type(args[0])}]"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        raw_tensor = args[0]

        supported_dtypes = dsl_support_dtype(intr)
        if not supported_dtypes:
            dict_args = {"errCode": "E90002", "detailed_cause": f"[{intr}] is not supported!"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        temp_tensor = auto_cast_tensor(raw_tensor, intr, supported_dtypes)

        return func(temp_tensor)

    return func(*args, **kwargs)
