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
conv3d backprop input tiling case
"""
import warnings


def calc_conv3dbp_input(outs, option=None):
    """
    tiling_case func for dynamic shape conv3d_bp_input

    Parameters
    ----------
    outs : tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    warnings.warn("te.lang.dynamic.schedule.conv3d_bp_input_tilingcase is expired, "
        "please replace it with the func tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase",
        DeprecationWarning)
    from tbe.dsl.unify_schedule.conv3d_bp_input_tilingcase import calc_conv3dbp_input
    if option:
        return calc_conv3dbp_input(outs, option)
    else:
        return calc_conv3dbp_input(outs)
