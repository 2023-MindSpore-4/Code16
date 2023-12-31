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
conv3d_backprop_input schedule
"""
import warnings


def schedule(outs, tiling_case):
    """
    schedule for conv3d backprop input dynamic shape
    """
    warnings.warn("te.lang.dynamic.schedule.conv3d_bp_input_schedule is expired, "
        "please replace it with the func tbe.dsl.unify_schedule.conv3d_bp_input_schedule",
        DeprecationWarning)
    from tbe.dsl.unify_schedule.conv3d_bp_input_schedule import schedule
    return schedule(outs, tiling_case)
