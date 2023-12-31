/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file op_attr.h
 * \brief
 */

#ifndef OPS_COMMON_INC_OP_ATTR_H_
#define OPS_COMMON_INC_OP_ATTR_H_

#include <vector>
#include "op_log.h"
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"

namespace ops {
using namespace ge;
/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const T& paras, const std::pair<int64_t, std::string>& attr_info, int64_t& value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  if (!AttrUtils::GetInt(op_desc, attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %lld", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const T& paras, const std::pair<int64_t, std::string>& attr_info, int64_t& value,
                  const int64_t default_value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  if (!AttrUtils::GetInt(op_desc, attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %lld", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const T& paras, const std::pair<int64_t, std::string>& attr_info, uint32_t& value,
                  const uint32_t default_value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  if (!AttrUtils::GetInt(op_desc, attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const T& paras, const std::pair<int64_t, std::string>& attr_info, bool& value,
                  const bool default_value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  if (!AttrUtils::GetBool(op_desc, attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const T& paras, const std::pair<int64_t, std::string>& attr_info, bool& value) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(paras);
  if (!AttrUtils::GetBool(op_desc, attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  return true;
}

}  // namespace ops
#endif  // OPS_COMMON_INC_OP_ATTR_H_
