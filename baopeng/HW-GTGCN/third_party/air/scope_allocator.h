/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
#define FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_

#include "graph/op_desc.h"

namespace fe {
class ScopeAllocator {
 public:
  ScopeAllocator();
  virtual ~ScopeAllocator();
  ScopeAllocator(const ScopeAllocator& in) = delete;
  ScopeAllocator& operator = (const ScopeAllocator& in) = delete;

 public:
  void Init();
  int64_t GetCurrentScopeId();
  int64_t AllocateScopeId(void);
  bool HasScopeAttr(ge::ConstOpDescPtr opdef);
  bool GetScopeAttr(ge::ConstOpDescPtr opdef, int64_t &scope_id);
  bool SetScopeAttr(ge::OpDescPtr opdef, int64_t scope_id);
  bool HasL1ScopeAttr(const ge::OpDescPtr &op_desc);
  bool GetL1ScopeAttr(const ge::OpDescPtr &op_desc, int64_t &scope_id);
  bool SetL1ScopeAttr(ge::OpDescPtr &op_desc, const int64_t &scope_id);
  bool ResetScopeId(int64_t scope_id);
 private:
    int64_t scope_id;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
