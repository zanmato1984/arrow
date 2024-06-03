// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/array/scatter.h"
#include "arrow/array.h"
#include "arrow/array/data.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"

namespace arrow {

namespace {

class ScatterImpl {
 public:
  ScatterImpl(const ArrayData& in, const BooleanArray& mask, MemoryPool* pool)
      : in_(in), mask_(mask), pool_(pool), out_(std::shared_ptr<ArrayData>()) {
    DCHECK_GE(in_.length, mask.length());
    out_->type = in_.type;
    out_->length = mask_.length();
    out_->buffers.resize(in_.buffers.size());
    out_->child_data.resize(in_.child_data.size());
    for (auto& data : out_->child_data) {
      data = std::make_shared<ArrayData>();
    }
  }

  Status Scatter(std::shared_ptr<ArrayData>* out) && {
    // out_->null_count = kUnknownNullCount;

    *out = std::move(out_);
    return Status::OK();
  }

 private:
  const ArrayData& in_;
  const BooleanArray& mask_;
  MemoryPool* pool_;
  std::shared_ptr<ArrayData> out_;
};

}  // namespace

Result<std::shared_ptr<Array>> Scatter(const ArrayData& in, const BooleanArray& mask,
                                       MemoryPool* pool) {
  if (mask.length() == 0) {
    return MakeEmptyArray(in.type, pool);
  }

  std::shared_ptr<ArrayData> out_data;
  RETURN_NOT_OK(ScatterImpl(in, mask, pool).Scatter(&out_data));
  return MakeArray(std::move(out_data));
}

}  // namespace arrow
