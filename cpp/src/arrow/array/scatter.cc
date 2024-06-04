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
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/logging.h"
#include "arrow/visit_data_inline.h"
#include "arrow/visit_type_inline.h"

namespace arrow {

namespace {

Status ScatterBitmap(const uint8_t* in_bitmap, const uint8_t* mask_bitmap,
                     uint8_t* out_bitmap, int64_t length) {
  int64_t i_in = 0, i_out = 0;
  VisitNullBitmapInline(
      mask_bitmap, /*valid_bits_offset=*/0, length, kUnknownNullCount,
      [&] {
        ::arrow::bit_util::SetBitTo(out_bitmap, i_out++,
                                    ::arrow::bit_util::GetBit(in_bitmap, i_in++));
      },
      [&] { ++i_out; });

  return Status::OK();
}

struct ScatterImpl {
  explicit ScatterImpl(const ArrayData& in, const BooleanArray& mask, MemoryPool* pool)
      : in_(in), mask_(mask), pool_(pool), out_(std::shared_ptr<ArrayData>()) {
    out_->type = in_.type;
    out_->length = mask_.length();
    out_->buffers.resize(in_.buffers.size());
    out_->child_data.resize(in_.child_data.size());
    for (auto& data : out_->child_data) {
      data = std::make_shared<ArrayData>();
    }
  }

  Status Scatter(std::shared_ptr<ArrayData>* out) && {
    RETURN_NOT_OK(MakeValidityBitmap());
    RETURN_NOT_OK(VisitTypeInline(*out_->type, this));
    *out = std::move(out_);
    return Status::OK();
  }

  Status MakeValidityBitmap() {
    if (internal::may_have_validity_bitmap(in_.type->id())) {
      return Status::OK();
    }

    if (!mask_.null_bitmap() && !in_.HasValidityBitmap()) {
      out_->null_count = mask_.length() - mask_.true_count();
      out_->buffers[0] =
          SliceBuffer(mask_.data()->buffers[1], mask_.offset(), mask_.length());
      return Status::OK();
    }

    out_->null_count = kUnknownNullCount;
    ARROW_ASSIGN_OR_RAISE(out_->buffers[0], AllocateBitmap(mask_.length(), pool_));
    ::arrow::internal::CopyBitmap(mask_.data()->buffers[1]->data(), mask_.offset(),
                                  mask_.length(), out_->buffers[0]->mutable_data(), 0);
    if (mask_.null_bitmap()) {
      ::arrow::internal::BitmapAnd(out_->buffers[0]->data(), 0,
                                   mask_.data()->buffers[0]->data(), mask_.offset(),
                                   mask_.length(), 0, out_->buffers[0]->mutable_data());
    }
    if (in_.HasValidityBitmap()) {
      RETURN_NOT_OK(ScatterBitmap(in_.buffers[0]->data(), out_->buffers[0]->data(),
                                  out_->buffers[0]->mutable_data(), out_->length));
    }

    return Status::OK();
  }

  Status Visit(const NullType&) { return Status::OK(); }

  Status Visit(const BooleanType&) {
    RETURN_NOT_OK(ScatterBitmap(in_.buffers[1]->data(), out_->buffers[0]->data(),
                                out_->buffers[1]->mutable_data(), out_->length));
    return Status::OK();
  }

  Status Visit(const DataType&) {
    return Status::NotImplemented("Scatter not implemented for type ", *out_->type);
  }

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

  auto true_count = mask.true_count();
  DCHECK_GE(in.length, true_count);
  if (true_count == 0) {
    return MakeArrayOfNull(in.type, mask.length(), pool);
  } else if (true_count == mask.length()) {
    return MakeArray(in.Slice(0, mask.length()));
  }

  std::shared_ptr<ArrayData> out_data;
  RETURN_NOT_OK(ScatterImpl(in, mask, pool).Scatter(&out_data));
  return MakeArray(std::move(out_data));
}

}  // namespace arrow
