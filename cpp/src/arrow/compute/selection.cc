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

#include "arrow/compute/selection_internal.h"

#include <algorithm>
#include <utility>

#include "arrow/array/array_base.h"
#include "arrow/buffer.h"
#include "arrow/compute/api_vector.h"
#include "arrow/type.h"
#include "arrow/util/bitmap_ops.h"

namespace arrow::compute::detail {

using ::arrow::internal::BitmapAnd;
using ::arrow::internal::CopyBitmap;

namespace {

int64_t CountSelected(const ContiguousSpan& span) { return span.length; }
int64_t CountSelected(const DiscreteSpan& span) { return span.length; }
int64_t CountSelected(const FilteredSpan& span) {
  DCHECK_NE(span.bitmap, nullptr);
  return ::arrow::internal::CountSetBits(span.bitmap, span.bitmap_offset, span.length);
}
}  // namespace

int64_t SelectedCount(const SelectionSpan& selection) {
  return std::visit([](const auto& span) { return CountSelected(span); }, selection);
}

void SetSpanAllNull(ArraySpan* out) {
  out->null_count = out->length;
  if (!out->type->layout().buffers.empty() &&
      out->type->layout().buffers[0].kind == DataTypeLayout::BITMAP &&
      out->buffers[0].data != nullptr) {
    bit_util::SetBitsTo(out->buffers[0].data, out->offset, out->length, false);
  }
}

namespace {

void ClearUnselectedValidity(const ContiguousSpan& span, int64_t base, int64_t len,
                             uint8_t* out_bitmap) {
  if (span.start_offset > 0) {
    bit_util::SetBitsTo(out_bitmap, base, span.start_offset, false);
  }
  const int64_t end = span.start_offset + span.length;
  if (end < len) {
    bit_util::SetBitsTo(out_bitmap, base + end, len - end, false);
  }
}

void ClearUnselectedValidity(const FilteredSpan& span, int64_t base, int64_t len,
                             uint8_t* out_bitmap) {
  DCHECK_NE(span.bitmap, nullptr);
  ClearUnselectedValidity(ContiguousSpan{span.start_offset, span.length}, base, len,
                          out_bitmap);
  BitmapAnd(out_bitmap, base + span.start_offset, span.bitmap, span.bitmap_offset,
            span.length, base + span.start_offset, out_bitmap);
}

void ClearUnselectedValidity(const DiscreteSpan& span, int64_t base, int64_t len,
                             uint8_t* out_bitmap) {
  // Clear only non-selected ranges, preserving selected validity.
  int64_t cursor = 0;
  for (int64_t i = 0; i < span.length; ++i) {
    const int64_t idx = span[i];
    DCHECK_GE(idx, 0);
    DCHECK_LT(idx, len);
    if (idx > cursor) {
      bit_util::SetBitsTo(out_bitmap, base + cursor, idx - cursor, false);
    }
    cursor = idx + 1;
  }
  if (cursor < len) {
    bit_util::SetBitsTo(out_bitmap, base + cursor, len - cursor, false);
  }
}

void ClearUnselectedValidity(const SelectionSpan& selection, int64_t base, int64_t len,
                             uint8_t* out_bitmap) {
  std::visit(
      [&](const auto& span) { ClearUnselectedValidity(span, base, len, out_bitmap); },
      selection);
}

}  // namespace

bool HasValidityBitmap(const DataType& type) {
  const auto& buffers = type.layout().buffers;
  return !buffers.empty() && buffers[0].kind == DataTypeLayout::BITMAP;
}

namespace {

Status MaskSpan(const SelectionSpan& selection, ArraySpan* out) {
  if (out->length == 0 || SelectedCount(selection) == out->length ||
      out->type->id() == Type::NA)
    return Status::OK();
  if (!HasValidityBitmap(*out->type)) {
    return Status::NotImplemented("Selection masking for ", out->type->ToString());
  }
  DCHECK_NE(out->buffers[0].data, nullptr);
  ClearUnselectedValidity(selection, out->offset, out->length, out->buffers[0].data);
  out->null_count = kUnknownNullCount;
  return Status::OK();
}

class IndexSelectionCursor : public SelectionCursor {
 public:
  explicit IndexSelectionCursor(std::shared_ptr<ArrayData> data)
      : data_(std::move(data)) {}
  SelectionSpan Next(int64_t chunk_start, int64_t chunk_end) override {
    DCHECK_GE(chunk_start, 0);
    DCHECK_LE(chunk_start, chunk_end);
    if (position_ == data_->length) return ContiguousSpan{};
    const int32_t* indices = data_->GetValues<int32_t>(1);
    const int32_t* begin = indices + position_;
    const int32_t* end = indices + data_->length;
    const int32_t* limit = std::lower_bound(
        begin, end, chunk_end,
        [](int32_t index, int64_t end) { return static_cast<int64_t>(index) < end; });
    const int64_t length = limit - begin;
    position_ += length;
    if (length == 0) return ContiguousSpan{};
    DCHECK_GE(static_cast<int64_t>(begin[0]), chunk_start);
    if (static_cast<int64_t>(begin[length - 1]) - begin[0] == length - 1) {
      return ContiguousSpan{begin[0] - chunk_start, length};
    }
    return DiscreteSpan{begin, length, chunk_start};
  }
  int64_t position() const override { return position_; }

 private:
  std::shared_ptr<ArrayData> data_;
  int64_t position_ = 0;
};

class IndexSelectionBackend : public SelectionBackend {
 public:
  explicit IndexSelectionBackend(std::shared_ptr<ArrayData> data)
      : data_(std::move(data)) {
    DCHECK_NE(data_, nullptr);
    DCHECK_EQ(data_->type->id(), Type::INT32);
    indices_ = data_->GetValues<int32_t>(1);
  }
  int64_t length() const override { return data_->length; }
  Status Validate(int64_t values_length) const override {
    if (data_ == nullptr) {
      return Status::Invalid("SelectionVector not initialized");
    }
    ARROW_CHECK_NE(indices_, nullptr);
    if (data_->type->id() != Type::INT32) {
      return Status::Invalid("SelectionVector indices must be of type int32");
    }
    if (data_->GetNullCount() != 0) {
      return Status::Invalid("SelectionVector indices cannot contain nulls");
    }
    for (int64_t i = 1; i < length(); ++i) {
      if (indices_[i - 1] >= indices_[i]) {
        return Status::Invalid("SelectionVector indices must be strictly increasing");
      }
    }
    for (int64_t i = 0; i < length(); ++i) {
      if (indices_[i] < 0) {
        return Status::Invalid("SelectionVector indices must be non-negative");
      }
    }
    if (values_length >= 0) {
      for (int64_t i = 0; i < length(); ++i) {
        if (indices_[i] >= values_length) {
          return Status::Invalid("SelectionVector index ", indices_[i],
                                 " >= values length ", values_length);
        }
      }
    }
    return Status::OK();
  }

  std::unique_ptr<SelectionCursor> MakeCursor() const override {
    return std::make_unique<IndexSelectionCursor>(data_);
  }
  Result<Datum> Gather(const Datum& values, ExecContext* ctx) const override {
    return Take(values, Datum(data_), TakeOptions{/*boundscheck=*/false}, ctx);
  }
  Result<Datum> Scatter(const Datum& values, int64_t output_length,
                        ExecContext* ctx) const override {
    return compute::Scatter(values, Datum(data_),
                            ScatterOptions{/*max_index=*/output_length - 1}, ctx);
  }

 private:
  std::shared_ptr<ArrayData> data_;
  const int32_t* indices_;
};

}  // namespace

Status ApplySelectionMask(const SelectionSpan& selection, ExecResult* out,
                          MemoryPool* pool) {
  if (out->is_array_span()) return MaskSpan(selection, out->array_span_mutable());
  auto& data = std::get<std::shared_ptr<ArrayData>>(out->value);
  if (data->length == 0 || SelectedCount(selection) == data->length ||
      data->type->id() == Type::NA)
    return Status::OK();
  if (!HasValidityBitmap(*data->type)) {
    return Status::NotImplemented("Selection masking for ", data->type->ToString());
  }
  if (data.use_count() != 1) data = data->Copy();
  const int64_t bit_length = data->offset + data->length;
  auto& bitmap = data->buffers[0];
  if (!bitmap) {
    ARROW_ASSIGN_OR_RAISE(bitmap, AllocateEmptyBitmap(bit_length, pool));
    bit_util::SetBitsTo(bitmap->mutable_data(), 0, bit_length, true);
  } else if (bitmap.use_count() != 1 || !bitmap->is_mutable() || bitmap->parent()) {
    ARROW_ASSIGN_OR_RAISE(auto writable, AllocateEmptyBitmap(bit_length, pool));
    CopyBitmap(bitmap->data(), 0, bit_length, writable->mutable_data(), 0);
    bitmap = std::move(writable);
  }
  ArraySpan span(*data);
  RETURN_NOT_OK(MaskSpan(selection, &span));
  data->null_count = span.null_count;
  return Status::OK();
}

std::shared_ptr<SelectionVector> SelectionBackend::Make(
    std::shared_ptr<SelectionBackend> impl) {
  return std::shared_ptr<SelectionVector>(new SelectionVector(std::move(impl)));
}
const SelectionBackend& SelectionBackend::Get(const SelectionVector& selection) {
  return *selection.backend_;
}

}  // namespace arrow::compute::detail

namespace arrow::compute {

std::shared_ptr<SelectionVector> SelectionVector::MakeIndices(
    std::shared_ptr<ArrayData> data) {
  return detail::SelectionBackend::Make(
      std::make_shared<detail::IndexSelectionBackend>(std::move(data)));
}
std::shared_ptr<SelectionVector> SelectionVector::MakeIndices(const Array& arr) {
  return MakeIndices(arr.data());
}
int64_t SelectionVector::length() const { return backend_->length(); }
Status SelectionVector::Validate(int64_t values_length) const {
  return backend_->Validate(values_length);
}

}  // namespace arrow::compute
