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

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "arrow/compute/exec.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/logging_internal.h"

namespace arrow::compute::detail {

// Each execution owns a cursor. Representation-specific traversal state never leaks
// into the iterator over input chunks, and the selection can be reused independently.
class ARROW_EXPORT SelectionCursor {
 public:
  virtual ~SelectionCursor() = default;
  // Consume successive, non-overlapping input windows. Returned offsets are relative
  // to chunk_start; span storage is borrowed from the selection/cursor.
  virtual SelectionSpan Next(int64_t chunk_start, int64_t chunk_end) = 0;
  virtual int64_t position() const = 0;
};

// Internal backend protocol. Scalar handling and batch orchestration belong to the
// dense adapter; backends gather/scatter only array-like values.
class ARROW_EXPORT SelectionBackend {
 public:
  virtual ~SelectionBackend() = default;
  static std::shared_ptr<SelectionVector> Make(std::shared_ptr<SelectionBackend> impl);
  static const SelectionBackend& Get(const SelectionVector& selection);
  virtual int64_t length() const = 0;
  virtual Status Validate(int64_t values_length) const = 0;
  virtual std::unique_ptr<SelectionCursor> MakeCursor() const = 0;
  virtual Result<Datum> Gather(const Datum& values, ExecContext* ctx) const = 0;
  virtual Result<Datum> Scatter(const Datum& values, int64_t output_length,
                                ExecContext* ctx) const = 0;
};

ARROW_EXPORT int64_t SelectedCount(const SelectionSpan& selection);
inline bool IsEmptySelection(const SelectionSpan* selection) {
  return selection && SelectedCount(*selection) == 0;
}
ARROW_EXPORT bool HasValidityBitmap(const DataType& type);
ARROW_EXPORT void SetSpanAllNull(ArraySpan* out);

// Operates on the actual post-kernel result, including replacement/borrowed outputs.
// ArraySpan outputs are executor-owned preallocation. ArrayData metadata and bitmaps
// may be shared with inputs and must be made independently writable before masking.
ARROW_EXPORT Status ApplySelectionMask(const SelectionSpan& selection, ExecResult* out,
                                       MemoryPool* pool);

// The bool-returning primitive preserves early exit without requiring Status in the
// hot loop. Kernels with range-aware implementations can visit SelectionSpan directly.
template <typename OnIndex>
bool VisitSelectedIndicesWhile(const ContiguousSpan& span, OnIndex&& on_index) {
  for (int64_t i = 0; i < span.length; ++i) {
    if (!on_index(span.start_offset + i)) return false;
  }
  return true;
}

template <typename OnIndex>
bool VisitSelectedIndicesWhile(const FilteredSpan& span, OnIndex&& on_index) {
  DCHECK_NE(span.bitmap, nullptr);
  for (int64_t i = 0; i < span.length; ++i) {
    if (bit_util::GetBit(span.bitmap, span.bitmap_offset + i) &&
        !on_index(span.start_offset + i))
      return false;
  }
  return true;
}

template <typename OnIndex>
bool VisitSelectedIndicesWhile(const DiscreteSpan& span, OnIndex&& on_index) {
  for (int64_t i = 0; i < span.length; ++i) {
    if (!on_index(span[i])) return false;
  }
  return true;
}

// Convenience fallback for kernels that operate one selected row at a time.
// This does not replace direct variant dispatch for bulk contiguous execution.
template <typename OnSelectionFn>
std::invoke_result_t<OnSelectionFn&, int64_t> VisitSelectionSpanInline(
    const SelectionSpan& selection, OnSelectionFn&& on_selection) {
  using Return = std::invoke_result_t<OnSelectionFn&, int64_t>;
  static_assert(std::is_void_v<Return> || std::is_same_v<Return, Status>);
  if constexpr (std::is_void_v<Return>) {
    std::visit(
        [&](const auto& span) {
          VisitSelectedIndicesWhile(span, [&](int64_t i) {
            on_selection(i);
            return true;
          });
        },
        selection);
  } else {
    Status status;
    std::visit(
        [&](const auto& span) {
          VisitSelectedIndicesWhile(span, [&](int64_t i) {
            status = on_selection(i);
            return status.ok();
          });
        },
        selection);
    return status;
  }
}

}  // namespace arrow::compute::detail
