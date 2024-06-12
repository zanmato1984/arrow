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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "arrow/array/array_binary.h"
#include "arrow/array/array_dict.h"
#include "arrow/array/array_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/concatenate.h"
#include "arrow/buffer_builder.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/kernels/common_internal.h"
#include "arrow/compute/kernels/util_internal.h"
#include "arrow/compute/kernels/vector_scatter_by_mask_internal.h"
#include "arrow/extension_type.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/bitmap_reader.h"
#include "arrow/util/int_util.h"

namespace arrow {

using internal::BinaryBitBlockCounter;
using internal::BitBlockCount;
using internal::BitBlockCounter;
using internal::CheckIndexBounds;
using internal::CopyBitmap;
using internal::CountSetBits;
using internal::OptionalBitBlockCounter;
using internal::OptionalBitIndexer;

namespace compute {
namespace internal {

namespace {

// ----------------------------------------------------------------------

const FunctionDoc array_scatter_by_mask_doc(
    "Scatter with a boolean positional mask",
    ("The values of the input `array` will be placed into the output at positions where "
     "the `positional_mask` is non-zero.  The rest positions of the output will be "
     "populated by `null`s.\n"),
    {"array", "positional_mask"});

}  // namespace

void RegisterVectorScatter(FunctionRegistry* registry) {
  // Scatter by mask kernels
  std::vector<ScatterKernelData> scatter_by_mask_kernels;
  PopulateScatterByMaskKernels(&scatter_by_mask_kernels);

  VectorKernel scatter_by_mask_base;
  scatter_by_mask_base.can_execute_chunkwise = false;
  scatter_by_mask_base.output_chunked = false;
  RegisterScatterFunction("array_scatter_by_mask", array_scatter_by_mask_doc,
                          scatter_by_mask_base, std::move(scatter_by_mask_kernels),
                          NULLPTR, registry);

  DCHECK_OK(registry->AddFunction(MakeScatterByMaskMetaFunction()));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
