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

#include <cstdint>
#include <string>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/function.h"
#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/codegen_internal.h"

namespace arrow::compute::internal {

struct ScatterKernelData {
  InputType value_type;
  InputType selection_type;
  ArrayKernelExec exec;
};

void RegisterScatterFunction(const std::string& name, FunctionDoc doc,
                               VectorKernel base_kernel,
                               std::vector<ScatterKernelData>&& kernels,
                               const FunctionOptions* default_options,
                               FunctionRegistry* registry);

/// \brief Callback type for VisitPlainxREEFilterOutputSegments.
///
/// position is the logical position in the values array relative to its offset.
///
/// segment_length is the number of elements that should be emitted.
///
/// filter_valid is true if the filter run value is non-NULL. This value can
/// only be false if null_selection is NullSelectionBehavior::EMIT_NULL. For
/// NullSelectionBehavior::DROP, NULL values from the filter are simply skipped.
///
/// Return true if iteration should continue, false if iteration should stop.
// using EmitREEFilterSegment =
//     std::function<bool(int64_t position, int64_t segment_length, bool filter_valid)>;

// void VisitPlainxREEFilterOutputSegments(
//     const ArraySpan& filter, bool filter_may_have_nulls,
//     FilterOptions::NullSelectionBehavior null_selection,
//     const EmitREEFilterSegment& emit_segment);

Status PrimitiveScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);
Status ListScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);
Status LargeListScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);
Status FSLScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);
Status DenseUnionScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);
Status MapScatterByMaskExec(KernelContext*, const ExecSpan&, ExecResult*);

}  // namespace arrow::compute::internal
