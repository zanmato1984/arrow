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

#include "arrow/compute/exec_internal.h"

#include <memory>
#include <utility>
#include <vector>

#include "arrow/compute/kernel.h"
#include "arrow/datum.h"
#include "arrow/status.h"
#include "arrow/util/logging_internal.h"

namespace arrow::compute::detail {

namespace {

class DenseSelectionScalarExecutor : public KernelExecutor {
 public:
  explicit DenseSelectionScalarExecutor(std::unique_ptr<KernelExecutor> executor)
      : executor_(std::move(executor)) {
    DCHECK_NE(executor_, nullptr);
  }

  Status Init(KernelContext* kernel_ctx, KernelInitArgs args) override {
    kernel_ = static_cast<const ScalarKernel*>(args.kernel);
    exec_context_ = kernel_ctx->exec_context();
    return executor_->Init(kernel_ctx, args);
  }

  Status Execute(const ExecBatch& batch, ExecListener* listener) override {
    DCHECK_NE(kernel_, nullptr);

    // Preserve the scalar executor's zero-length handling. A selection cannot make a
    // zero-length input any denser.
    if (batch.length == 0 || !batch.selection_vector || kernel_->selective_exec) {
      return executor_->Execute(batch, listener);
    }

    if (CheckIfAllScalar(batch)) {
      // The result is scalar regardless of the selection, so gathering and scattering
      // would only box an otherwise scalar result.
      ExecBatch input = batch;
      input.selection_vector = nullptr;
      return executor_->Execute(input, listener);
    }

    const auto& selection = SelectionBackend::Get(*batch.selection_vector);
    std::vector<Datum> values;
    values.reserve(batch.values.size());
    for (const auto& value : batch.values) {
      if (value.is_scalar()) {
        values.push_back(value);
      } else {
        ARROW_ASSIGN_OR_RAISE(auto dense, selection.Gather(value, exec_context_));
        values.push_back(std::move(dense));
      }
    }
    ARROW_ASSIGN_OR_RAISE(
        ExecBatch input,
        ExecBatch::Make(std::move(values), batch.selection_vector->length()));

    DatumAccumulator dense_listener;
    RETURN_NOT_OK(executor_->Execute(input, &dense_listener));
    Datum dense_result = executor_->WrapResults(input.values, dense_listener.values());

    ARROW_ASSIGN_OR_RAISE(Datum result,
                          selection.Scatter(dense_result, batch.length, exec_context_));
    return listener->OnResult(std::move(result));
  }

  Datum WrapResults(const std::vector<Datum>& args,
                    const std::vector<Datum>& outputs) override {
    return executor_->WrapResults(args, outputs);
  }

  Status CheckResultType(const Datum& out, const char* function_name) override {
    return executor_->CheckResultType(out, function_name);
  }

 private:
  std::unique_ptr<KernelExecutor> executor_;
  const ScalarKernel* kernel_ = nullptr;
  ExecContext* exec_context_ = nullptr;
};

}  // namespace

std::unique_ptr<KernelExecutor> MakeDenseSelectionScalarExecutor(
    std::unique_ptr<KernelExecutor> executor) {
  return std::make_unique<DenseSelectionScalarExecutor>(std::move(executor));
}

}  // namespace arrow::compute::detail
