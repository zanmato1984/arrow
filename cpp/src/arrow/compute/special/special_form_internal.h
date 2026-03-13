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

#include "arrow/compute/expression_internal.h"
#include "arrow/compute/special_form.h"
#include "arrow/util/logging_internal.h"

namespace arrow::compute::internal {

/// @brief A CRTP base class for special forms whose binding are backed by a function
/// call.
///
/// Many special forms share the same binding logic as its non-special function
/// counterpart, e.g., implicit casts and output type resolution. This class encapsulates
/// the binding logic for such special forms, instantiating a Call instance and binding
/// it, then delegating the actual binding of the special form to the derived class via
/// BindWithBoundCall() with the bound Call instance.
template <typename Impl>
class FunctionBackedSpecialForm : public SpecialForm {
 public:
  using SpecialForm::SpecialForm;

  ARROW_DISALLOW_COPY_AND_ASSIGN(FunctionBackedSpecialForm);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(FunctionBackedSpecialForm);

  Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, std::shared_ptr<FunctionOptions> options,
      ExecContext* exec_context) const override {
    DCHECK(std::all_of(arguments.begin(), arguments.end(),
                       [](const Expression& argument) { return argument.IsBound(); }));
    Expression::Call call;
    call.function_name = name();
    call.arguments = std::move(arguments);
    call.options = std::move(options);
    ARROW_ASSIGN_OR_RAISE(
        auto bound, BindNonRecursive(call, /*insert_implicit_casts=*/true, exec_context));
    auto bound_call = CallNotNull(bound);
    auto bound_call_copy = *bound_call;
    arguments = std::move(bound_call->arguments);
    options = std::move(bound_call->options);
    return static_cast<const Impl*>(this)->BindWithBoundCall(std::move(bound_call_copy),
                                                             exec_context);
  }
};

}  // namespace arrow::compute::internal
