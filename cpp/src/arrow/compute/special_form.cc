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

#include "arrow/compute/special_form.h"

#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/expression_internal.h"
#include "arrow/compute/registry.h"

namespace arrow::compute {

Result<TypeHolder> IfElseSpecialForm::Resolve(std::vector<Expression>* arguments,
                                              ExecContext* exec_context) const {
  ARROW_ASSIGN_OR_RAISE(auto function,
                        exec_context->func_registry()->GetFunction("if_else"));
  std::vector<TypeHolder> types = GetTypes(*arguments);

  // TODO: DispatchBest and implicit cast.
  ARROW_ASSIGN_OR_RAISE(auto maybe_exact_match, function->DispatchExact(types));
  compute::KernelContext kernel_context(exec_context, maybe_exact_match);
  if (maybe_exact_match->init) {
    const FunctionOptions* options = function->default_options();
    ARROW_ASSIGN_OR_RAISE(
        auto kernel_state,
        maybe_exact_match->init(&kernel_context, {maybe_exact_match, types, options}));
    kernel_context.SetState(kernel_state.get());
  }
  return maybe_exact_match->signature->out_type().Resolve(&kernel_context, types);
}

namespace {

Result<ExecBatch> TakeBySelectionVector(const ExecBatch& input,
                                        const Datum& selection_vector,
                                        ExecContext* exec_context) {
  std::vector<Datum> values(input.num_values());
  for (int i = 0; i < input.num_values(); ++i) {
    ARROW_ASSIGN_OR_RAISE(
        values[i], Take(input[i], selection_vector, TakeOptions{/*boundcheck=*/false},
                        exec_context));
  }
  return ExecBatch::Make(std::move(values), selection_vector.length());
}

std::shared_ptr<ChunkedArray> ChunkedArrayFromDatums(const std::vector<Datum>& datums) {
  std::vector<std::shared_ptr<Array>> chunks;
  for (const auto& datum : datums) {
    DCHECK(datum.is_arraylike());
    if (datum.is_array()) {
      chunks.push_back(datum.make_array());
    } else {
      DCHECK(datum.is_chunked_array());
      for (const auto& chunk : datum.chunked_array()->chunks()) {
        chunks.push_back(chunk);
      }
    }
  }
  return std::make_shared<ChunkedArray>(std::move(chunks));
}

}  // namespace

Result<Datum> IfElseSpecialForm::Execute(const std::vector<Expression>& arguments,
                                         const ExecBatch& input,
                                         compute::ExecContext* exec_context) const {
  // TODO: Selection vector aware path.
  DCHECK_EQ(input.selection_vector, nullptr);
  DCHECK_EQ(arguments.size(), 3);

  const auto& cond_expr = arguments[0];
  DCHECK_EQ(cond_expr.type()->id(), Type::BOOL);
  const auto& if_true_expr = arguments[1];
  const auto& if_false_expr = arguments[2];
  DCHECK_EQ(if_true_expr.type()->id(), if_false_expr.type()->id());

  ARROW_ASSIGN_OR_RAISE(auto cond,
                        ExecuteScalarExpression(cond_expr, input, exec_context));
  DCHECK_EQ(cond.type()->id(), Type::BOOL);

  ARROW_ASSIGN_OR_RAISE(auto sel_true,
                        CallFunction("indices_nonzero", {cond}, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto input_true,
                        TakeBySelectionVector(input, sel_true, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto if_true,
                        ExecuteScalarExpression(if_true_expr, input_true, exec_context));

  ARROW_ASSIGN_OR_RAISE(auto cond_inverted, CallFunction("invert", {cond}, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto sel_false,
                        CallFunction("indices_nonzero", {cond_inverted}, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto input_false,
                        TakeBySelectionVector(input, sel_false, exec_context));
  ARROW_ASSIGN_OR_RAISE(
      auto if_false, ExecuteScalarExpression(if_false_expr, input_false, exec_context));

  auto if_true_false = ChunkedArrayFromDatums({if_true, if_false});
  auto sel = ChunkedArrayFromDatums({sel_true, sel_false});
  ARROW_ASSIGN_OR_RAISE(
      auto result_datum,
      Permute(if_true_false, sel, PermuteOptions{/*output_length=*/input.length}));
  DCHECK(result_datum.is_arraylike());
  if (result_datum.is_chunked_array() &&
      result_datum.chunked_array()->num_chunks() == 1) {
    return result_datum.chunked_array()->chunk(0);
  } else {
    return result_datum;
  }
}

}  // namespace arrow::compute
