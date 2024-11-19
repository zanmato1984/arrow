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

#include "benchmark/benchmark.h"

#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/function.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/random.h"
#include "arrow/util/logging.h"

namespace arrow {

namespace compute {

namespace {

Status IdentityExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
  ARROW_CHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  ARROW_CHECK(arg.is_array());
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

Status RegisterAuxilaryFunctions() {
  auto registry = GetFunctionRegistry();

  {
    auto register_sv_awareness_func = [&](const std::string& name,
                                          bool sv_awareness) -> Status {
      auto func =
          std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

      ArrayKernelExec exec = IdentityExec;
      ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
      kernel.selection_vector_aware = sv_awareness;
      kernel.can_write_into_slices = false;
      kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
      kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
      RETURN_NOT_OK(func->AddKernel(kernel));
      if (registry->CanAddFunction(func, /*allow_overwrite=*/false).ok()) {
        RETURN_NOT_OK(registry->AddFunction(std::move(func)));
      }
      return Status::OK();
    };

    RETURN_NOT_OK(register_sv_awareness_func("sv_suppress", false));
  }

  return Status::OK();
}

Expression if_else_regular(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Expression sv_suppress(Expression arg) { return call("sv_suppress", {std::move(arg)}); }

}  // namespace

template <typename... Args>
static void BM_IfElseArray(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Args&&...) {
  ARROW_CHECK_OK(RegisterAuxilaryFunctions());

  auto schema =
      arrow::schema({field("a", boolean()), field("b", int32()), field("c", int32())});
  random::RandomArrayGenerator rag(42);
  int64_t num_rows = 65536;
  auto cond = rag.Boolean(num_rows, 0.5, 0.0);
  auto if_true = rag.Int32(num_rows, 0, 42);
  auto if_false = rag.Int32(num_rows, 0, 42);
  auto if_else = if_else_func(field_ref("a"), field_ref("b"), field_ref("c"));
  auto bound = if_else.Bind(*schema).ValueOrDie();
  ExecBatch batch{
      std::vector<Datum>{std::move(cond), std::move(if_true), std::move(if_false)},
      num_rows};
  for (auto _ : state) {
    ARROW_CHECK_OK(ExecuteScalarExpression(bound, batch).status());
  }
  state.SetItemsProcessed(num_rows * state.iterations());
}

BENCHMARK_CAPTURE(BM_IfElseArray, "regular_sv_aware", {if_else_regular});
BENCHMARK_CAPTURE(BM_IfElseArray, "special_sv_aware", {if_else_special});
BENCHMARK_CAPTURE(BM_IfElseArray, "regular_sv_unaware",
                  {[](Expression cond, Expression if_true, Expression if_false) {
                    return if_else_regular(sv_suppress(cond), sv_suppress(if_true),
                                           sv_suppress(if_false));
                  }});
BENCHMARK_CAPTURE(BM_IfElseArray, "special_sv_unaware",
                  {[](Expression cond, Expression if_true, Expression if_false) {
                    return if_else_special(sv_suppress(cond), sv_suppress(if_true),
                                           sv_suppress(if_false));
                  }});

}  // namespace compute

}  // namespace arrow
