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

Expression sv_unaware_if_else_regular(Expression cond, Expression if_true,
                                      Expression if_false) {
  return if_else_regular(std::move(cond), sv_suppress(std::move(if_true)),
                         sv_suppress(std::move(if_false)));
}

Expression sv_unaware_if_else_special(Expression cond, Expression if_true,
                                      Expression if_false) {
  return if_else_special(std::move(cond), sv_suppress(std::move(if_true)),
                         sv_suppress(std::move(if_false)));
}

auto kBooleanNull = literal(MakeNullScalar(boolean()));
auto kIntNull = literal(MakeNullScalar(int32()));

void BenchmarkIfElse(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Expression cond, Expression if_true, Expression if_false,
    const std::shared_ptr<Schema>& schema, const ExecBatch& batch) {
  ARROW_CHECK_OK(RegisterAuxilaryFunctions());

  auto if_else = if_else_func(std::move(cond), std::move(if_true), std::move(if_false));
  auto bound = if_else.Bind(*schema).ValueOrDie();
  for (auto _ : state) {
    ARROW_CHECK_OK(ExecuteScalarExpression(bound, batch).status());
  }

  state.SetItemsProcessed(batch.length * state.iterations());
}

}  // namespace

template <typename... Args>
static void BM_IfElseTrivialCond(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Expression cond, Args&&...) {
  const int64_t num_rows = state.range(0);

  auto schema = arrow::schema({field("i1", int32()), field("i2", int32())});

  auto i1 = ConstantArrayGenerator::Int32(num_rows, 1);
  auto i2 = ConstantArrayGenerator::Int32(num_rows, 0);
  ExecBatch batch{std::vector<Datum>{std::move(i1), std::move(i2)}, num_rows};

  BenchmarkIfElse(state, std::move(if_else_func), std::move(cond), field_ref("i1"),
                  field_ref("i2"), schema, batch);
}

#ifdef BM
#  error("BM is defined")
#else
#  define BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, name, ...)    \
    BM(name##_regular, if_else_regular, __VA_ARGS__);                       \
    BM(name##_special, if_else_special, __VA_ARGS__);                       \
    BM(name##_regular_sv_unaware, sv_unaware_if_else_regular, __VA_ARGS__); \
    BM(name##_special_sv_unaware, sv_unaware_if_else_special, __VA_ARGS__);
#endif

const std::vector<std::string> kNumRowsArgNames{"num_rows"};
const std::vector<int64_t> kNumRowsArg = benchmark::CreateRange(1, 64 * 1024, 32);

#define BENCHMARK_IF_ELSE(BM, name, if_else, arg_names, args) \
  BENCHMARK_CAPTURE(BM, name, if_else)->ArgNames(arg_names)->ArgsProduct({args})

#define BENCHMARK_IF_ELSE_ARGS(BM, name, if_else, arg_names, args, ...) \
  BENCHMARK_CAPTURE(BM, name, if_else, __VA_ARGS__)                     \
      ->ArgNames(arg_names)                                             \
      ->ArgsProduct(args)

#define BM(name, if_else, ...)                                                  \
  BENCHMARK_IF_ELSE_ARGS(BM_IfElseTrivialCond, name, if_else, kNumRowsArgNames, \
                         {kNumRowsArg}, __VA_ARGS__)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, literal_null, kBooleanNull)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, literal_true, literal(true))
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, literal_false, literal(false))
#undef BM

namespace {

void BenchmarkIfElseWithCondArray(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    int64_t num_rows, double true_probability, double null_probability) {
  random::RandomArrayGenerator rag(42);
  auto cond = rag.Boolean(num_rows, true_probability, null_probability);
  auto schema = arrow::schema({field("b", boolean())});

  ExecBatch batch{std::vector<Datum>{cond}, cond->length()};

  BenchmarkIfElse(state, std::move(if_else_func), field_ref("b"), literal(1), literal(0),
                  schema, batch);
}

}  // namespace

template <typename... Args>
static void BM_IfElseNumRows(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Args&&...) {
  const int64_t num_rows = state.range(0);

  BenchmarkIfElseWithCondArray(state, std::move(if_else_func), num_rows,
                               /*true_probability=*/0.5, /*null_probability=*/0.0);
}

#define BM(name, if_else, ...) \
  BENCHMARK_IF_ELSE(BM_IfElseNumRows, name, if_else, kNumRowsArgNames, {kNumRowsArg})
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, num_rows)
#undef BM

template <typename... Args>
static void BM_IfElseNullProbability(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Args&&...) {
  const int64_t num_rows = state.range(0);
  const double null_probability = state.range(1) / 100.0;

  BenchmarkIfElseWithCondArray(state, std::move(if_else_func), num_rows,
                               /*true_probability=*/0.5, null_probability);
}

const std::vector<std::string> kNumRowsAndNullProbabilityArgNames{"num_rows",
                                                                  "null_probability"};
const std::vector<std::vector<int64_t>> kNumRowsAndNullProbabilityArgs{
    {4 * 1024, 64 * 1024}, {0, 50, 90, 100}};

#define BM(name, if_else, ...)                               \
  BENCHMARK_IF_ELSE(BM_IfElseNullProbability, name, if_else, \
                    kNumRowsAndNullProbabilityArgNames, kNumRowsAndNullProbabilityArgs)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, null_probability);
#undef BM

}  // namespace compute

}  // namespace arrow
