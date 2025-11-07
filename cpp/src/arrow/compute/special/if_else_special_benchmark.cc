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

#include "arrow/compute/api_special.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/function.h"
#include "arrow/compute/function_internal.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/random.h"
#include "arrow/util/logging.h"

namespace arrow::compute {

namespace {

// A trivial kernel that just keeps the CPU busy for a specified number of iterations per
// input row. Has both regular and selective variants. Used to benchmark the overhead of
// the execution framework.

struct SpinOptions : public FunctionOptions {
  explicit SpinOptions(int64_t count = 0);
  static constexpr char kTypeName[] = "SpinOptions";
  static SpinOptions Defaults() { return SpinOptions(); }
  int64_t count = 0;
};

static auto kSpinOptionsType = internal::GetFunctionOptionsType<SpinOptions>(
    arrow::internal::DataMember("count", &SpinOptions::count));

SpinOptions::SpinOptions(int64_t count)
    : FunctionOptions(kSpinOptionsType), count(count) {}

const SpinOptions* GetDefaultSpinOptions() {
  static const auto kDefaultSpinOptions = SpinOptions::Defaults();
  return &kDefaultSpinOptions;
}

using SpinState = internal::OptionsWrapper<SpinOptions>;

inline void Spin(volatile int64_t count) {
  while (count-- > 0) {
    // Do nothing, just burn CPU cycles.
  }
}

Status SpinExec(KernelContext* ctx, const ExecSpan& span, ExecResult* out) {
  ARROW_CHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  ARROW_CHECK(arg.is_array());

  int64_t count = SpinState::Get(ctx).count;
  for (int64_t i = 0; i < arg.length(); ++i) {
    Spin(count);
  }
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

Status SpinSelectiveExec(KernelContext* ctx, const ExecSpan& span,
                         const SelectionVectorSpan& selection_span, ExecResult* out) {
  ARROW_CHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  ARROW_CHECK(arg.is_array());

  int64_t count = SpinState::Get(ctx).count;
  detail::VisitSelectionVectorSpanInline(selection_span, [&](int64_t i) { Spin(count); });
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

Status RegisterSpinFunction() {
  auto registry = GetFunctionRegistry();

  if (registry->CanAddFunctionOptionsType(kSpinOptionsType).ok()) {
    RETURN_NOT_OK(registry->AddFunctionOptionsType(kSpinOptionsType));
  }

  auto register_spin_function = [&](std::string name, ArrayKernelExec exec,
                                    ArrayKernelSelectiveExec selective_exec) {
    auto func = std::make_shared<ScalarFunction>(
        std::move(name), Arity::Unary(), FunctionDoc::Empty(), GetDefaultSpinOptions());
    ScalarKernel kernel({InputType::Any()}, internal::FirstType, exec, selective_exec,
                        SpinState::Init);
    kernel.can_write_into_slices = false;
    kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
    kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
    RETURN_NOT_OK(func->AddKernel(kernel));
    if (registry->CanAddFunction(func, /*allow_overwrite=*/false).ok()) {
      RETURN_NOT_OK(registry->AddFunction(std::move(func)));
    }
    return Status::OK();
  };

  // Register two variants, one with selective exec and one without.
  RETURN_NOT_OK(register_spin_function("spin_selective", SpinExec, SpinSelectiveExec));
  RETURN_NOT_OK(register_spin_function("spin", SpinExec, /*selective_exec=*/nullptr));

  return Status::OK();
}

Expression if_else(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

auto kBooleanNullScalar = MakeNullScalar(boolean());
auto kTrueScalar = MakeScalar(true);
auto kFalseScalar = MakeScalar(false);

using MakeIfElseFunc = std::function<Expression(Expression, Expression, Expression)>;

void BenchmarkIfElse(benchmark::State& state, MakeIfElseFunc make_if_else_func,
                     const std::string& spin_function, int64_t if_true_kernel_intensity,
                     int64_t if_false_kernel_intensity, Datum cond_datum,
                     int64_t length) {
  ARROW_CHECK_EQ(cond_datum.type()->id(), Type::BOOL);

  static auto registered = RegisterSpinFunction();
  ARROW_CHECK_OK(registered);

  auto expr = make_if_else_func(
      field_ref(0),
      call(spin_function, {field_ref(1)}, SpinOptions(if_true_kernel_intensity)),
      call(spin_function, {field_ref(2)}, SpinOptions(if_false_kernel_intensity)));
  auto bound = expr.Bind(*schema({field("", cond_datum.type()), field("", int32()),
                                  field("", int32())}))
                   .ValueOrDie();
  if (cond_datum.is_arraylike()) {
    ARROW_CHECK_EQ(cond_datum.length(), length);
  }
  auto if_true_datum = ConstantArrayGenerator::Int32(length, 1);
  auto if_false_datum = ConstantArrayGenerator::Int32(length, 0);
  auto batch = ExecBatch{
      {std::move(cond_datum), std::move(if_true_datum), std::move(if_false_datum)},
      length};

  for (auto _ : state) {
    ARROW_CHECK_OK(ExecuteScalarExpression(bound, batch).status());
  }

  state.SetItemsProcessed(state.iterations() * length);
}

}  // namespace

// For each benchmark, expand to three variants:
//  - Baseline: regular if_else with regular spin kernel.
//  - Special: if_else_special with non-selective spin kernel - triggering dense
//  execution.
//  - SpecialSelective: if_else_special with selective spin kernel - triggering (more
//  efficient) sparse execution.
#define BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, name, ...)            \
  BM(ARROW_CONCAT(name, Baseline), if_else, "spin", ##__VA_ARGS__);           \
  BM(ARROW_CONCAT(name, Special), if_else_special, "spin", ##__VA_ARGS__);    \
  BM(ARROW_CONCAT(name, SpecialSelective), if_else_special, "spin_selective", \
     ##__VA_ARGS__);

#define BENCHMARK_IF_ELSE(BM, name, if_else, spin_func, arg_names, args, ...) \
  BENCHMARK_CAPTURE(BM, name, if_else, spin_func, ##__VA_ARGS__)              \
      ->ArgNames(arg_names)                                                   \
      ->ArgsProduct(args)

// Benchmark with scalar condition, see if short-circuiting takes place.
static void BM_IfElseScalarCond(benchmark::State& state, MakeIfElseFunc make_if_else_func,
                                std::string spin_func, Datum cond_datum) {
  const int64_t num_rows = state.range(0);

  BenchmarkIfElse(state, std::move(make_if_else_func), spin_func,
                  /*if_true_kernel_intensity=*/0,
                  /*if_false_kernel_intensity=*/0, std::move(cond_datum), num_rows);
}

const std::string kNumRowsArgName = "num_rows";
const std::vector<int64_t> kNumRowsArg{1, 4 * 1024, 64 * 1024};

#define BM(name, if_else, spin_func, ...)                                             \
  BENCHMARK_IF_ELSE(BM_IfElseScalarCond, name, if_else, spin_func, {kNumRowsArgName}, \
                    {kNumRowsArg}, ##__VA_ARGS__)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, Null, kBooleanNullScalar)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, True, kTrueScalar)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, False, kFalseScalar)
#undef BM

static void BenchmarkIfElseArrayCond(benchmark::State& state,
                                     MakeIfElseFunc make_if_else_func,
                                     std::string spin_func, int64_t num_rows,
                                     double true_probability, double null_probability,
                                     int64_t if_true_kernel_intensity,
                                     int64_t if_false_kernel_intensity) {
  random::RandomArrayGenerator rag(42);
  auto cond_datum = rag.Boolean(num_rows, true_probability, null_probability);

  BenchmarkIfElse(state, std::move(make_if_else_func), spin_func,
                  if_true_kernel_intensity, if_false_kernel_intensity,
                  std::move(cond_datum), num_rows);
}

// Benchmark that:
// - Both branches are evenly heavy.
// - Array condition of tunable null probability.
// See if skipping evaluating both true/false branches takes place.
static void BM_IfElseEvenBranchesNullProbability(benchmark::State& state,
                                                 MakeIfElseFunc make_if_else_func,
                                                 std::string spin_func) {
  const double null_probability = state.range(0) / 100.0;

  BenchmarkIfElseArrayCond(state, std::move(make_if_else_func), spin_func,
                           /*num_rows=*/16 * 1024, /*true_probability=*/0.5,
                           null_probability,
                           /*if_true_kernel_intensity=*/0,
                           /*if_false_kernel_intensity=*/0);
}

const std::string kNullProbabilityArgName = "null_probability";
const std::vector<int64_t> kNullProbabilityArg{0, 25, 50, 100};

#define BM(name, if_else, spin_func, ...)                                           \
  BENCHMARK_IF_ELSE(BM_IfElseEvenBranchesNullProbability, name, if_else, spin_func, \
                    {kNullProbabilityArgName}, {kNullProbabilityArg}, ##__VA_ARGS__)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, )
#undef BM

// Benchmark that:
// - Both branches are evenly heavy.
// - Array condition of tunable true probability.
// See the performance of different selectiveties and short-circuiting for extreme cases.
static void BM_IfElseEvenBranchesTrueProbability(benchmark::State& state,
                                                 MakeIfElseFunc make_if_else_func,
                                                 std::string spin_func) {
  const double true_probability = state.range(0) / 100.0;

  BenchmarkIfElseArrayCond(state, std::move(make_if_else_func), spin_func,
                           /*num_rows=*/16 * 1024, true_probability,
                           /*null_probability=*/0,
                           /*if_true_kernel_intensity=*/0,
                           /*if_false_kernel_intensity=*/0);
}

const std::string kTrueProbabilityArgName = "true_probability";
const std::vector<int64_t> kTrueProbabilityArg{0, 25, 50, 100};

#define BM(name, if_else, spin_func, ...)                                           \
  BENCHMARK_IF_ELSE(BM_IfElseEvenBranchesTrueProbability, name, if_else, spin_func, \
                    {kTrueProbabilityArgName}, {kTrueProbabilityArg}, ##__VA_ARGS__)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, )
#undef BM

// Benchmark that:
// - False branch is tunably heavier than true branch.
// - Array condition of tunable true probability.
// See the performance benefit of maskable execution when skipping the heavy false branch.
static void BM_IfElseHeavyFalse(benchmark::State& state, MakeIfElseFunc make_if_else_func,
                                std::string spin_func) {
  const double true_probability = state.range(0) / 100.0;
  const int64_t heaviness = state.range(1);

  BenchmarkIfElseArrayCond(state, std::move(make_if_else_func), spin_func,
                           /*num_rows=*/16 * 1024, /*true_probability=*/true_probability,
                           /*null_probability=*/0,
                           /*if_true_kernel_intensity=*/0, heaviness);
}

const std::string kHeavinessArgName = "heaviness";
const std::vector<int64_t> kHeavinessArg{0, 10, 100};

#define BM(name, if_else, spin_func, ...)                                            \
  BENCHMARK_IF_ELSE(BM_IfElseHeavyFalse, name, if_else, spin_func,                   \
                    ARROW_ALLOW_COMMA({kTrueProbabilityArgName, kHeavinessArgName}), \
                    ARROW_ALLOW_COMMA({{10, 50, 90}, kHeavinessArg}), ##__VA_ARGS__)
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SPECIAL(BM, )
#undef BM

}  // namespace arrow::compute
