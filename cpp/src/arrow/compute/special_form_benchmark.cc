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
#include "arrow/compute/function_internal.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/random.h"
#include "arrow/util/logging.h"

namespace arrow {

namespace compute {

namespace {

struct PayloadOptions : public FunctionOptions {
  explicit PayloadOptions(int64_t load = 0);
  static constexpr char const kTypeName[] = "PayloadOptions";
  static PayloadOptions Defaults() { return PayloadOptions(); }
  int64_t load = 0;
};

static auto kPayloadOptionsType = internal::GetFunctionOptionsType<PayloadOptions>(
    arrow::internal::DataMember("load", &PayloadOptions::load));

PayloadOptions::PayloadOptions(int64_t load)
    : FunctionOptions(kPayloadOptionsType), load(load) {}

const PayloadOptions* GetDefaultPayloadOptions() {
  static const auto kDefaultPayloadOptions = PayloadOptions::Defaults();
  return &kDefaultPayloadOptions;
}

using PayloadState = internal::OptionsWrapper<PayloadOptions>;

Status PayloadExec(KernelContext* ctx, const ExecSpan& span, ExecResult* out) {
  ARROW_CHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  ARROW_CHECK(arg.is_array());

  int64_t load = PayloadState::Get(ctx).load;
  int64_t load_length =
      span.selection_vector ? span.selection_vector->length() : arg.length();
  for (int64_t i = 0; i < load_length; ++i) {
    volatile int64_t j = load;
    while (j-- > 0) {
    }
  }
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

Status RegisterAuxilaryFunctions() {
  auto registry = GetFunctionRegistry();

  {
    if (registry->CanAddFunctionOptionsType(kPayloadOptionsType).ok()) {
      RETURN_NOT_OK(registry->AddFunctionOptionsType(kPayloadOptionsType));
    }
  }
  {
    auto register_payload_func = [&](const std::string& name,
                                     bool sv_awareness) -> Status {
      auto func = std::make_shared<ScalarFunction>(
          name, Arity::Unary(), FunctionDoc::Empty(), GetDefaultPayloadOptions());

      ScalarKernel kernel({InputType::Any()}, internal::FirstType, PayloadExec,
                          PayloadState::Init);
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

    RETURN_NOT_OK(register_payload_func("payload_sv_aware", true));
    RETURN_NOT_OK(register_payload_func("payload_sv_unaware", false));
  }

  return Status::OK();
}

Expression if_else_regular(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Expression sv_suppress(Expression arg) {
  return call("payload_sv_unaware", {std::move(arg)});
}

Expression heavy(Expression arg) {
  return call("payload_sv_aware", {std::move(arg)},
              std::make_shared<PayloadOptions>(/*load=*/1024));
}

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

#ifdef BM
#  error("BM is defined")
#else
#  define BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, name, ...)      \
    BM(name##_regular, if_else_regular, ##__VA_ARGS__);                       \
    BM(name##_special, if_else_special, ##__VA_ARGS__);                       \
    BM(name##_regular_sv_unaware, sv_unaware_if_else_regular, ##__VA_ARGS__); \
    BM(name##_special_sv_unaware, sv_unaware_if_else_special, ##__VA_ARGS__);
#endif

#define BENCHMARK_IF_ELSE(BM, name, if_else, arg_names, args, ...) \
  BENCHMARK_CAPTURE(BM, name, if_else, ##__VA_ARGS__)              \
      ->ArgNames(arg_names)                                        \
      ->ArgsProduct(args)

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

const std::vector<std::string> kNumRowsArgNames{"num_rows"};
const std::vector<int64_t> kNumRowsArg = benchmark::CreateRange(1, 64 * 1024, 32);

#define BM(name, if_else, ...)                                             \
  BENCHMARK_IF_ELSE(BM_IfElseTrivialCond, name, if_else, kNumRowsArgNames, \
                    {kNumRowsArg}, ##__VA_ARGS__)
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
  auto b = rag.Boolean(num_rows, true_probability, null_probability);
  auto schema = arrow::schema({field("b", boolean())});

  ExecBatch batch{std::vector<Datum>{std::move(b)}, num_rows};

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

template <typename... Args>
void BM_IfElseWithOneHeavySide(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Args&&...) {
  const double heavy_ratio = state.range(0) / 100.0;
  constexpr int64_t num_rows = 65536;

  random::RandomArrayGenerator rag(42);
  auto b =
      rag.Boolean(num_rows, /*true_probability=*/heavy_ratio, /*null_probability=*/0.0);
  auto i = ConstantArrayGenerator::Int32(num_rows, 42);
  auto schema = arrow::schema({field("b", boolean()), field("i", int32())});

  ExecBatch batch{std::vector<Datum>{std::move(b), std::move(i)}, num_rows};

  BenchmarkIfElse(state, std::move(if_else_func), field_ref("b"), heavy(field_ref("i")),
                  literal(0), schema, batch);
}

const std::vector<std::string> kHeavyRatioArgNames{"heavy_ratio"};
const std::vector<int64_t> kHeavyRatioArgs{{0, 25, 50, 75, 100}};

#define BM(name, if_else, ...)                                                     \
  BENCHMARK_IF_ELSE(BM_IfElseWithOneHeavySide, name, if_else, kHeavyRatioArgNames, \
                    {kHeavyRatioArgs})
BENCHMARK_IF_ELSE_WITH_BASELINE_AND_SV_SUPPRESS(BM, one_heavy_side);
#undef BM

}  // namespace compute

}  // namespace arrow
