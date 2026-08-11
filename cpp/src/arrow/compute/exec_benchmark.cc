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

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "arrow/compute/exec_internal.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/function_internal.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/chunked_array.h"
#include "arrow/datum.h"
#include "arrow/testing/generator.h"
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
                         const SelectionSpan& selection, ExecResult* out) {
  ARROW_CHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  ARROW_CHECK(arg.is_array());

  int64_t count = SpinState::Get(ctx).count;
  detail::VisitSelectionSpanInline(selection, [&](int64_t i) { Spin(count); });
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

enum class SelectionPattern : int8_t {
  kPrefix,
  // Spread indices across chunks with (at most) one selected row per chunk, which
  // maximizes chunk-boundary overhead for a given number of selected rows.
  kOnePerChunk,
  // Random unique indices distributed across the full length (sorted).
  kRandomSortedUnique,
};

std::shared_ptr<ChunkedArray> MakeChunkedInt32(int64_t num_rows, int64_t chunk_size) {
  ARROW_CHECK_GT(chunk_size, 0);
  ArrayVector chunks;
  for (int64_t offset = 0; offset < num_rows; offset += chunk_size) {
    const int64_t len = std::min(chunk_size, num_rows - offset);
    chunks.push_back(ConstantArrayGenerator::Int32(len, 0));
  }
  return std::make_shared<ChunkedArray>(std::move(chunks), int32());
}

std::shared_ptr<SelectionVector> MakeSelectionVectorFromIndices(
    const std::vector<int32_t>& indices) {
  Int32Builder builder;
  ARROW_CHECK_OK(builder.Reserve(static_cast<int64_t>(indices.size())));
  ARROW_CHECK_OK(
      builder.AppendValues(indices.data(), static_cast<int64_t>(indices.size())));
  std::shared_ptr<Array> arr;
  ARROW_CHECK_OK(builder.Finish(&arr));
  return SelectionVector::MakeIndices(*arr);
}

std::shared_ptr<SelectionVector> MakeSelectionVectorPrefix(int64_t length) {
  auto res = gen::Step<int32_t>()->Generate(length);
  ARROW_CHECK_OK(res.status());
  auto arr = res.ValueUnsafe();
  return SelectionVector::MakeIndices(*arr);
}

std::shared_ptr<SelectionVector> MakeSelectionVectorRandomSortedUnique(
    int64_t length, int64_t selected_length, uint64_t seed = 0x4d595df4d0f33173ULL) {
  ARROW_CHECK_GE(length, 0);
  selected_length = std::max<int64_t>(0, std::min<int64_t>(selected_length, length));
  std::vector<int32_t> indices;
  indices.reserve(static_cast<size_t>(selected_length));
  if (selected_length == 0) {
    return MakeSelectionVectorFromIndices(indices);
  }

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(length - 1));
  std::vector<uint8_t> seen(static_cast<size_t>(length), 0);
  while (static_cast<int64_t>(indices.size()) < selected_length) {
    const int32_t v = dist(rng);
    auto& flag = seen[static_cast<size_t>(v)];
    if (!flag) {
      flag = 1;
      indices.push_back(v);
    }
  }
  std::sort(indices.begin(), indices.end());
  return MakeSelectionVectorFromIndices(indices);
}

std::shared_ptr<SelectionVector> MakeSelectionVectorOnePerChunk(const ChunkedArray& input,
                                                                int64_t selected_length) {
  const int64_t length = input.length();
  selected_length = std::max<int64_t>(0, std::min<int64_t>(selected_length, length));
  std::vector<int32_t> indices;
  indices.reserve(static_cast<size_t>(selected_length));
  if (selected_length == 0) {
    return MakeSelectionVectorFromIndices(indices);
  }

  const int num_chunks = input.num_chunks();
  if (num_chunks == 0) {
    return MakeSelectionVectorFromIndices(indices);
  }

  // Choose (up to) one index per distinct chunk, spreading chunk ids across the full
  // range. We then sort indices to satisfy SelectionVector's strict-increasing contract.
  const uint64_t stride = 1315423911ULL;  // odd constant, deterministic
  std::vector<int64_t> chunk_offsets(static_cast<size_t>(num_chunks));
  std::vector<int> eligible_chunk_ids;
  eligible_chunk_ids.reserve(static_cast<size_t>(num_chunks));
  int64_t offset = 0;
  for (int i = 0; i < num_chunks; ++i) {
    chunk_offsets[static_cast<size_t>(i)] = offset;
    const int64_t chunk_len = input.chunk(i)->length();
    if (chunk_len > 0) {
      eligible_chunk_ids.push_back(i);
    }
    offset += chunk_len;
  }

  if (eligible_chunk_ids.empty()) {
    return MakeSelectionVectorFromIndices(indices);
  }

  std::vector<uint8_t> chunk_used(eligible_chunk_ids.size(), 0);
  uint64_t k = 0;
  while (static_cast<int64_t>(indices.size()) < selected_length) {
    const size_t idx =
        static_cast<size_t>((k * stride) % static_cast<uint64_t>(eligible_chunk_ids.size()));
    const int chunk_id = eligible_chunk_ids[idx];
    ++k;
    auto& used = chunk_used[idx];
    if (used) {
      continue;
    }
    used = 1;
    const int64_t chunk_start = chunk_offsets[static_cast<size_t>(chunk_id)];
    const int64_t chunk_len = input.chunk(chunk_id)->length();
    // Pick a deterministic position within the chunk (not always 0), to avoid
    // accidentally creating large contiguous runs across chunks when chunk_size is big.
    const int64_t local = static_cast<int64_t>(chunk_id % chunk_len);
    indices.push_back(static_cast<int32_t>(chunk_start + local));

    // If we exhausted all chunks, fall back to random fill (still sorted/unique)
    // to reach selected_length.
    if (static_cast<int>(indices.size()) ==
        static_cast<int>(eligible_chunk_ids.size())) {
      break;
    }
  }

  if (static_cast<int64_t>(indices.size()) < selected_length) {
    // Fill remaining indices randomly across the whole length, avoiding duplicates.
    // Since we've already picked <= num_chunks indices, this is cheap for our sizes.
    std::sort(indices.begin(), indices.end());
    std::vector<uint8_t> seen(static_cast<size_t>(length), 0);
    for (int32_t v : indices) {
      seen[static_cast<size_t>(v)] = 1;
    }
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(length - 1));
    while (static_cast<int64_t>(indices.size()) < selected_length) {
      const int32_t v = dist(rng);
      auto& flag = seen[static_cast<size_t>(v)];
      if (!flag) {
        flag = 1;
        indices.push_back(v);
      }
    }
    std::sort(indices.begin(), indices.end());
  } else {
    std::sort(indices.begin(), indices.end());
  }

  return MakeSelectionVectorFromIndices(indices);
}

std::shared_ptr<SelectionVector> MakeSelectionVector(SelectionPattern pattern,
                                                     const Datum& input,
                                                     int64_t selected_length) {
  const int64_t length = input.length();
  switch (pattern) {
    case SelectionPattern::kPrefix:
      return MakeSelectionVectorPrefix(selected_length);
    case SelectionPattern::kOnePerChunk:
      if (input.is_chunked_array()) {
        return MakeSelectionVectorOnePerChunk(*input.chunked_array(), selected_length);
      }
      // For non-chunked inputs, this degenerates to a "random spread" pattern.
      return MakeSelectionVectorRandomSortedUnique(length, selected_length);
    case SelectionPattern::kRandomSortedUnique:
      return MakeSelectionVectorRandomSortedUnique(length, selected_length);
  }
  ARROW_LOG(FATAL) << "unreachable";
  return nullptr;
}

}  // namespace

void BenchmarkExec(benchmark::State& state, std::string spin_function,
                   int64_t kernel_intensity, Datum input,
                   std::shared_ptr<SelectionVector> selection = nullptr) {
  static auto registered = RegisterSpinFunction();
  ARROW_CHECK_OK(registered);

  auto expr =
      call(std::move(spin_function), {field_ref(0)}, SpinOptions(kernel_intensity));
  auto bound = expr.Bind(*schema({field("", input.type())})).ValueOrDie();
  auto length = input.length();
  auto batch = ExecBatch(std::vector<Datum>{std::move(input)}, length, std::move(selection));

  for (auto _ : state) {
    ARROW_CHECK_OK(ExecuteScalarExpression(bound, batch).status());
  }

  state.SetItemsProcessed(state.iterations() * length);
}

// Baseline: Run the spin kernel without selection vector.
static void BM_ExecBaseline(benchmark::State& state) {
  const int64_t kernel_intensity = state.range(0);
  const int64_t num_rows = state.range(1);

  auto input = ConstantArrayGenerator::Int32(num_rows, 0);
  BenchmarkExec(state, "spin", kernel_intensity, Datum(std::move(input)));
}

// Selective: Run the spin kernel with a selection vector, either sparsely or densely,
// depending on whether the spin kernel has a selective exec implementation.
static void BM_ExecSelective(benchmark::State& state, std::string spin_function) {
  const int64_t selectivity = state.range(0);
  ARROW_CHECK(selectivity >= 0 && selectivity <= 100);
  const int64_t kernel_intensity = state.range(1);
  const int64_t num_rows = state.range(2);

  auto input = ConstantArrayGenerator::Int32(num_rows, 0);
  auto selection =
      MakeSelectionVectorPrefix(static_cast<int64_t>(num_rows * selectivity / 100));
  BenchmarkExec(state, std::move(spin_function), kernel_intensity, Datum(std::move(input)),
                std::move(selection));
}

static void BM_ExecBaselineChunked(benchmark::State& state) {
  const int64_t kernel_intensity = state.range(0);
  const int64_t num_rows = state.range(1);
  const int64_t chunk_size = state.range(2);

  auto input = MakeChunkedInt32(num_rows, chunk_size);
  BenchmarkExec(state, "spin", kernel_intensity, Datum(std::move(input)));
}

static void BM_ExecSelectiveChunked(benchmark::State& state, std::string spin_function,
                                   SelectionPattern pattern) {
  const int64_t selectivity = state.range(0);
  ARROW_CHECK(selectivity >= 0 && selectivity <= 100);
  const int64_t kernel_intensity = state.range(1);
  const int64_t num_rows = state.range(2);
  const int64_t chunk_size = state.range(3);

  Datum input(MakeChunkedInt32(num_rows, chunk_size));
  const int64_t selected_length = static_cast<int64_t>(num_rows * selectivity / 100);
  auto selection = MakeSelectionVector(pattern, input, selected_length);
  BenchmarkExec(state, std::move(spin_function), kernel_intensity, std::move(input),
                std::move(selection));
}

const char* kSelectivityArgName = "selectivity";
const std::vector<int64_t> kSelectivityArg{0, 20, 50, 100};
const char* kKernelIntensityArgName = "kernel_intensity";
const std::vector<int64_t> kKernelIntensityArg = benchmark::CreateDenseRange(0, 100, 20);
const char* kNumRowsArgName = "num_rows";
const std::vector<int64_t> kNumRowsArg{4096};

const char* kChunkSizeArgName = "chunk_size";
const std::vector<int64_t> kChunkSizeArg{4096, 256, 16};
const std::vector<int64_t> kSelectivityChunkedArg{0, 1, 20, 50, 100};
const std::vector<int64_t> kKernelIntensityChunkedArg{0, 100};

BENCHMARK(BM_ExecBaseline)
    ->ArgNames({kKernelIntensityArgName, kNumRowsArgName})
    ->ArgsProduct({kKernelIntensityArg, kNumRowsArg});

BENCHMARK_CAPTURE(BM_ExecSelective, sparse, "spin_selective")
    ->ArgNames({kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName})
    ->ArgsProduct({kSelectivityArg, kKernelIntensityArg, kNumRowsArg});

BENCHMARK_CAPTURE(BM_ExecSelective, dense, "spin")
    ->ArgNames({kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName})
    ->ArgsProduct({kSelectivityArg, kKernelIntensityArg, kNumRowsArg});

BENCHMARK(BM_ExecBaselineChunked)
    ->ArgNames({kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct({kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, sparse_prefix, "spin_selective",
                  SelectionPattern::kPrefix)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, sparse_one_per_chunk, "spin_selective",
                  SelectionPattern::kOnePerChunk)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, sparse_random, "spin_selective",
                  SelectionPattern::kRandomSortedUnique)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, dense_prefix, "spin", SelectionPattern::kPrefix)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, dense_one_per_chunk, "spin",
                  SelectionPattern::kOnePerChunk)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

BENCHMARK_CAPTURE(BM_ExecSelectiveChunked, dense_random, "spin",
                  SelectionPattern::kRandomSortedUnique)
    ->ArgNames(
        {kSelectivityArgName, kKernelIntensityArgName, kNumRowsArgName, kChunkSizeArgName})
    ->ArgsProduct(
        {kSelectivityChunkedArg, kKernelIntensityChunkedArg, kNumRowsArg, kChunkSizeArg});

}  // namespace arrow::compute
