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

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/acero/aggregate_node.h"
#include "arrow/acero/exec_plan.h"
#include "arrow/acero/options.h"
#include "arrow/acero/test_util_internal.h"
#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/concatenate.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/cast.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/compute/row/grouper.h"
#include "arrow/table.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/matchers.h"
#include "arrow/testing/random.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/logging.h"
#include "arrow/util/string.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/vector.h"

using testing::Eq;
using testing::HasSubstr;

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;
using internal::ToChars;

using compute::ArgShape;
using compute::CallFunction;
using compute::CountOptions;
using compute::default_exec_context;
using compute::ExecBatchFromJSON;
using compute::ExecSpan;
using compute::FunctionOptions;
using compute::Grouper;
using compute::PivotWiderOptions;
using compute::RowSegmenter;
using compute::ScalarAggregateOptions;
using compute::Segment;
using compute::SkewOptions;
using compute::SortIndices;
using compute::SortKey;
using compute::SortOrder;
using compute::Take;
using compute::TDigestOptions;
using compute::ValidateOutput;
using compute::VarianceOptions;

namespace acero {

TEST(AggregateSchema, NoKeys) {
  auto input_schema = schema({field("x", int32())});
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, HasSubstr("is a hash aggregate function"),
      aggregate::MakeOutputSchema(input_schema, {}, {},
                                  {{"hash_count", nullptr, "x", "hash_count"}}));
  ASSERT_OK_AND_ASSIGN(auto output_schema,
                       aggregate::MakeOutputSchema(input_schema, {}, {},
                                                   {{"count", nullptr, "x", "count"}}));
  AssertSchemaEqual(schema({field("count", int64())}), output_schema);
}

TEST(AggregateSchema, SingleKey) {
  auto input_schema = schema({field("x", int32()), field("y", int32())});
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, HasSubstr("is a scalar aggregate function"),
      aggregate::MakeOutputSchema(input_schema, {FieldRef("y")}, {},
                                  {{"count", nullptr, "x", "count"}}));
  ASSERT_OK_AND_ASSIGN(
      auto output_schema,
      aggregate::MakeOutputSchema(input_schema, {FieldRef("y")}, {},
                                  {{"hash_count", nullptr, "x", "hash_count"}}));
  AssertSchemaEqual(schema({field("y", int32()), field("hash_count", int64())}),
                    output_schema);
}

TEST(AggregateSchema, DoubleKey) {
  auto input_schema =
      schema({field("x", int32()), field("y", int32()), field("z", int32())});
  ASSERT_OK_AND_ASSIGN(
      auto output_schema,
      aggregate::MakeOutputSchema(input_schema, {FieldRef("z"), FieldRef("y")}, {},
                                  {{"hash_count", nullptr, "x", "hash_count"}}));
  AssertSchemaEqual(
      schema({field("z", int32()), field("y", int32()), field("hash_count", int64())}),
      output_schema);
}

TEST(AggregateSchema, SingleSegmentKey) {
  auto input_schema = schema({field("x", int32()), field("y", int32())});
  ASSERT_OK_AND_ASSIGN(auto output_schema,
                       aggregate::MakeOutputSchema(input_schema, {}, {FieldRef("y")},
                                                   {{"count", nullptr, "x", "count"}}));
  AssertSchemaEqual(schema({field("y", int32()), field("count", int64())}),
                    output_schema);
}

TEST(AggregateSchema, DoubleSegmentKey) {
  auto input_schema =
      schema({field("x", int32()), field("y", int32()), field("z", int32())});
  ASSERT_OK_AND_ASSIGN(
      auto output_schema,
      aggregate::MakeOutputSchema(input_schema, {}, {FieldRef("z"), FieldRef("y")},
                                  {{"count", nullptr, "x", "count"}}));
  AssertSchemaEqual(
      schema({field("z", int32()), field("y", int32()), field("count", int64())}),
      output_schema);
}

TEST(AggregateSchema, SingleKeyAndSegmentKey) {
  auto input_schema =
      schema({field("x", int32()), field("y", int32()), field("z", int32())});
  ASSERT_OK_AND_ASSIGN(
      auto output_schema,
      aggregate::MakeOutputSchema(input_schema, {FieldRef("y")}, {FieldRef("z")},
                                  {{"hash_count", nullptr, "x", "hash_count"}}));
  AssertSchemaEqual(
      schema({field("z", int32()), field("y", int32()), field("hash_count", int64())}),
      output_schema);
}

using GroupByFunction = std::function<Result<Datum>(
    const std::vector<Datum>&, const std::vector<Datum>&, const std::vector<Datum>&,
    const std::vector<Aggregate>&, bool, bool)>;

Result<Datum> NaiveGroupBy(std::vector<Datum> arguments, std::vector<Datum> keys,
                           const std::vector<Aggregate>& aggregates) {
  ARROW_ASSIGN_OR_RAISE(auto key_batch, ExecBatch::Make(std::move(keys)));

  ARROW_ASSIGN_OR_RAISE(auto grouper, Grouper::Make(key_batch.GetTypes()));

  ARROW_ASSIGN_OR_RAISE(Datum id_batch, grouper->Consume(ExecSpan(key_batch)));

  ARROW_ASSIGN_OR_RAISE(
      auto groupings,
      Grouper::MakeGroupings(*id_batch.array_as<UInt32Array>(), grouper->num_groups()));

  ArrayVector out_columns;
  std::vector<std::string> out_names;

  int key_idx = 0;
  ARROW_ASSIGN_OR_RAISE(auto uniques, grouper->GetUniques());
  std::vector<SortKey> sort_keys;
  std::vector<std::shared_ptr<Field>> sort_table_fields;
  for (const Datum& key : uniques.values) {
    out_columns.push_back(key.make_array());
    sort_keys.emplace_back(FieldRef(key_idx));
    sort_table_fields.push_back(field("key_" + ToChars(key_idx), key.type()));
    out_names.push_back("key_" + ToChars(key_idx++));
  }

  for (size_t i = 0; i < arguments.size(); ++i) {
    out_names.push_back(aggregates[i].function);

    // trim "hash_" prefix
    auto scalar_agg_function = aggregates[i].function.substr(5);

    ARROW_ASSIGN_OR_RAISE(
        auto grouped_argument,
        Grouper::ApplyGroupings(*groupings, *arguments[i].make_array()));

    ScalarVector aggregated_scalars;

    for (int64_t i_group = 0; i_group < grouper->num_groups(); ++i_group) {
      auto slice = grouped_argument->value_slice(i_group);
      if (slice->length() == 0) continue;
      ARROW_ASSIGN_OR_RAISE(Datum d, CallFunction(scalar_agg_function, {slice},
                                                  aggregates[i].options.get()));
      aggregated_scalars.push_back(d.scalar());
    }

    ARROW_ASSIGN_OR_RAISE(Datum aggregated_column,
                          ScalarVectorToArray(aggregated_scalars));
    out_columns.push_back(aggregated_column.make_array());
  }

  // Return a struct array sorted by the keys
  SortOptions sort_options(std::move(sort_keys));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<RecordBatch> sort_batch,
                        uniques.ToRecordBatch(schema(std::move(sort_table_fields))));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> sorted_indices,
                        SortIndices(sort_batch, sort_options));

  ARROW_ASSIGN_OR_RAISE(auto struct_arr,
                        StructArray::Make(std::move(out_columns), std::move(out_names)));
  return Take(struct_arr, sorted_indices);
}

Result<Datum> MakeGroupByOutput(const std::vector<ExecBatch>& output_batches,
                                const std::shared_ptr<Schema> output_schema,
                                size_t num_aggregates, size_t num_keys, bool naive) {
  ArrayVector out_arrays(num_aggregates + num_keys);
  for (size_t i = 0; i < out_arrays.size(); ++i) {
    std::vector<std::shared_ptr<Array>> arrays(output_batches.size());
    for (size_t j = 0; j < output_batches.size(); ++j) {
      arrays[j] = output_batches[j].values[i].make_array();
    }
    if (arrays.empty()) {
      ARROW_ASSIGN_OR_RAISE(
          out_arrays[i],
          MakeArrayOfNull(output_schema->field(static_cast<int>(i))->type(),
                          /*length=*/0));
    } else {
      ARROW_ASSIGN_OR_RAISE(out_arrays[i], Concatenate(arrays));
    }
  }

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<Array> struct_arr,
      StructArray::Make(std::move(out_arrays), output_schema->fields()));

  bool need_sort = !naive;
  for (size_t i = 0; need_sort && i < num_keys; i++) {
    if (output_schema->field(static_cast<int>(i))->type()->id() == Type::DICTIONARY) {
      need_sort = false;
    }
  }
  if (!need_sort) {
    return struct_arr;
  }

  // The exec plan may reorder the output rows.  The tests are all setup to expect output
  // in ascending order of keys.  So we need to sort the result by the key columns.  To do
  // that we create a table using the key columns, calculate the sort indices from that
  // table (sorting on all fields) and then use those indices to calculate our result.
  std::vector<std::shared_ptr<Field>> key_fields;
  std::vector<std::shared_ptr<Array>> key_columns;
  std::vector<SortKey> sort_keys;
  for (std::size_t i = 0; i < num_keys; i++) {
    const std::shared_ptr<Array>& arr = out_arrays[i];
    key_columns.push_back(arr);
    key_fields.push_back(field("name_does_not_matter", arr->type()));
    sort_keys.emplace_back(static_cast<int>(i));
  }
  std::shared_ptr<Schema> key_schema = schema(std::move(key_fields));
  std::shared_ptr<Table> key_table = Table::Make(std::move(key_schema), key_columns);
  SortOptions sort_options(std::move(sort_keys));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> sort_indices,
                        SortIndices(key_table, sort_options));
  return Take(struct_arr, sort_indices);
}

Result<Datum> RunGroupBy(const BatchesWithSchema& input,
                         const std::vector<std::string>& key_names,
                         const std::vector<std::string>& segment_key_names,
                         const std::vector<Aggregate>& aggregates, ExecContext* ctx,
                         bool use_threads, bool segmented = false, bool naive = false) {
  // The `use_threads` flag determines whether threads are used in generating the input to
  // the group-by.
  //
  // When segment_keys is non-empty the `segmented` flag is always true; otherwise (when
  // empty), it may still be set to true. In this case, the tester restructures (without
  // changing the data of) the result of RunGroupBy from `std::vector<ExecBatch>`
  // (output_batches) to `std::vector<ArrayVector>` (out_arrays), which have the structure
  // typical of the case of a non-empty segment_keys (with multiple arrays per column, one
  // array per segment) but only one array per column (because, technically, there is only
  // one segment in this case). Thus, this case focuses on the structure of the result.
  //
  // The `naive` flag means that the output is expected to be like that of `NaiveGroupBy`,
  // which in particular doesn't require sorting. The reason for the naive flag is that
  // the expected output of some test-cases is naive and of some others it is not. The
  // current `RunGroupBy` function deals with both kinds of expected output.
  std::vector<FieldRef> keys(key_names.size());
  for (size_t i = 0; i < key_names.size(); ++i) {
    keys[i] = FieldRef(key_names[i]);
  }
  std::vector<FieldRef> segment_keys(segment_key_names.size());
  for (size_t i = 0; i < segment_key_names.size(); ++i) {
    segment_keys[i] = FieldRef(segment_key_names[i]);
  }

  ARROW_ASSIGN_OR_RAISE(auto plan, ExecPlan::Make(*ctx));
  AsyncGenerator<std::optional<ExecBatch>> sink_gen;
  RETURN_NOT_OK(
      Declaration::Sequence(
          {
              {"source",
               SourceNodeOptions{input.schema, input.gen(use_threads, /*slow=*/false)}},
              {"aggregate", AggregateNodeOptions{aggregates, std::move(keys),
                                                 std::move(segment_keys)}},
              {"sink", SinkNodeOptions{&sink_gen}},
          })
          .AddToPlan(plan.get()));

  RETURN_NOT_OK(plan->Validate());
  plan->StartProducing();

  auto collected_fut = CollectAsyncGenerator(sink_gen);

  auto start_and_collect =
      AllFinished({plan->finished(), Future<>(collected_fut)})
          .Then([collected_fut]() -> Result<std::vector<ExecBatch>> {
            ARROW_ASSIGN_OR_RAISE(auto collected, collected_fut.result());
            return ::arrow::internal::MapVector(
                [](std::optional<ExecBatch> batch) {
                  return batch.value_or(ExecBatch());
                },
                std::move(collected));
          });

  ARROW_ASSIGN_OR_RAISE(std::vector<ExecBatch> output_batches,
                        start_and_collect.MoveResult());

  const auto& output_schema = plan->nodes()[0]->output()->output_schema();
  if (!segmented) {
    return MakeGroupByOutput(output_batches, output_schema, aggregates.size(),
                             key_names.size(), naive);
  }

  std::vector<ArrayVector> out_arrays(aggregates.size() + key_names.size() +
                                      segment_key_names.size());
  for (size_t i = 0; i < out_arrays.size(); ++i) {
    std::vector<std::shared_ptr<Array>> arrays(output_batches.size());
    for (size_t j = 0; j < output_batches.size(); ++j) {
      auto& value = output_batches[j].values[i];
      if (value.is_scalar()) {
        ARROW_ASSIGN_OR_RAISE(
            arrays[j], MakeArrayFromScalar(*value.scalar(), output_batches[j].length));
      } else if (value.is_array()) {
        arrays[j] = value.make_array();
      } else {
        return Status::Invalid("GroupByUsingExecPlan unsupported value kind ",
                               ToString(value.kind()));
      }
    }
    if (arrays.empty()) {
      arrays.resize(1);
      ARROW_ASSIGN_OR_RAISE(
          arrays[0], MakeArrayOfNull(output_schema->field(static_cast<int>(i))->type(),
                                     /*length=*/0));
    }
    out_arrays[i] = {std::move(arrays)};
  }

  if (segmented && segment_key_names.size() > 0) {
    ArrayVector struct_arrays;
    struct_arrays.reserve(output_batches.size());
    for (size_t j = 0; j < output_batches.size(); ++j) {
      ArrayVector struct_fields;
      struct_fields.reserve(out_arrays.size());
      for (auto out_array : out_arrays) {
        struct_fields.push_back(out_array[j]);
      }
      ARROW_ASSIGN_OR_RAISE(auto struct_array,
                            StructArray::Make(struct_fields, output_schema->fields()));
      struct_arrays.push_back(struct_array);
    }
    return ChunkedArray::Make(struct_arrays);
  } else {
    ArrayVector struct_fields(out_arrays.size());
    for (size_t i = 0; i < out_arrays.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(struct_fields[i], Concatenate(out_arrays[i]));
    }
    return StructArray::Make(std::move(struct_fields), output_schema->fields());
  }
}

Result<Datum> RunGroupBy(const BatchesWithSchema& input,
                         const std::vector<std::string>& key_names,
                         const std::vector<std::string>& segment_key_names,
                         const std::vector<Aggregate>& aggregates, bool use_threads,
                         bool segmented = false, bool naive = false) {
  if (!use_threads) {
    ARROW_ASSIGN_OR_RAISE(auto thread_pool, arrow::internal::ThreadPool::Make(1));
    ExecContext seq_ctx(default_memory_pool(), thread_pool.get());
    return RunGroupBy(input, key_names, segment_key_names, aggregates, &seq_ctx,
                      use_threads, segmented, naive);
  } else {
    return RunGroupBy(input, key_names, segment_key_names, aggregates,
                      threaded_exec_context(), use_threads, segmented, naive);
  }
}

Result<Datum> RunGroupBy(const BatchesWithSchema& input,
                         const std::vector<std::string>& key_names,
                         const std::vector<Aggregate>& aggregates, bool use_threads,
                         bool segmented = false, bool naive = false) {
  return RunGroupBy(input, key_names, {}, aggregates, use_threads, segmented);
}

/// Simpler overload where you can give the columns as datums
Result<Datum> RunGroupBy(const std::vector<Datum>& arguments,
                         const std::vector<Datum>& keys,
                         const std::vector<Datum>& segment_keys,
                         const std::vector<Aggregate>& aggregates, bool use_threads,
                         bool segmented = false, bool naive = false) {
  using arrow::compute::detail::ExecSpanIterator;

  FieldVector scan_fields(arguments.size() + keys.size() + segment_keys.size());
  std::vector<std::string> key_names(keys.size());
  std::vector<std::string> segment_key_names(segment_keys.size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto name = std::string("agg_") + ToChars(i);
    scan_fields[i] = field(name, arguments[i].type());
  }
  size_t base = arguments.size();
  for (size_t i = 0; i < keys.size(); ++i) {
    auto name = std::string("key_") + ToChars(i);
    scan_fields[base + i] = field(name, keys[i].type());
    key_names[i] = std::move(name);
  }
  base += keys.size();
  size_t j = keys.size();
  std::string prefix("key_");
  for (size_t i = 0; i < segment_keys.size(); ++i) {
    auto name = prefix + std::to_string(j++);
    scan_fields[base + i] = field(name, segment_keys[i].type());
    segment_key_names[i] = std::move(name);
  }

  std::vector<Datum> inputs = arguments;
  inputs.reserve(inputs.size() + keys.size() + segment_keys.size());
  inputs.insert(inputs.end(), keys.begin(), keys.end());
  inputs.insert(inputs.end(), segment_keys.begin(), segment_keys.end());

  ExecSpanIterator span_iterator;
  ARROW_ASSIGN_OR_RAISE(auto batch, ExecBatch::Make(inputs));
  RETURN_NOT_OK(span_iterator.Init(batch));
  BatchesWithSchema input;
  input.schema = schema(std::move(scan_fields));
  ExecSpan span;
  while (span_iterator.Next(&span)) {
    if (span.length == 0) continue;
    input.batches.push_back(span.ToExecBatch());
  }

  return RunGroupBy(input, key_names, segment_key_names, aggregates, use_threads,
                    segmented, naive);
}

Result<Datum> RunGroupByImpl(const std::vector<Datum>& arguments,
                             const std::vector<Datum>& keys,
                             const std::vector<Datum>& segment_keys,
                             const std::vector<Aggregate>& aggregates, bool use_threads,
                             bool naive = false) {
  return RunGroupBy(arguments, keys, segment_keys, aggregates, use_threads,
                    /*segmented=*/false, naive);
}

Result<Datum> RunSegmentedGroupByImpl(const std::vector<Datum>& arguments,
                                      const std::vector<Datum>& keys,
                                      const std::vector<Datum>& segment_keys,
                                      const std::vector<Aggregate>& aggregates,
                                      bool use_threads, bool naive = false) {
  return RunGroupBy(arguments, keys, segment_keys, aggregates, use_threads,
                    /*segmented=*/true, naive);
}

void ValidateGroupBy(GroupByFunction group_by, const std::vector<Aggregate>& aggregates,
                     std::vector<Datum> arguments, std::vector<Datum> keys,
                     bool naive = true) {
  ASSERT_OK_AND_ASSIGN(Datum expected, NaiveGroupBy(arguments, keys, aggregates));

  ASSERT_OK_AND_ASSIGN(Datum actual, group_by(arguments, keys, {}, aggregates,
                                              /*use_threads=*/false, naive));

  ASSERT_OK(expected.make_array()->ValidateFull());
  ValidateOutput(actual);

  AssertDatumsEqual(expected, actual, /*verbose=*/true);
}

ExecContext* small_chunksize_context(bool use_threads = false) {
  static ExecContext ctx,
      ctx_with_threads{default_memory_pool(), arrow::internal::GetCpuThreadPool()};
  ctx.set_exec_chunksize(2);
  ctx_with_threads.set_exec_chunksize(2);
  return use_threads ? &ctx_with_threads : &ctx;
}

struct TestAggregate {
  std::string function;
  std::shared_ptr<FunctionOptions> options;
};

Result<Datum> GroupByTest(GroupByFunction group_by, const std::vector<Datum>& arguments,
                          const std::vector<Datum>& keys,
                          const std::vector<Datum>& segment_keys,
                          const std::vector<TestAggregate>& aggregates,
                          bool use_threads) {
  std::vector<Aggregate> internal_aggregates;
  int idx = 0;
  for (auto t_agg : aggregates) {
    internal_aggregates.push_back(
        {t_agg.function, t_agg.options, "agg_" + ToChars(idx), t_agg.function});
    idx = idx + 1;
  }
  return group_by(arguments, keys, segment_keys, internal_aggregates, use_threads,
                  /*naive=*/false);
}

Result<Datum> GroupByTest(GroupByFunction group_by, const std::vector<Datum>& arguments,
                          const std::vector<Datum>& keys,
                          const std::vector<TestAggregate>& aggregates,
                          bool use_threads) {
  return GroupByTest(group_by, arguments, keys, {}, aggregates, use_threads);
}

void TestSegmentKey(GroupByFunction group_by, const std::shared_ptr<Table>& table,
                    Datum output, const std::vector<Datum>& segment_keys);

class GroupBy : public ::testing::TestWithParam<GroupByFunction> {
 public:
  void ValidateGroupBy(const std::vector<Aggregate>& aggregates,
                       std::vector<Datum> arguments, std::vector<Datum> keys,
                       bool naive = true) {
    acero::ValidateGroupBy(GetParam(), aggregates, arguments, keys, naive);
  }

  Result<Datum> GroupByTest(const std::vector<Datum>& arguments,
                            const std::vector<Datum>& keys,
                            const std::vector<Datum>& segment_keys,
                            const std::vector<TestAggregate>& aggregates,
                            bool use_threads) {
    return acero::GroupByTest(GetParam(), arguments, keys, segment_keys, aggregates,
                              use_threads);
  }

  Result<Datum> GroupByTest(const std::vector<Datum>& arguments,
                            const std::vector<Datum>& keys,
                            const std::vector<TestAggregate>& aggregates,
                            bool use_threads) {
    return acero::GroupByTest(GetParam(), arguments, keys, aggregates, use_threads);
  }

  // This is not named GroupByTest to avoid ambiguities between overloads
  Result<Datum> AltGroupBy(const std::vector<Datum>& arguments,
                           const std::vector<Datum>& keys,
                           const std::vector<Datum>& segment_keys,
                           const std::vector<Aggregate>& aggregates,
                           bool use_threads = false) {
    return GetParam()(arguments, keys, segment_keys, aggregates, use_threads,
                      /*naive=*/false);
  }

  Result<Datum> RunPivot(const std::shared_ptr<DataType>& key_type,
                         const std::shared_ptr<DataType>& value_type,
                         const PivotWiderOptions& options,
                         const std::shared_ptr<Table>& table, bool use_threads = false) {
    Aggregate agg{"hash_pivot_wider", std::make_shared<PivotWiderOptions>(options),
                  /*target=*/std::vector<FieldRef>{"agg_0", "agg_1"}, /*name=*/"out"};
    ARROW_ASSIGN_OR_RAISE(
        Datum aggregated_and_grouped,
        AltGroupBy({table->GetColumnByName("key"), table->GetColumnByName("value")},
                   {table->GetColumnByName("group_key")},
                   /*segment_keys=*/{}, {agg}, use_threads));
    ValidateOutput(aggregated_and_grouped);
    return aggregated_and_grouped;
  }

  Result<Datum> RunPivot(const std::shared_ptr<DataType>& key_type,
                         const std::shared_ptr<DataType>& value_type,
                         const PivotWiderOptions& options,
                         const std::vector<std::string>& table_json,
                         bool use_threads = false) {
    auto table =
        TableFromJSON(schema({field("group_key", int64()), field("key", key_type),
                              field("value", value_type)}),
                      table_json);
    return RunPivot(key_type, value_type, options, table, use_threads);
  }

  void CheckPivoted(const std::shared_ptr<DataType>& key_type,
                    const std::shared_ptr<DataType>& value_type,
                    const PivotWiderOptions& options, const Datum& pivoted,
                    const std::string& expected_json) {
    FieldVector pivoted_fields;
    for (const auto& key_name : options.key_names) {
      pivoted_fields.push_back(field(key_name, value_type));
    }
    auto expected_type = struct_({
        field("key_0", int64()),
        field("out", struct_(std::move(pivoted_fields))),
    });
    auto expected = ArrayFromJSON(expected_type, expected_json);
    AssertDatumsEqual(expected, pivoted, /*verbose=*/true);
  }

  void TestPivot(const std::shared_ptr<DataType>& key_type,
                 const std::shared_ptr<DataType>& value_type,
                 const PivotWiderOptions& options,
                 const std::vector<std::string>& table_json,
                 const std::string& expected_json, bool use_threads) {
    ASSERT_OK_AND_ASSIGN(
        auto pivoted, RunPivot(key_type, value_type, options, table_json, use_threads));
    CheckPivoted(key_type, value_type, options, pivoted, expected_json);
  }

  void TestPivot(const std::shared_ptr<DataType>& key_type,
                 const std::shared_ptr<DataType>& value_type,
                 const PivotWiderOptions& options,
                 const std::vector<std::string>& table_json,
                 const std::string& expected_json) {
    for (bool use_threads : {false, true}) {
      ARROW_SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      TestPivot(key_type, value_type, options, table_json, expected_json, use_threads);
    }
  }

  void TestSegmentKey(const std::shared_ptr<Table>& table, Datum output,
                      const std::vector<Datum>& segment_keys) {
    return acero::TestSegmentKey(GetParam(), table, output, segment_keys);
  }
};

TEST_P(GroupBy, Errors) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("group_id", uint32())}), R"([
    [1.0,   1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   0],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  0],
    [null,  3]
  ])");

  EXPECT_THAT(CallFunction("hash_sum", {batch->GetColumnByName("argument"),
                                        batch->GetColumnByName("group_id")}),
              Raises(StatusCode::NotImplemented,
                     HasSubstr("Direct execution of HASH_AGGREGATE functions")));
}

TEST_P(GroupBy, NoBatches) {
  // Regression test for ARROW-14583: handle when no batches are
  // passed to the group by node before finalizing
  auto table =
      TableFromJSON(schema({field("argument", float64()), field("key", int64())}), {});
  ASSERT_OK_AND_ASSIGN(
      Datum aggregated_and_grouped,
      GroupByTest({table->GetColumnByName("argument")}, {table->GetColumnByName("key")},
                  {
                      {"hash_count", nullptr},
                  },
                  /*use_threads=*/true));
  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_count", int64()),
                                  }),
                                  R"([])"),
                    aggregated_and_grouped, /*verbose=*/true);
}

namespace {
void SortBy(std::vector<std::string> names, Datum* aggregated_and_grouped) {
  SortOptions options;
  for (auto&& name : names) {
    options.sort_keys.emplace_back(std::move(name), SortOrder::Ascending);
  }

  ASSERT_OK_AND_ASSIGN(
      auto batch, RecordBatch::FromStructArray(aggregated_and_grouped->make_array()));

  // decode any dictionary columns:
  ArrayVector cols = batch->columns();
  for (auto& col : cols) {
    if (col->type_id() != Type::DICTIONARY) continue;

    auto dict_col = checked_cast<const DictionaryArray*>(col.get());
    ASSERT_OK_AND_ASSIGN(col, Take(*dict_col->dictionary(), *dict_col->indices()));
  }
  batch = RecordBatch::Make(batch->schema(), batch->num_rows(), std::move(cols));

  ASSERT_OK_AND_ASSIGN(Datum sort_indices, SortIndices(batch, options));

  ASSERT_OK_AND_ASSIGN(*aggregated_and_grouped,
                       Take(*aggregated_and_grouped, sort_indices));
}
}  // namespace

TEST_P(GroupBy, CountOnly) {
  const std::vector<std::string> json = {
      // Test inputs ("argument", "key")
      R"([[1.0,   1],
          [null,  1]])",
      R"([[0.0,   2],
          [null,  3],
          [null,  2],
          [4.0,   null],
          [3.25,  1],
          [3.25,  1],
          [0.125, 2]])",
      R"([[-0.25, 2],
          [0.75,  null],
          [null,  3]])",
  };
  const auto skip_nulls = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  const auto only_nulls = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  const auto count_all = std::make_shared<CountOptions>(CountOptions::ALL);
  const auto possible_count_options = std::vector<std::shared_ptr<CountOptions>>{
      nullptr,  // default = skip_nulls
      skip_nulls,
      only_nulls,
      count_all,
  };
  const auto expected_results = std::vector<std::string>{
      // Results ("key_0", "hash_count")
      // nullptr = skip_nulls
      R"([[1, 3],
          [2, 3],
          [3, 0],
          [null, 2]])",
      // skip_nulls
      R"([[1, 3],
          [2, 3],
          [3, 0],
          [null, 2]])",
      // only_nulls
      R"([[1, 1],
          [2, 1],
          [3, 2],
          [null, 0]])",
      // count_all
      R"([[1, 4],
          [2, 4],
          [3, 2],
          [null, 2]])",
  };
  // NOTE: the "key" column (1) does not appear in the possible run-end
  // encoding transformations because GroupBy kernels do not support run-end
  // encoded key arrays.
  for (const auto& re_encode_cols : std::vector<std::vector<int>>{{}, {0}}) {
    for (bool use_threads : {/*true, */ false}) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      for (size_t i = 0; i < possible_count_options.size(); i++) {
        SCOPED_TRACE(possible_count_options[i] ? possible_count_options[i]->ToString()
                                               : "default");
        auto table = TableFromJSON(
            schema({field("argument", float64()), field("key", int64())}), json);

        auto transformed_table = table;
        if (!re_encode_cols.empty()) {
          ASSERT_OK_AND_ASSIGN(transformed_table,
                               RunEndEncodeTableColumns(*table, re_encode_cols));
        }

        ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                             GroupByTest({transformed_table->GetColumnByName("argument")},
                                         {transformed_table->GetColumnByName("key")},
                                         {
                                             {"hash_count", possible_count_options[i]},
                                         },
                                         use_threads));
        SortBy({"key_0"}, &aggregated_and_grouped);

        AssertDatumsEqual(aggregated_and_grouped,
                          ArrayFromJSON(struct_({field("key_0", int64()),
                                                 field("hash_count", int64())}),
                                        expected_results[i]),
                          /*verbose=*/true);
      }
    }
  }
}

TEST_P(GroupBy, CountScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[1, 1], [1, 1], [1, 2], [1, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[null, 1], [null, 1], [null, 2], [null, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, "[[2, 1], [3, 2], [4, 3]]"),
  };
  input.schema = schema({field("argument", int32()), field("key", int64())});

  auto skip_nulls = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  auto keep_nulls = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  auto count_all = std::make_shared<CountOptions>(CountOptions::ALL);
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum actual, RunGroupBy(input, {"key"},
                                 {
                                     {"hash_count", skip_nulls, "argument", "hash_count"},
                                     {"hash_count", keep_nulls, "argument", "hash_count"},
                                     {"hash_count", count_all, "argument", "hash_count"},
                                 },
                                 use_threads));

    Datum expected = ArrayFromJSON(struct_({
                                       field("key", int64()),
                                       field("hash_count", int64()),
                                       field("hash_count", int64()),
                                       field("hash_count", int64()),
                                   }),
                                   R"([
      [1, 3, 2, 5],
      [2, 2, 1, 3],
      [3, 2, 1, 3]
    ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, SumOnly) {
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table =
        TableFromJSON(schema({field("argument", float64()), field("key", int64())}), {R"([
    [1.0,   1],
    [null,  1]
                        ])",
                                                                                      R"([
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2]
                        ])",
                                                                                      R"([
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
                        ])"});

    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        GroupByTest({table->GetColumnByName("argument")}, {table->GetColumnByName("key")},
                    {
                        {"hash_sum", nullptr},
                    },
                    use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_sum", float64()),
                                    }),
                                    R"([
    [1, 4.25],
    [2, -0.125],
    [3, null],
    [null, 4.75]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, SumMeanProductDecimal) {
  auto in_schema = schema({
      field("argument0", decimal128(3, 2)),
      field("argument1", decimal256(3, 2)),
      field("key", int64()),
  });

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(in_schema, {R"([
    ["1.00",  "1.00",  1],
    [null,    null,    1]
  ])",
                                           R"([
    ["0.00",  "0.00",  2],
    [null,    null,    3],
    ["4.00",  "4.00",  null],
    ["3.25",  "3.25",  1],
    ["0.12",  "0.12",  2]
  ])",
                                           R"([
    ["-0.25", "-0.25", 2],
    ["0.75",  "0.75",  null],
    [null,    null,    3],
    ["1.01",  "1.01",  4],
    ["1.01",  "1.01",  4],
    ["1.01",  "1.01",  4],
    ["1.02",  "1.02",  4]
  ])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument1"),
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument1"),
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument1"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_sum", nullptr},
                                 {"hash_sum", nullptr},
                                 {"hash_mean", nullptr},
                                 {"hash_mean", nullptr},
                                 {"hash_product", nullptr},
                                 {"hash_product", nullptr},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_sum", decimal128(38, 2)),
                                        field("hash_sum", decimal256(76, 2)),
                                        field("hash_mean", decimal128(3, 2)),
                                        field("hash_mean", decimal256(3, 2)),
                                        field("hash_product", decimal128(3, 2)),
                                        field("hash_product", decimal256(3, 2)),
                                    }),
                                    R"([
    [1, "4.25",  "4.25",  "2.13",  "2.13",  "3.25", "3.25"],
    [2, "-0.13", "-0.13", "-0.04", "-0.04", "0.00", "0.00"],
    [3, null,    null,    null,    null,    null,   null],
    [4, "4.05",  "4.05",  "1.01",  "1.01",  "1.05", "1.05"],
    [null, "4.75",  "4.75",  "2.38",  "2.38",  "3.00", "3.00"]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, MeanOnly) {
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table =
        TableFromJSON(schema({field("argument", float64()), field("key", int64())}), {R"([
    [1.0,   1],
    [null,  1]
                        ])",
                                                                                      R"([
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2]
                        ])",
                                                                                      R"([
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
                        ])"});

    auto min_count =
        std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest({table->GetColumnByName("argument"),
                                      table->GetColumnByName("argument")},
                                     {table->GetColumnByName("key")},
                                     {
                                         {"hash_mean", nullptr},
                                         {"hash_mean", min_count},
                                     },
                                     use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                              field("key_0", int64()),
                                              field("hash_mean", float64()),
                                              field("hash_mean", float64()),
                                          }),
                                          R"([
    [1,    2.125,                 null                 ],
    [2,    -0.041666666666666664, -0.041666666666666664],
    [3,    null,                  null                 ],
    [null, 2.375,                 null                 ]
  ])"),
                            aggregated_and_grouped,
                            /*verbose=*/true);
  }
}

TEST_P(GroupBy, SumMeanProductScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},

                        "[[1, 1], [1, 1], [1, 2], [1, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[null, 1], [null, 1], [null, 2], [null, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, "[[2, 1], [3, 2], [4, 3]]"),
  };
  input.schema = schema({field("argument", int32()), field("key", int64())});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum actual,
        RunGroupBy(input, {"key"},
                   {
                       {"hash_sum", nullptr, "argument", "hash_sum"},
                       {"hash_mean", nullptr, "argument", "hash_mean"},
                       {"hash_product", nullptr, "argument", "hash_product"},
                   },
                   use_threads));
    Datum expected = ArrayFromJSON(struct_({
                                       field("key", int64()),
                                       field("hash_sum", int64()),
                                       field("hash_mean", float64()),
                                       field("hash_product", int64()),
                                   }),
                                   R"([
      [1, 4, 1.333333, 2],
      [2, 4, 2,        3],
      [3, 5, 2.5,      4]
    ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, MeanOverflow) {
  BatchesWithSchema input;
  // would overflow if intermediate sum is integer
  input.batches = {
      ExecBatchFromJSON({int64(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},

                        "[[9223372036854775805, 1], [9223372036854775805, 1], "
                        "[9223372036854775805, 2], [9223372036854775805, 3]]"),
      ExecBatchFromJSON({int64(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[null, 1], [null, 1], [null, 2], [null, 3]]"),
      ExecBatchFromJSON({int64(), int64()},
                        "[[9223372036854775805, 1], [9223372036854775805, 2], "
                        "[9223372036854775805, 3]]"),
  };
  input.schema = schema({field("argument", int64()), field("key", int64())});
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum actual,
                         RunGroupBy(input, {"key"},
                                    {
                                        {"hash_mean", nullptr, "argument", "hash_mean"},
                                    },
                                    use_threads));
    Datum expected = ArrayFromJSON(struct_({
                                       field("key", int64()),
                                       field("hash_mean", float64()),
                                   }),
                                   R"([
      [1, 9223372036854775805],
      [2, 9223372036854775805],
      [3, 9223372036854775805]
    ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, VarianceStddevSkewKurtosis) {
  for (auto value_type : {int32(), float64()}) {
    ARROW_SCOPED_TRACE("value_type = ", *value_type);
    auto batch = RecordBatchFromJSON(
        schema({field("argument", value_type), field("key", int64())}), R"([
      [1,   1],
      [null,  1],
      [0,   2],
      [null,  3],
      [4,   null],
      [3,  1],
      [0, 2],
      [-1, 2],
      [1,  null],
      [null,  3]
    ])");

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                             },
                             {
                                 batch->GetColumnByName("key"),
                             },
                             {},
                             {
                                 {"hash_variance", nullptr},
                                 {"hash_stddev", nullptr},
                                 {"hash_skew", nullptr},
                                 {"hash_kurtosis", nullptr},
                             },
                             false));

    auto expected = ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_variance", float64()),
                                      field("hash_stddev", float64()),
                                      field("hash_skew", float64()),
                                      field("hash_kurtosis", float64()),
                                  }),
                                  R"([
      [1,    1.0,                 1.0,                0.0,                 -2.0],
      [2,    0.22222222222222224, 0.4714045207910317, -0.7071067811865478, -1.5],
      [3,    null,                null,               null,                null],
      [null, 2.25,                1.5,                0.0,                 -2.0]
    ])");
    AssertDatumsApproxEqual(expected, aggregated_and_grouped,
                            /*verbose=*/true);
  }
}

TEST_P(GroupBy, VarianceAndStddevDdof) {
  // Test ddof
  auto variance_options = std::make_shared<VarianceOptions>(/*ddof=*/2);

  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [1,   1],
    [null,  1],
    [0,   2],
    [null,  3],
    [4,   null],
    [3,  1],
    [0, 2],
    [-1, 2],
    [1,  null],
    [null,  3]
  ])");
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       GroupByTest(
                           {
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                           },
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_variance", variance_options},
                               {"hash_stddev", variance_options},
                           },
                           false));

  AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                            field("key_0", int64()),
                                            field("hash_variance", float64()),
                                            field("hash_stddev", float64()),
                                        }),
                                        R"([
    [1,    null,                null             ],
    [2,    0.6666666666666667,  0.816496580927726],
    [3,    null,                null             ],
    [null, null,                null             ]
  ])"),
                          aggregated_and_grouped,
                          /*verbose=*/true);
}

TEST_P(GroupBy, VarianceStddevSkewKurtosisDecimal) {
  for (auto value_type :
       {decimal32(3, 2), decimal64(3, 2), decimal128(3, 2), decimal256(3, 2)}) {
    ARROW_SCOPED_TRACE("value_type = ", *value_type);
    auto batch = RecordBatchFromJSON(
        schema({field("argument", value_type), field("key", int64())}),
        R"([
      ["1.00",  1],
      [null,    1],
      ["0.00",  2],
      ["4.00",  null],
      ["3.00",  1],
      ["0.00",  2],
      ["-1.00", 2],
      ["1.00",  null]
    ])");

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                             },
                             {
                                 batch->GetColumnByName("key"),
                             },
                             {},
                             {
                                 {"hash_variance", nullptr},
                                 {"hash_stddev", nullptr},
                                 {"hash_skew", nullptr},
                                 {"hash_kurtosis", nullptr},
                             },
                             false));

    auto expected = ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_variance", float64()),
                                      field("hash_stddev", float64()),
                                      field("hash_skew", float64()),
                                      field("hash_kurtosis", float64()),
                                  }),
                                  R"([
    [1,    1.0,                 1.0,                0.0,                 -2.0],
    [2,    0.22222222222222224, 0.4714045207910317, -0.7071067811865478, -1.5],
    [null, 2.25,                1.5,                0.0,                 -2.0]
  ])");

    AssertDatumsApproxEqual(expected, aggregated_and_grouped,
                            /*verbose=*/true);
  }
}

TEST_P(GroupBy, TDigest) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [1,    1],
    [null, 1],
    [0,    2],
    [null, 3],
    [1,    4],
    [4,    null],
    [3,    1],
    [0,    2],
    [-1,   2],
    [1,    null],
    [NaN,  3],
    [1,    4],
    [1,    4],
    [null, 4]
  ])");

  auto options1 = std::make_shared<TDigestOptions>(std::vector<double>{0.5, 0.9, 0.99});
  auto options2 =
      std::make_shared<TDigestOptions>(std::vector<double>{0.5, 0.9, 0.99}, /*delta=*/50,
                                       /*buffer_size=*/1024);
  auto keep_nulls =
      std::make_shared<TDigestOptions>(/*q=*/0.5, /*delta=*/100, /*buffer_size=*/500,
                                       /*skip_nulls=*/false, /*min_count=*/0);
  auto min_count =
      std::make_shared<TDigestOptions>(/*q=*/0.5, /*delta=*/100, /*buffer_size=*/500,
                                       /*skip_nulls=*/true, /*min_count=*/3);
  auto keep_nulls_min_count =
      std::make_shared<TDigestOptions>(/*q=*/0.5, /*delta=*/100, /*buffer_size=*/500,
                                       /*skip_nulls=*/false, /*min_count=*/3);
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       GroupByTest(
                           {
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                           },
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_tdigest", nullptr},
                               {"hash_tdigest", options1},
                               {"hash_tdigest", options2},
                               {"hash_tdigest", keep_nulls},
                               {"hash_tdigest", min_count},
                               {"hash_tdigest", keep_nulls_min_count},
                           },
                           false));

  AssertDatumsApproxEqual(
      ArrayFromJSON(struct_({
                        field("key_0", int64()),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                        field("hash_tdigest", fixed_size_list(float64(), 3)),
                        field("hash_tdigest", fixed_size_list(float64(), 3)),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                    }),
                    R"([
    [1,    [1.0],  [1.0, 3.0, 3.0],    [1.0, 3.0, 3.0],    [null], [null], [null]],
    [2,    [0.0],  [0.0, 0.0, 0.0],    [0.0, 0.0, 0.0],    [0.0],  [0.0],  [0.0] ],
    [3,    [null], [null, null, null], [null, null, null], [null], [null], [null]],
    [4,    [1.0],  [1.0, 1.0, 1.0],    [1.0, 1.0, 1.0],    [null], [1.0],  [null]],
    [null, [1.0],  [1.0, 4.0, 4.0],    [1.0, 4.0, 4.0],    [1.0],  [null], [null]]
  ])"),
      aggregated_and_grouped,
      /*verbose=*/true);
}

TEST_P(GroupBy, TDigestDecimal) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument0", decimal128(3, 2)), field("argument1", decimal256(3, 2)),
              field("key", int64())}),
      R"([
    ["1.01",  "1.01",  1],
    [null,    null,    1],
    ["0.00",  "0.00",  2],
    ["4.42",  "4.42",  null],
    ["3.86",  "3.86",  1],
    ["0.00",  "0.00",  2],
    ["-1.93", "-1.93", 2],
    ["1.85",  "1.85",  null]
  ])");

  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       GroupByTest(
                           {
                               batch->GetColumnByName("argument0"),
                               batch->GetColumnByName("argument1"),
                           },
                           {batch->GetColumnByName("key")},
                           {
                               {"hash_tdigest", nullptr},
                               {"hash_tdigest", nullptr},
                           },
                           false));

  AssertDatumsApproxEqual(
      ArrayFromJSON(struct_({
                        field("key_0", int64()),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                        field("hash_tdigest", fixed_size_list(float64(), 1)),
                    }),
                    R"([
    [1,    [1.01], [1.01]],
    [2,    [0.0],  [0.0] ],
    [null, [1.85], [1.85]]
  ])"),
      aggregated_and_grouped,
      /*verbose=*/true);
}

TEST_P(GroupBy, ApproximateMedian) {
  for (const auto& type : {float64(), int8()}) {
    auto batch =
        RecordBatchFromJSON(schema({field("argument", type), field("key", int64())}), R"([
    [1,    1],
    [null, 1],
    [0,    2],
    [null, 3],
    [1,    4],
    [4,    null],
    [3,    1],
    [0,    2],
    [-1,   2],
    [1,    null],
    [null, 3],
    [1,    4],
    [1,    4],
    [null, 4]
  ])");

    std::shared_ptr<ScalarAggregateOptions> options;
    auto keep_nulls = std::make_shared<ScalarAggregateOptions>(
        /*skip_nulls=*/false, /*min_count=*/0);
    auto min_count = std::make_shared<ScalarAggregateOptions>(
        /*skip_nulls=*/true, /*min_count=*/3);
    auto keep_nulls_min_count = std::make_shared<ScalarAggregateOptions>(
        /*skip_nulls=*/false, /*min_count=*/3);
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                                 batch->GetColumnByName("argument"),
                             },
                             {
                                 batch->GetColumnByName("key"),
                             },
                             {},
                             {
                                 {"hash_approximate_median", options},
                                 {"hash_approximate_median", keep_nulls},
                                 {"hash_approximate_median", min_count},
                                 {"hash_approximate_median", keep_nulls_min_count},
                             },
                             false));

    AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                              field("key_0", int64()),
                                              field("hash_approximate_median", float64()),
                                              field("hash_approximate_median", float64()),
                                              field("hash_approximate_median", float64()),
                                              field("hash_approximate_median", float64()),
                                          }),
                                          R"([
    [1,    1.0,  null, null, null],
    [2,    0.0,  0.0,  0.0,  0.0 ],
    [3,    null, null, null, null],
    [4,    1.0,  null, 1.0,  null],
    [null, 1.0,  1.0,  null, null]
  ])"),
                            aggregated_and_grouped,
                            /*verbose=*/true);
  }
}

TEST_P(GroupBy, StddevVarianceTDigestScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), float32(), int64()},
                        {ArgShape::SCALAR, ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[1, 1.0, 1], [1, 1.0, 1], [1, 1.0, 2], [1, 1.0, 3]]"),
      ExecBatchFromJSON(
          {int32(), float32(), int64()},
          {ArgShape::SCALAR, ArgShape::SCALAR, ArgShape::ARRAY},
          "[[null, null, 1], [null, null, 1], [null, null, 2], [null, null, 3]]"),
      ExecBatchFromJSON({int32(), float32(), int64()},
                        "[[2, 2.0, 1], [3, 3.0, 2], [4, 4.0, 3]]"),
  };
  input.schema = schema(
      {field("argument", int32()), field("argument1", float32()), field("key", int64())});

  for (bool use_threads : {false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum actual,
        RunGroupBy(input, {"key"},
                   {
                       {"hash_stddev", nullptr, "argument", "hash_stddev"},
                       {"hash_variance", nullptr, "argument", "hash_variance"},
                       {"hash_tdigest", nullptr, "argument", "hash_tdigest"},
                       {"hash_stddev", nullptr, "argument1", "hash_stddev"},
                       {"hash_variance", nullptr, "argument1", "hash_variance"},
                       {"hash_tdigest", nullptr, "argument1", "hash_tdigest"},
                   },
                   use_threads));
    Datum expected =
        ArrayFromJSON(struct_({
                          field("key", int64()),
                          field("hash_stddev", float64()),
                          field("hash_variance", float64()),
                          field("hash_tdigest", fixed_size_list(float64(), 1)),
                          field("hash_stddev", float64()),
                          field("hash_variance", float64()),
                          field("hash_tdigest", fixed_size_list(float64(), 1)),
                      }),
                      R"([
         [1, 0.4714045, 0.222222, [1.0], 0.4714045, 0.222222, [1.0]],
         [2, 1.0,       1.0,      [1.0], 1.0,       1.0,      [1.0]],
         [3, 1.5,       2.25,     [1.0], 1.5,       2.25,     [1.0]]
       ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, VarianceOptionsAndSkewOptions) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON(
          {int32(), float32(), int64()},
          {ArgShape::SCALAR, ArgShape::SCALAR, ArgShape::ARRAY},
          "[[1, 1.0, 1], [1, 1.0, 1], [1, 1.0, 2], [1, 1.0, 2], [1, 1.0, 3]]"),
      ExecBatchFromJSON({int32(), float32(), int64()},
                        {ArgShape::SCALAR, ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[1, 1.0, 4], [1, 1.0, 4]]"),
      ExecBatchFromJSON({int32(), float32(), int64()},
                        {ArgShape::SCALAR, ArgShape::SCALAR, ArgShape::ARRAY},

                        "[[null, null, 1]]"),
      ExecBatchFromJSON({int32(), float32(), int64()}, "[[2, 2.0, 1], [3, 3.0, 2]]"),
      ExecBatchFromJSON({int32(), float32(), int64()}, "[[4, 4.0, 2], [2, 2.0, 4]]"),
      ExecBatchFromJSON({int32(), float32(), int64()}, "[[null, null, 4], [6, 6.0, 3]]"),
  };
  input.schema = schema(
      {field("argument", int32()), field("argument1", float32()), field("key", int64())});

  auto var_keep_nulls =
      std::make_shared<VarianceOptions>(/*ddof=*/0, /*skip_nulls=*/false,
                                        /*min_count=*/0);
  auto var_min_count =
      std::make_shared<VarianceOptions>(/*ddof=*/0, /*skip_nulls=*/true, /*min_count=*/3);
  auto var_keep_nulls_min_count = std::make_shared<VarianceOptions>(
      /*ddof=*/0, /*skip_nulls=*/false, /*min_count=*/3);

  auto skew_keep_nulls = std::make_shared<SkewOptions>(/*skip_nulls=*/false,
                                                       /*biased=*/true,
                                                       /*min_count=*/0);
  auto skew_min_count = std::make_shared<SkewOptions>(/*skip_nulls=*/true,
                                                      /*biased=*/true, /*min_count=*/3);
  auto skew_keep_nulls_min_count = std::make_shared<SkewOptions>(
      /*skip_nulls=*/false, /*biased=*/true, /*min_count=*/3);

  auto skew_unbiased = std::make_shared<SkewOptions>(
      /*skip_nulls=*/false, /*biased=*/false, /*min_count=*/0);

  for (std::string value_column : {"argument", "argument1"}) {
    for (bool use_threads : {false}) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      ASSERT_OK_AND_ASSIGN(
          Datum actual,
          RunGroupBy(
              input, {"key"},
              {
                  {"hash_stddev", var_keep_nulls, value_column, "hash_stddev"},
                  {"hash_stddev", var_min_count, value_column, "hash_stddev"},
                  {"hash_stddev", var_keep_nulls_min_count, value_column, "hash_stddev"},
                  {"hash_variance", var_keep_nulls, value_column, "hash_variance"},
                  {"hash_variance", var_min_count, value_column, "hash_variance"},
                  {"hash_variance", var_keep_nulls_min_count, value_column,
                   "hash_variance"},
              },
              use_threads));
      Datum expected = ArrayFromJSON(struct_({
                                         field("key", int64()),
                                         field("hash_stddev", float64()),
                                         field("hash_stddev", float64()),
                                         field("hash_stddev", float64()),
                                         field("hash_variance", float64()),
                                         field("hash_variance", float64()),
                                         field("hash_variance", float64()),
                                     }),
                                     R"([
         [1, null,    0.471405, null,    null,   0.222222, null  ],
         [2, 1.29904, 1.29904,  1.29904, 1.6875, 1.6875,   1.6875],
         [3, 2.5,     null,     null,    6.25,   null,     null  ],
         [4, null,    0.471405, null,    null,   0.222222, null  ]
         ])");
      ValidateOutput(actual);
      AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);

      ASSERT_OK_AND_ASSIGN(
          actual,
          RunGroupBy(
              input, {"key"},
              {
                  {"hash_skew", skew_keep_nulls, value_column, "hash_skew"},
                  {"hash_skew", skew_min_count, value_column, "hash_skew"},
                  {"hash_skew", skew_keep_nulls_min_count, value_column, "hash_skew"},
                  {"hash_skew", skew_unbiased, value_column, "hash_skew"},
                  {"hash_kurtosis", skew_keep_nulls, value_column, "hash_kurtosis"},
                  {"hash_kurtosis", skew_min_count, value_column, "hash_kurtosis"},
                  {"hash_kurtosis", skew_keep_nulls_min_count, value_column,
                   "hash_kurtosis"},
                  {"hash_kurtosis", skew_unbiased, value_column, "hash_kurtosis"},
              },
              use_threads));
      expected = ArrayFromJSON(struct_({
                                   field("key", int64()),
                                   field("hash_skew", float64()),
                                   field("hash_skew", float64()),
                                   field("hash_skew", float64()),
                                   field("hash_skew", float64()),
                                   field("hash_kurtosis", float64()),
                                   field("hash_kurtosis", float64()),
                                   field("hash_kurtosis", float64()),
                                   field("hash_kurtosis", float64()),
                               }),
                               R"([
         [1, null,      0.707106,  null,     null,    null,      -1.5,      null,      null    ],
         [2, 0.213833,  0.213833,  0.213833, 0.37037, -1.720164, -1.720164, -1.720164, -3.90123],
         [3, 0.0,       null,      null,     null,    -2.0,       null,     null,      null    ],
         [4, null,      0.707106,  null,     null,    null,      -1.5,      null,      null    ]
         ])");
      ValidateOutput(actual);
      AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
    }
  }
}

TEST_P(GroupBy, MinMaxOnly) {
  auto in_schema = schema({
      field("argument", float64()),
      field("argument1", null()),
      field("argument2", boolean()),
      field("key", int64()),
  });
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(in_schema, {R"([
    [1.0,   null, true, 1],
    [null,  null, true, 1]
])",
                                           R"([
    [0.0,   null, false, 2],
    [null,  null, false, 3],
    [4.0,   null, null,  null],
    [3.25,  null, true,  1],
    [0.125, null, false, 2]
])",
                                           R"([
    [-0.25, null, false, 2],
    [0.75,  null, true,  null],
    [null,  null, true,  3]
])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument1"),
                                 table->GetColumnByName("argument2"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_min_max", nullptr},
                                 {"hash_min_max", nullptr},
                                 {"hash_min_max", nullptr},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_min_max", struct_({
                                                                  field("min", float64()),
                                                                  field("max", float64()),
                                                              })),
                                        field("hash_min_max", struct_({
                                                                  field("min", null()),
                                                                  field("max", null()),
                                                              })),
                                        field("hash_min_max", struct_({
                                                                  field("min", boolean()),
                                                                  field("max", boolean()),
                                                              })),
                                    }),
                                    R"([
    [1, {"min": 1.0,   "max": 3.25},  {"min": null, "max": null}, {"min": true, "max": true}   ],
    [2, {"min": -0.25, "max": 0.125}, {"min": null, "max": null}, {"min": false, "max": false} ],
    [3, {"min": null,  "max": null},  {"min": null, "max": null}, {"min": false, "max": true}  ],
    [null, {"min": 0.75,  "max": 4.0},   {"min": null, "max": null}, {"min": true, "max": true}]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, MinMaxTypes) {
  std::vector<std::shared_ptr<DataType>> types;
  types.insert(types.end(), NumericTypes().begin(), NumericTypes().end());
  types.insert(types.end(), TemporalTypes().begin(), TemporalTypes().end());
  types.push_back(month_interval());

  const std::vector<std::string> default_table = {R"([
    [1,    1],
    [null, 1]
])",
                                                  R"([
    [0,    2],
    [null, 3],
    [3,    4],
    [5,    4],
    [4,    null],
    [3,    1],
    [0,    2]
])",
                                                  R"([
    [0,    2],
    [1,    null],
    [null, 3]
])"};

  const std::vector<std::string> date64_table = {R"([
    [86400000,    1],
    [null, 1]
])",
                                                 R"([
    [0,    2],
    [null, 3],
    [259200000,    4],
    [432000000,    4],
    [345600000,    null],
    [259200000,    1],
    [0,    2]
])",
                                                 R"([
    [0,    2],
    [86400000,    null],
    [null, 3]
])"};

  const std::string default_expected =
      R"([
    [1,    {"min": 1, "max": 3}      ],
    [2,    {"min": 0, "max": 0}      ],
    [3,    {"min": null, "max": null}],
    [4,    {"min": 3, "max": 5}      ],
    [null, {"min": 1, "max": 4}   ]
    ])";

  const std::string date64_expected =
      R"([
    [1,    {"min": 86400000, "max": 259200000} ],
    [2,    {"min": 0, "max": 0}                ],
    [3,    {"min": null, "max": null}          ],
    [4,    {"min": 259200000, "max": 432000000}],
    [null, {"min": 86400000, "max": 345600000} ]
    ])";

  for (const auto& ty : types) {
    SCOPED_TRACE(ty->ToString());
    auto in_schema = schema({field("argument0", ty), field("key", int64())});
    auto table =
        TableFromJSON(in_schema, (ty->name() == "date64") ? date64_table : default_table);

    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        GroupByTest({table->GetColumnByName("argument0")},
                    {table->GetColumnByName("key")}, {{"hash_min_max", nullptr}},
                    /*use_threads=*/true));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(
        ArrayFromJSON(
            struct_({
                field("key_0", int64()),
                field("hash_min_max", struct_({field("min", ty), field("max", ty)})),
            }),
            (ty->name() == "date64") ? date64_expected : default_expected),
        aggregated_and_grouped,
        /*verbose=*/true);
  }
}

TEST_P(GroupBy, MinMaxDecimal) {
  auto in_schema = schema({
      field("argument0", decimal128(3, 2)),
      field("argument1", decimal256(3, 2)),
      field("key", int64()),
  });
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(in_schema, {R"([
    ["1.01", "1.01",   1],
    [null,   null,     1]
                        ])",
                                           R"([
    ["0.00", "0.00",   2],
    [null,   null,     3],
    ["-3.25", "-3.25", 4],
    ["-5.25", "-5.25", 4],
    ["4.01", "4.01",   null],
    ["3.25", "3.25",   1],
    ["0.12", "0.12",   2]
                        ])",
                                           R"([
    ["-0.25", "-0.25", 2],
    ["0.75",  "0.75",  null],
    [null,    null,    3]
                        ])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument1"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_min_max", nullptr},
                                 {"hash_min_max", nullptr},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(
        ArrayFromJSON(struct_({
                          field("key_0", int64()),
                          field("hash_min_max", struct_({
                                                    field("min", decimal128(3, 2)),
                                                    field("max", decimal128(3, 2)),
                                                })),
                          field("hash_min_max", struct_({
                                                    field("min", decimal256(3, 2)),
                                                    field("max", decimal256(3, 2)),
                                                })),
                      }),
                      R"([
    [1,    {"min": "1.01", "max": "3.25"},   {"min": "1.01", "max": "3.25"}  ],
    [2,    {"min": "-0.25", "max": "0.12"},  {"min": "-0.25", "max": "0.12"} ],
    [3,    {"min": null, "max": null},       {"min": null, "max": null}      ],
    [4,    {"min": "-5.25", "max": "-3.25"}, {"min": "-5.25", "max": "-3.25"}],
    [null, {"min": "0.75", "max": "4.01"},   {"min": "0.75", "max": "4.01"}  ]
  ])"),
        aggregated_and_grouped,
        /*verbose=*/true);
  }
}

TEST_P(GroupBy, MinMaxBinary) {
  for (bool use_threads : {true, false}) {
    for (const auto& ty : BaseBinaryTypes()) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

      auto table = TableFromJSON(schema({
                                     field("argument0", ty),
                                     field("key", int64()),
                                 }),
                                 {R"([
    ["aaaa", 1],
    [null,   1]
])",
                                  R"([
    ["bcd",  2],
    [null,   3],
    ["2",    null],
    ["d",    1],
    ["bc",   2]
])",
                                  R"([
    ["babcd", 2],
    ["123",   null],
    [null,    3]
])"});

      ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                           GroupByTest({table->GetColumnByName("argument0")},
                                       {table->GetColumnByName("key")},
                                       {{"hash_min_max", nullptr}}, use_threads));
      ValidateOutput(aggregated_and_grouped);
      SortBy({"key_0"}, &aggregated_and_grouped);

      AssertDatumsEqual(
          ArrayFromJSON(
              struct_({
                  field("key_0", int64()),
                  field("hash_min_max", struct_({field("min", ty), field("max", ty)})),
              }),
              R"([
    [1,    {"min": "aaaa", "max": "d"}   ],
    [2,    {"min": "babcd", "max": "bcd"}],
    [3,    {"min": null, "max": null}    ],
    [null, {"min": "123", "max": "2"}    ]
  ])"),
          aggregated_and_grouped,
          /*verbose=*/true);
    }
  }
}

TEST_P(GroupBy, MinMaxFixedSizeBinary) {
  const auto ty = fixed_size_binary(3);
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(schema({
                                   field("argument0", ty),
                                   field("key", int64()),
                               }),
                               {R"([
    ["aaa", 1],
    [null,  1]
])",
                                R"([
    ["bac", 2],
    [null,  3],
    ["234", null],
    ["ddd", 1],
    ["bcd", 2]
])",
                                R"([
    ["bab", 2],
    ["123", null],
    [null,  3]
])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest({table->GetColumnByName("argument0")},
                                     {table->GetColumnByName("key")},
                                     {{"hash_min_max", nullptr}}, use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(
        ArrayFromJSON(
            struct_({
                field("key_0", int64()),
                field("hash_min_max", struct_({field("min", ty), field("max", ty)})),
            }),
            R"([
    [1,    {"min": "aaa", "max": "ddd"}],
    [2,    {"min": "bab", "max": "bcd"}],
    [3,    {"min": null, "max": null}  ],
    [null, {"min": "123", "max": "234"}]
  ])"),
        aggregated_and_grouped,
        /*verbose=*/true);
  }
}

TEST_P(GroupBy, MinOrMax) {
  auto table =
      TableFromJSON(schema({field("argument", float64()), field("key", int64())}), {R"([
    [1.0,   1],
    [null,  1]
])",
                                                                                    R"([
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2]
])",
                                                                                    R"([
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
])",
                                                                                    R"([
    [NaN,   4],
    [null,  4],
    [Inf,   4],
    [-Inf,  4],
    [0.0,   4]
])"});

  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       GroupByTest({table->GetColumnByName("argument"),
                                    table->GetColumnByName("argument")},
                                   {table->GetColumnByName("key")},
                                   {
                                       {"hash_min", nullptr},
                                       {"hash_max", nullptr},
                                   },
                                   /*use_threads=*/true));
  SortBy({"key_0"}, &aggregated_and_grouped);

  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_min", float64()),
                                      field("hash_max", float64()),
                                  }),
                                  R"([
    [1,    1.0,   3.25 ],
    [2,    -0.25, 0.125],
    [3,    null,  null ],
    [4,    -Inf,  Inf  ],
    [null, 0.75,  4.0  ]
  ])"),
                    aggregated_and_grouped,
                    /*verbose=*/true);
}

TEST_P(GroupBy, MinMaxScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},

                        "[[-1, 1], [-1, 1], [-1, 2], [-1, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[null, 1], [null, 1], [null, 2], [null, 3]]"),
      ExecBatchFromJSON({int32(), int64()}, "[[2, 1], [3, 2], [4, 3]]"),
  };
  input.schema = schema({field("agg_0", int32()), field("key", int64())});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum actual,
        RunGroupBy(input, {"key"}, {{"hash_min_max", nullptr, "agg_0", "hash_min_max"}},
                   use_threads));
    Datum expected =
        ArrayFromJSON(struct_({
                          field("key", int64()),
                          field("hash_min_max",
                                struct_({field("min", int32()), field("max", int32())})),
                      }),
                      R"([
      [1, {"min": -1, "max": 2}],
      [2, {"min": -1, "max": 3}],
      [3, {"min": -1, "max": 4}]
    ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, AnyAndAll) {
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table =
        TableFromJSON(schema({field("argument", boolean()), field("key", int64())}), {R"([
    [true,  1],
    [null,  1]
                        ])",
                                                                                      R"([
    [false, 2],
    [null,  3],
    [null,  4],
    [false, 4],
    [true,  5],
    [false, null],
    [true,  1],
    [true,  2]
                        ])",
                                                                                      R"([
    [false, 2],
    [false, null],
    [null,  3]
                        ])"});

    auto no_min =
        std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
    auto min_count =
        std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
    auto keep_nulls =
        std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
    auto keep_nulls_min_count =
        std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         AltGroupBy(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")}, {},
                             {
                                 {"hash_any", no_min, "agg_0", "hash_any"},
                                 {"hash_any", min_count, "agg_1", "hash_any"},
                                 {"hash_any", keep_nulls, "agg_2", "hash_any"},
                                 {"hash_any", keep_nulls_min_count, "agg_3", "hash_any"},
                                 {"hash_all", no_min, "agg_4", "hash_all"},
                                 {"hash_all", min_count, "agg_5", "hash_all"},
                                 {"hash_all", keep_nulls, "agg_6", "hash_all"},
                                 {"hash_all", keep_nulls_min_count, "agg_7", "hash_all"},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    // Group 1: trues and nulls
    // Group 2: trues and falses
    // Group 3: nulls
    // Group 4: falses and nulls
    // Group 5: trues
    // Group null: falses
    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_any", boolean()),
                                        field("hash_any", boolean()),
                                        field("hash_any", boolean()),
                                        field("hash_any", boolean()),
                                        field("hash_all", boolean()),
                                        field("hash_all", boolean()),
                                        field("hash_all", boolean()),
                                        field("hash_all", boolean()),
                                    }),
                                    R"([
    [1,    true,  null, true,  null, true,  null,  null,  null ],
    [2,    true,  true, true,  true, false, false, false, false],
    [3,    false, null, null,  null, true,  null,  null,  null ],
    [4,    false, null, null,  null, false, null,  false, null ],
    [5,    true,  null, true,  null, true,  null,  true,  null ],
    [null, false, null, false, null, false, null,  false, null ]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, AnyAllScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({boolean(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},

                        "[[true, 1], [true, 1], [true, 2], [true, 3]]"),
      ExecBatchFromJSON({boolean(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        "[[null, 1], [null, 1], [null, 2], [null, 3]]"),
      ExecBatchFromJSON({boolean(), int64()}, "[[true, 1], [false, 2], [null, 3]]"),
  };
  input.schema = schema({field("argument", boolean()), field("key", int64())});

  auto keep_nulls =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum actual,
                         RunGroupBy(input, {"key"},
                                    {
                                        {"hash_any", nullptr, "argument", "hash_any"},
                                        {"hash_all", nullptr, "argument", "hash_all"},
                                        {"hash_any", keep_nulls, "argument", "hash_any"},
                                        {"hash_all", keep_nulls, "argument", "hash_all"},
                                    },
                                    use_threads));
    Datum expected = ArrayFromJSON(struct_({
                                       field("key", int64()),
                                       field("hash_any", boolean()),
                                       field("hash_all", boolean()),
                                       field("hash_any", boolean()),
                                       field("hash_all", boolean()),
                                   }),
                                   R"([
      [1, true, true,  true, null ],
      [2, true, false, true, false],
      [3, true, true,  true, null ]
    ])");
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, CountDistinct) {
  auto all = std::make_shared<CountOptions>(CountOptions::ALL);
  auto only_valid = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  auto only_null = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table =
        TableFromJSON(schema({field("argument", float64()), field("key", int64())}), {R"([
    [1,    1],
    [1,    1]
])",
                                                                                      R"([
    [0,    2],
    [null, 3],
    [null, 3]
])",
                                                                                      R"([
    [null, 4],
    [null, 4]
])",
                                                                                      R"([
    [4,    null],
    [1,    3]
])",
                                                                                      R"([
    [0,    2],
    [-1,   2]
])",
                                                                                      R"([
    [1,    null],
    [NaN,  3]
  ])",
                                                                                      R"([
    [2,    null],
    [3,    null]
  ])"});

    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        AltGroupBy(
            {
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
            },
            {
                table->GetColumnByName("key"),
            },
            {},
            {
                {"hash_count_distinct", all, "agg_0", "hash_count_distinct"},
                {"hash_count_distinct", only_valid, "agg_1", "hash_count_distinct"},
                {"hash_count_distinct", only_null, "agg_2", "hash_count_distinct"},
            },
            use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);
    ValidateOutput(aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                    }),
                                    R"([
    [1,    1, 1, 0],
    [2,    2, 2, 0],
    [3,    3, 2, 1],
    [4,    1, 0, 1],
    [null, 4, 4, 0]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);

    table =
        TableFromJSON(schema({field("argument", utf8()), field("key", int64())}), {R"([
    ["foo",  1],
    ["foo",  1]
])",
                                                                                   R"([
    ["bar",  2],
    [null,   3],
    [null,   3]
])",
                                                                                   R"([
    [null, 4],
    [null, 4]
])",
                                                                                   R"([
    ["baz",  null],
    ["foo",  3]
])",
                                                                                   R"([
    ["bar",  2],
    ["spam", 2]
])",
                                                                                   R"([
    ["eggs", null],
    ["ham",  3]
  ])",
                                                                                   R"([
    ["a",    null],
    ["b",    null]
  ])"});

    ASSERT_OK_AND_ASSIGN(
        aggregated_and_grouped,
        AltGroupBy(
            {
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
            },
            {
                table->GetColumnByName("key"),
            },
            {},
            {
                {"hash_count_distinct", all, "agg_0", "hash_count_distinct"},
                {"hash_count_distinct", only_valid, "agg_1", "hash_count_distinct"},
                {"hash_count_distinct", only_null, "agg_2", "hash_count_distinct"},
            },
            use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                    }),
                                    R"([
    [1,    1, 1, 0],
    [2,    2, 2, 0],
    [3,    3, 2, 1],
    [4,    1, 0, 1],
    [null, 4, 4, 0]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);

    table =
        TableFromJSON(schema({field("argument", utf8()), field("key", int64())}), {
                                                                                      R"([
    ["foo",  1],
    ["foo",  1],
    ["bar",  2],
    ["bar",  2],
    ["spam", 2]
])",
                                                                                  });

    ASSERT_OK_AND_ASSIGN(
        aggregated_and_grouped,
        AltGroupBy(
            {
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
            },
            {
                table->GetColumnByName("key"),
            },
            {},
            {
                {"hash_count_distinct", all, "agg_0", "hash_count_distinct"},
                {"hash_count_distinct", only_valid, "agg_1", "hash_count_distinct"},
                {"hash_count_distinct", only_null, "agg_2", "hash_count_distinct"},
            },
            use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                        field("hash_count_distinct", int64()),
                                    }),
                                    R"([
    [1, 1, 1, 0],
    [2, 2, 2, 0]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, Distinct) {
  auto all = std::make_shared<CountOptions>(CountOptions::ALL);
  auto only_valid = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  auto only_null = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  for (bool use_threads : {false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table =
        TableFromJSON(schema({field("argument", utf8()), field("key", int64())}), {R"([
    ["foo",  1],
    ["foo",  1]
])",
                                                                                   R"([
    ["bar",  2],
    [null,   3],
    [null,   3]
])",
                                                                                   R"([
    [null,   4],
    [null,   4]
])",
                                                                                   R"([
    ["baz",  null],
    ["foo",  3]
])",
                                                                                   R"([
    ["bar",  2],
    ["spam", 2]
])",
                                                                                   R"([
    ["eggs", null],
    ["ham",  3]
  ])",
                                                                                   R"([
    ["a",    null],
    ["b",    null]
  ])"});

    ASSERT_OK_AND_ASSIGN(auto aggregated_and_grouped,
                         AltGroupBy(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {
                                 table->GetColumnByName("key"),
                             },
                             {},
                             {
                                 {"hash_distinct", all, "agg_0", "hash_distinct"},
                                 {"hash_distinct", only_valid, "agg_1", "hash_distinct"},
                                 {"hash_distinct", only_null, "agg_2", "hash_distinct"},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    // Order of sub-arrays is not stable
    auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
      EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
      EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
      return sorted.make_array();
    };

    auto struct_arr = aggregated_and_grouped.array_as<StructArray>();

    auto all_arr = checked_pointer_cast<ListArray>(struct_arr->field(1));
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["foo"])"), sort(*all_arr->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["bar", "spam"])"),
                      sort(*all_arr->value_slice(1)), /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["foo", "ham", null])"),
                      sort(*all_arr->value_slice(2)), /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([null])"), sort(*all_arr->value_slice(3)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["a", "b", "baz", "eggs"])"),
                      sort(*all_arr->value_slice(4)), /*verbose=*/true);

    auto valid_arr = checked_pointer_cast<ListArray>(struct_arr->field(2));
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["foo"])"),
                      sort(*valid_arr->value_slice(0)), /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["bar", "spam"])"),
                      sort(*valid_arr->value_slice(1)), /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["foo", "ham"])"),
                      sort(*valid_arr->value_slice(2)), /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([])"), sort(*valid_arr->value_slice(3)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"(["a", "b", "baz", "eggs"])"),
                      sort(*valid_arr->value_slice(4)), /*verbose=*/true);

    auto null_arr = checked_pointer_cast<ListArray>(struct_arr->field(3));
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([])"), sort(*null_arr->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([])"), sort(*null_arr->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([null])"), sort(*null_arr->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([null])"), sort(*null_arr->value_slice(3)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(utf8(), R"([])"), sort(*null_arr->value_slice(4)),
                      /*verbose=*/true);

    table =
        TableFromJSON(schema({field("argument", utf8()), field("key", int64())}), {
                                                                                      R"([
    ["foo",  1],
    ["foo",  1],
    ["bar",  2],
    ["bar",  2]
])",
                                                                                  });
    ASSERT_OK_AND_ASSIGN(aggregated_and_grouped,
                         AltGroupBy(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {
                                 table->GetColumnByName("key"),
                             },
                             {},
                             {
                                 {"hash_distinct", all, "agg_0", "hash_distinct"},
                                 {"hash_distinct", only_valid, "agg_1", "hash_distinct"},
                                 {"hash_distinct", only_null, "agg_2", "hash_distinct"},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(
        ArrayFromJSON(struct_({
                          field("key_0", int64()),
                          field("hash_distinct", list(utf8())),
                          field("hash_distinct", list(utf8())),
                          field("hash_distinct", list(utf8())),
                      }),
                      R"([[1, ["foo"], ["foo"], []], [2, ["bar"], ["bar"], []]])"),
        aggregated_and_grouped,
        /*verbose=*/true);
  }
}

TEST_P(GroupBy, OneMiscTypes) {
  auto in_schema = schema({
      field("floats", float64()),
      field("nulls", null()),
      field("booleans", boolean()),
      field("decimal128", decimal128(3, 2)),
      field("decimal256", decimal256(3, 2)),
      field("fixed_binary", fixed_size_binary(3)),
      field("key", int64()),
  });
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(in_schema, {R"([
    [null, null, true,   null,    null,    null,  1],
    [1.0,  null, true,   "1.01",  "1.01",  "aaa", 1]
])",
                                           R"([
    [0.0,   null, false, "0.00",  "0.00",  "bac", 2],
    [null,  null, false, null,    null,    null,  3],
    [4.0,   null, null,  "4.01",  "4.01",  "234", null],
    [3.25,  null, true,  "3.25",  "3.25",  "ddd", 1],
    [0.125, null, false, "0.12",  "0.12",  "bcd", 2]
])",
                                           R"([
    [-0.25, null, false, "-0.25", "-0.25", "bab", 2],
    [0.75,  null, true,  "0.75",  "0.75",  "123", null],
    [null,  null, true,  null,    null,    null,  3]
])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("floats"),
                                 table->GetColumnByName("nulls"),
                                 table->GetColumnByName("booleans"),
                                 table->GetColumnByName("decimal128"),
                                 table->GetColumnByName("decimal256"),
                                 table->GetColumnByName("fixed_binary"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_one", nullptr},
                                 {"hash_one", nullptr},
                                 {"hash_one", nullptr},
                                 {"hash_one", nullptr},
                                 {"hash_one", nullptr},
                                 {"hash_one", nullptr},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
    //  Check the key column
    AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, null])"), struct_arr->field(0));

    //  Check values individually
    auto col_0_type = float64();
    const auto& col_0 = struct_arr->field(1);
    EXPECT_THAT(col_0->GetScalar(0), ResultWith(AnyOfJSON(col_0_type, R"([1.0, 3.25])")));
    EXPECT_THAT(col_0->GetScalar(1),
                ResultWith(AnyOfJSON(col_0_type, R"([0.0, 0.125, -0.25])")));
    EXPECT_THAT(col_0->GetScalar(2), ResultWith(AnyOfJSON(col_0_type, R"([null])")));
    EXPECT_THAT(col_0->GetScalar(3), ResultWith(AnyOfJSON(col_0_type, R"([4.0, 0.75])")));

    auto col_1_type = null();
    const auto& col_1 = struct_arr->field(2);
    EXPECT_THAT(col_1->GetScalar(0), ResultWith(AnyOfJSON(col_1_type, R"([null])")));
    EXPECT_THAT(col_1->GetScalar(1), ResultWith(AnyOfJSON(col_1_type, R"([null])")));
    EXPECT_THAT(col_1->GetScalar(2), ResultWith(AnyOfJSON(col_1_type, R"([null])")));
    EXPECT_THAT(col_1->GetScalar(3), ResultWith(AnyOfJSON(col_1_type, R"([null])")));

    auto col_2_type = boolean();
    const auto& col_2 = struct_arr->field(3);
    EXPECT_THAT(col_2->GetScalar(0), ResultWith(AnyOfJSON(col_2_type, R"([true])")));
    EXPECT_THAT(col_2->GetScalar(1), ResultWith(AnyOfJSON(col_2_type, R"([false])")));
    EXPECT_THAT(col_2->GetScalar(2),
                ResultWith(AnyOfJSON(col_2_type, R"([true, false])")));
    EXPECT_THAT(col_2->GetScalar(3),
                ResultWith(AnyOfJSON(col_2_type, R"([true, null])")));

    auto col_3_type = decimal128(3, 2);
    const auto& col_3 = struct_arr->field(4);
    EXPECT_THAT(col_3->GetScalar(0),
                ResultWith(AnyOfJSON(col_3_type, R"(["1.01", "3.25"])")));
    EXPECT_THAT(col_3->GetScalar(1),
                ResultWith(AnyOfJSON(col_3_type, R"(["0.00", "0.12", "-0.25"])")));
    EXPECT_THAT(col_3->GetScalar(2), ResultWith(AnyOfJSON(col_3_type, R"([null])")));
    EXPECT_THAT(col_3->GetScalar(3),
                ResultWith(AnyOfJSON(col_3_type, R"(["4.01", "0.75"])")));

    auto col_4_type = decimal256(3, 2);
    const auto& col_4 = struct_arr->field(5);
    EXPECT_THAT(col_4->GetScalar(0),
                ResultWith(AnyOfJSON(col_4_type, R"(["1.01", "3.25"])")));
    EXPECT_THAT(col_4->GetScalar(1),
                ResultWith(AnyOfJSON(col_4_type, R"(["0.00", "0.12", "-0.25"])")));
    EXPECT_THAT(col_4->GetScalar(2), ResultWith(AnyOfJSON(col_4_type, R"([null])")));
    EXPECT_THAT(col_4->GetScalar(3),
                ResultWith(AnyOfJSON(col_4_type, R"(["4.01", "0.75"])")));

    auto col_5_type = fixed_size_binary(3);
    const auto& col_5 = struct_arr->field(6);
    EXPECT_THAT(col_5->GetScalar(0),
                ResultWith(AnyOfJSON(col_5_type, R"(["aaa", "ddd"])")));
    EXPECT_THAT(col_5->GetScalar(1),
                ResultWith(AnyOfJSON(col_5_type, R"(["bab", "bcd", "bac"])")));
    EXPECT_THAT(col_5->GetScalar(2), ResultWith(AnyOfJSON(col_5_type, R"([null])")));
    EXPECT_THAT(col_5->GetScalar(3),
                ResultWith(AnyOfJSON(col_5_type, R"(["123", "234"])")));
  }
}

TEST_P(GroupBy, OneNumericTypes) {
  std::vector<std::shared_ptr<DataType>> types;
  types.insert(types.end(), NumericTypes().begin(), NumericTypes().end());
  types.insert(types.end(), TemporalTypes().begin(), TemporalTypes().end());
  types.push_back(month_interval());

  const std::vector<std::string> numeric_table_json = {R"([
      [null, 1],
      [1,    1]
    ])",
                                                       R"([
      [0,    2],
      [null, 3],
      [3,    4],
      [5,    4],
      [4,    null],
      [3,    1],
      [0,    2]
    ])",
                                                       R"([
      [0,    2],
      [1,    null],
      [null, 3]
    ])"};

  const std::vector<std::string> temporal_table_json = {R"([
      [null,      1],
      [86400000,  1]
    ])",
                                                        R"([
      [0,         2],
      [null,      3],
      [259200000, 4],
      [432000000, 4],
      [345600000, null],
      [259200000, 1],
      [0,         2]
    ])",
                                                        R"([
      [0,         2],
      [86400000,  null],
      [null,      3]
    ])"};

  for (const auto& type : types) {
    for (bool use_threads : {true, false}) {
      SCOPED_TRACE(type->ToString());
      auto in_schema = schema({field("argument0", type), field("key", int64())});
      auto table =
          TableFromJSON(in_schema, (type->name() == "date64") ? temporal_table_json
                                                              : numeric_table_json);
      ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                           GroupByTest({table->GetColumnByName("argument0")},
                                       {table->GetColumnByName("key")},
                                       {{"hash_one", nullptr}}, use_threads));
      ValidateOutput(aggregated_and_grouped);
      SortBy({"key_0"}, &aggregated_and_grouped);

      const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
      //  Check the key column
      AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, 4, null])"),
                        struct_arr->field(0));

      //  Check values individually
      const auto& col = struct_arr->field(1);
      if (type->name() == "date64") {
        EXPECT_THAT(col->GetScalar(0),
                    ResultWith(AnyOfJSON(type, R"([86400000, 259200000])")));
        EXPECT_THAT(col->GetScalar(1), ResultWith(AnyOfJSON(type, R"([0])")));
        EXPECT_THAT(col->GetScalar(2), ResultWith(AnyOfJSON(type, R"([null])")));
        EXPECT_THAT(col->GetScalar(3),
                    ResultWith(AnyOfJSON(type, R"([259200000, 432000000])")));
        EXPECT_THAT(col->GetScalar(4),
                    ResultWith(AnyOfJSON(type, R"([345600000, 86400000])")));
      } else {
        EXPECT_THAT(col->GetScalar(0), ResultWith(AnyOfJSON(type, R"([1, 3])")));
        EXPECT_THAT(col->GetScalar(1), ResultWith(AnyOfJSON(type, R"([0])")));
        EXPECT_THAT(col->GetScalar(2), ResultWith(AnyOfJSON(type, R"([null])")));
        EXPECT_THAT(col->GetScalar(3), ResultWith(AnyOfJSON(type, R"([3, 5])")));
        EXPECT_THAT(col->GetScalar(4), ResultWith(AnyOfJSON(type, R"([4, 1])")));
      }
    }
  }
}

TEST_P(GroupBy, OneBinaryTypes) {
  for (bool use_threads : {true, false}) {
    for (const auto& type : BaseBinaryTypes()) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

      auto table = TableFromJSON(schema({
                                     field("argument0", type),
                                     field("key", int64()),
                                 }),
                                 {R"([
    [null,   1],
    ["aaaa", 1]
])",
                                  R"([
    ["babcd",2],
    [null,   3],
    ["2",    null],
    ["d",    1],
    ["bc",   2]
])",
                                  R"([
    ["bcd", 2],
    ["123", null],
    [null,  3]
])"});

      ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                           GroupByTest({table->GetColumnByName("argument0")},
                                       {table->GetColumnByName("key")},
                                       {{"hash_one", nullptr}}, use_threads));
      ValidateOutput(aggregated_and_grouped);
      SortBy({"key_0"}, &aggregated_and_grouped);

      const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
      //  Check the key column
      AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, null])"),
                        struct_arr->field(0));

      const auto& col = struct_arr->field(1);
      EXPECT_THAT(col->GetScalar(0), ResultWith(AnyOfJSON(type, R"(["aaaa", "d"])")));
      EXPECT_THAT(col->GetScalar(1),
                  ResultWith(AnyOfJSON(type, R"(["bcd", "bc", "babcd"])")));
      EXPECT_THAT(col->GetScalar(2), ResultWith(AnyOfJSON(type, R"([null])")));
      EXPECT_THAT(col->GetScalar(3), ResultWith(AnyOfJSON(type, R"(["2", "123"])")));
    }
  }
}

TEST_P(GroupBy, OneScalar) {
  BatchesWithSchema input;
  input.batches = {
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},

                        R"([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])"),
      ExecBatchFromJSON({int32(), int64()}, {ArgShape::SCALAR, ArgShape::ARRAY},
                        R"([[null, 1], [null, 1], [null, 2], [null, 3]])"),
      ExecBatchFromJSON({int32(), int64()}, R"([[22, 1], [3, 2], [4, 3]])")};
  input.schema = schema({field("argument", int32()), field("key", int64())});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum actual,
        RunGroupBy(input, {"key"}, {{"hash_one", nullptr, "argument", "hash_one"}},
                   use_threads));

    const auto& struct_arr = actual.array_as<StructArray>();
    //  Check the key column
    AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3])"), struct_arr->field(0));

    const auto& col = struct_arr->field(1);
    EXPECT_THAT(col->GetScalar(0), ResultWith(AnyOfJSON(int32(), R"([-1, 22])")));
    EXPECT_THAT(col->GetScalar(1), ResultWith(AnyOfJSON(int32(), R"([3])")));
    EXPECT_THAT(col->GetScalar(2), ResultWith(AnyOfJSON(int32(), R"([4])")));
  }
}

TEST_P(GroupBy, ListNumeric) {
  for (const auto& type : NumericTypes()) {
    for (auto use_threads : {true, false}) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      {
        SCOPED_TRACE("with nulls");
        auto table =
            TableFromJSON(schema({field("argument", type), field("key", int64())}), {R"([
    [99,  1],
    [99,  1]
])",
                                                                                     R"([
    [88,  2],
    [null,   3],
    [null,   3]
])",
                                                                                     R"([
    [null,   4],
    [null,   4]
])",
                                                                                     R"([
    [77,  null],
    [99,  3]
])",
                                                                                     R"([
    [88,  2],
    [66, 2]
])",
                                                                                     R"([
    [55, null],
    [44,  3]
  ])",
                                                                                     R"([
    [33,    null],
    [22,    null]
  ])"});

        ASSERT_OK_AND_ASSIGN(auto aggregated_and_grouped,
                             AltGroupBy(
                                 {
                                     table->GetColumnByName("argument"),
                                 },
                                 {
                                     table->GetColumnByName("key"),
                                 },
                                 {},
                                 {
                                     {"hash_list", nullptr, "agg_0", "hash_list"},
                                 },
                                 use_threads));
        ValidateOutput(aggregated_and_grouped);
        SortBy({"key_0"}, &aggregated_and_grouped);

        // Order of sub-arrays is not stable
        auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
          EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
          EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
          return sorted.make_array();
        };

        auto struct_arr = aggregated_and_grouped.array_as<StructArray>();

        auto list_arr = checked_pointer_cast<ListArray>(struct_arr->field(1));
        AssertDatumsEqual(ArrayFromJSON(type, R"([99, 99])"),
                          sort(*list_arr->value_slice(0)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([66, 88, 88])"),
                          sort(*list_arr->value_slice(1)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([44, 99, null, null])"),
                          sort(*list_arr->value_slice(2)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([null, null])"),
                          sort(*list_arr->value_slice(3)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([22, 33, 55, 77])"),
                          sort(*list_arr->value_slice(4)), /*verbose=*/true);
      }
      {
        SCOPED_TRACE("without nulls");
        auto table =
            TableFromJSON(schema({field("argument", type), field("key", int64())}), {R"([
    [99,  1],
    [99,  1]
])",
                                                                                     R"([
    [88,  2],
    [100,   3],
    [100,   3]
])",
                                                                                     R"([
    [86,   4],
    [86,   4]
])",
                                                                                     R"([
    [77,  null],
    [99,  3]
])",
                                                                                     R"([
    [88,  2],
    [66, 2]
])",
                                                                                     R"([
    [55, null],
    [44,  3]
  ])",
                                                                                     R"([
    [33,    null],
    [22,    null]
  ])"});

        ASSERT_OK_AND_ASSIGN(auto aggregated_and_grouped,
                             AltGroupBy(
                                 {
                                     table->GetColumnByName("argument"),
                                 },
                                 {
                                     table->GetColumnByName("key"),
                                 },
                                 {},
                                 {
                                     {"hash_list", nullptr, "agg_0", "hash_list"},
                                 },
                                 use_threads));
        ValidateOutput(aggregated_and_grouped);
        SortBy({"key_0"}, &aggregated_and_grouped);

        // Order of sub-arrays is not stable
        auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
          EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
          EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
          return sorted.make_array();
        };

        auto struct_arr = aggregated_and_grouped.array_as<StructArray>();

        auto list_arr = checked_pointer_cast<ListArray>(struct_arr->field(1));
        AssertDatumsEqual(ArrayFromJSON(type, R"([99, 99])"),
                          sort(*list_arr->value_slice(0)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([66, 88, 88])"),
                          sort(*list_arr->value_slice(1)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([44, 99, 100, 100])"),
                          sort(*list_arr->value_slice(2)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([86, 86])"),
                          sort(*list_arr->value_slice(3)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([22, 33, 55, 77])"),
                          sort(*list_arr->value_slice(4)), /*verbose=*/true);
      }
    }
  }
}

TEST_P(GroupBy, ListBinaryTypes) {
  for (bool use_threads : {true, false}) {
    for (const auto& type : BaseBinaryTypes()) {
      SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      {
        SCOPED_TRACE("with nulls");
        auto table = TableFromJSON(schema({
                                       field("argument0", type),
                                       field("key", int64()),
                                   }),
                                   {R"([
    [null,   1],
    ["aaaa", 1]
])",
                                    R"([
    ["babcd",2],
    [null,   3],
    ["2",    null],
    ["d",    1],
    ["bc",   2]
])",
                                    R"([
    ["bcd", 2],
    ["123", null],
    [null,  3]
])"});

        ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                             AltGroupBy(
                                 {
                                     table->GetColumnByName("argument0"),
                                 },
                                 {
                                     table->GetColumnByName("key"),
                                 },
                                 {},
                                 {
                                     {"hash_list", nullptr, "agg_0", "hash_list"},
                                 },
                                 use_threads));
        ValidateOutput(aggregated_and_grouped);
        SortBy({"key_0"}, &aggregated_and_grouped);

        // Order of sub-arrays is not stable
        auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
          EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
          EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
          return sorted.make_array();
        };

        const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
        // Check the key column
        AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, null])"),
                          struct_arr->field(0));

        auto list_arr = checked_pointer_cast<ListArray>(struct_arr->field(1));
        AssertDatumsEqual(ArrayFromJSON(type, R"(["aaaa", "d", null])"),
                          sort(*list_arr->value_slice(0)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"(["babcd", "bc", "bcd"])"),
                          sort(*list_arr->value_slice(1)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"([null, null])"),
                          sort(*list_arr->value_slice(2)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"(["123", "2"])"),
                          sort(*list_arr->value_slice(3)),
                          /*verbose=*/true);
      }
      {
        SCOPED_TRACE("without nulls");
        auto table = TableFromJSON(schema({
                                       field("argument0", type),
                                       field("key", int64()),
                                   }),
                                   {R"([
    ["y",   1],
    ["aaaa", 1]
])",
                                    R"([
    ["babcd",2],
    ["z",   3],
    ["2",    null],
    ["d",    1],
    ["bc",   2]
])",
                                    R"([
    ["bcd", 2],
    ["123", null],
    ["z",  3]
])"});

        ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                             AltGroupBy(
                                 {
                                     table->GetColumnByName("argument0"),
                                 },
                                 {
                                     table->GetColumnByName("key"),
                                 },
                                 {},
                                 {
                                     {"hash_list", nullptr, "agg_0", "hash_list"},
                                 },
                                 use_threads));
        ValidateOutput(aggregated_and_grouped);
        SortBy({"key_0"}, &aggregated_and_grouped);

        // Order of sub-arrays is not stable
        auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
          EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
          EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
          return sorted.make_array();
        };

        const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
        // Check the key column
        AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, null])"),
                          struct_arr->field(0));

        auto list_arr = checked_pointer_cast<ListArray>(struct_arr->field(1));
        AssertDatumsEqual(ArrayFromJSON(type, R"(["aaaa", "d", "y"])"),
                          sort(*list_arr->value_slice(0)),
                          /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"(["babcd", "bc", "bcd"])"),
                          sort(*list_arr->value_slice(1)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"(["z", "z"])"),
                          sort(*list_arr->value_slice(2)), /*verbose=*/true);
        AssertDatumsEqual(ArrayFromJSON(type, R"(["123", "2"])"),
                          sort(*list_arr->value_slice(3)),
                          /*verbose=*/true);
      }
    }
  }
}

TEST_P(GroupBy, ListMiscTypes) {
  auto in_schema = schema({
      field("floats", float64()),
      field("nulls", null()),
      field("booleans", boolean()),
      field("decimal128", decimal128(3, 2)),
      field("decimal256", decimal256(3, 2)),
      field("fixed_binary", fixed_size_binary(3)),
      field("key", int64()),
  });
  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    auto table = TableFromJSON(in_schema, {R"([
        [null, null, true,   null,    null,    null,  1],
        [1.0,  null, true,   "1.01",  "1.01",  "aaa", 1]
        ])",
                                           R"([
        [0.0,   null, false, "0.00",  "0.00",  "bac", 2],
        [null,  null, false, null,    null,    null,  3],
        [4.0,   null, null,  "4.01",  "4.01",  "234", null],
        [3.25,  null, true,  "3.25",  "3.25",  "ddd", 1],
        [0.125, null, false, "0.12",  "0.12",  "bcd", 2]
        ])",
                                           R"([
        [-0.25, null, false, "-0.25", "-0.25", "bab", 2],
        [0.75,  null, true,  "0.75",  "0.75",  "123", null],
        [null,  null, true,  null,    null,    null,  3]
        ])"});

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("floats"),
                                 table->GetColumnByName("nulls"),
                                 table->GetColumnByName("booleans"),
                                 table->GetColumnByName("decimal128"),
                                 table->GetColumnByName("decimal256"),
                                 table->GetColumnByName("fixed_binary"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_list", nullptr},
                                 {"hash_list", nullptr},
                                 {"hash_list", nullptr},
                                 {"hash_list", nullptr},
                                 {"hash_list", nullptr},
                                 {"hash_list", nullptr},
                             },
                             use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    // Order of sub-arrays is not stable
    auto sort = [](const Array& arr) -> std::shared_ptr<Array> {
      EXPECT_OK_AND_ASSIGN(auto indices, SortIndices(arr));
      EXPECT_OK_AND_ASSIGN(auto sorted, Take(arr, indices));
      return sorted.make_array();
    };

    const auto& struct_arr = aggregated_and_grouped.array_as<StructArray>();
    //  Check the key column
    AssertDatumsEqual(ArrayFromJSON(int64(), R"([1, 2, 3, null])"), struct_arr->field(0));

    //  Check values individually
    auto type_0 = float64();
    auto list_arr_0 = checked_pointer_cast<ListArray>(struct_arr->field(1));
    AssertDatumsEqual(ArrayFromJSON(type_0, R"([1.0, 3.25, null])"),
                      sort(*list_arr_0->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_0, R"([-0.25, 0.0, 0.125])"),
                      sort(*list_arr_0->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_0, R"([null, null])"),
                      sort(*list_arr_0->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_0, R"([0.75, 4.0])"),
                      sort(*list_arr_0->value_slice(3)),
                      /*verbose=*/true);

    auto type_1 = null();
    auto list_arr_1 = checked_pointer_cast<ListArray>(struct_arr->field(2));
    AssertDatumsEqual(ArrayFromJSON(type_1, R"([null, null, null])"),
                      sort(*list_arr_1->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_1, R"([null, null, null])"),
                      sort(*list_arr_1->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_1, R"([null, null])"),
                      sort(*list_arr_1->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_1, R"([null, null])"),
                      sort(*list_arr_1->value_slice(3)),
                      /*verbose=*/true);

    auto type_2 = boolean();
    auto list_arr_2 = checked_pointer_cast<ListArray>(struct_arr->field(3));
    AssertDatumsEqual(ArrayFromJSON(type_2, R"([true, true, true])"),
                      sort(*list_arr_2->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_2, R"([false, false, false])"),
                      sort(*list_arr_2->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_2, R"([false, true])"),
                      sort(*list_arr_2->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_2, R"([true, null])"),
                      sort(*list_arr_2->value_slice(3)),
                      /*verbose=*/true);

    auto type_3 = decimal128(3, 2);
    auto list_arr_3 = checked_pointer_cast<ListArray>(struct_arr->field(4));
    AssertDatumsEqual(ArrayFromJSON(type_3, R"(["1.01", "3.25", null])"),
                      sort(*list_arr_3->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_3, R"(["-0.25", "0.00", "0.12"])"),
                      sort(*list_arr_3->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_3, R"([null, null])"),
                      sort(*list_arr_3->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_3, R"(["0.75", "4.01"])"),
                      sort(*list_arr_3->value_slice(3)),
                      /*verbose=*/true);

    auto type_4 = decimal256(3, 2);
    auto list_arr_4 = checked_pointer_cast<ListArray>(struct_arr->field(5));
    AssertDatumsEqual(ArrayFromJSON(type_4, R"(["1.01", "3.25", null])"),
                      sort(*list_arr_4->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_4, R"(["-0.25", "0.00", "0.12"])"),
                      sort(*list_arr_4->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_4, R"([null, null])"),
                      sort(*list_arr_4->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_4, R"(["0.75", "4.01"])"),
                      sort(*list_arr_4->value_slice(3)),
                      /*verbose=*/true);

    auto type_5 = fixed_size_binary(3);
    auto list_arr_5 = checked_pointer_cast<ListArray>(struct_arr->field(6));
    AssertDatumsEqual(ArrayFromJSON(type_5, R"(["aaa", "ddd", null])"),
                      sort(*list_arr_5->value_slice(0)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_5, R"(["bab", "bac", "bcd"])"),
                      sort(*list_arr_5->value_slice(1)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_5, R"([null, null])"),
                      sort(*list_arr_5->value_slice(2)),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(type_5, R"(["123", "234"])"),
                      sort(*list_arr_5->value_slice(3)),
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, CountAndSum) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [1.0,   1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
  ])");

  std::shared_ptr<CountOptions> count_opts;
  auto count_nulls_opts = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  auto count_all_opts = std::make_shared<CountOptions>(CountOptions::ALL);
  auto min_count_opts =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  ASSERT_OK_AND_ASSIGN(
      Datum aggregated_and_grouped,
      AltGroupBy(
          {
              // NB: passing an argument twice or also using it as a key is legal
              batch->GetColumnByName("argument"),
              batch->GetColumnByName("argument"),
              batch->GetColumnByName("argument"),
              batch->GetColumnByName("argument"),
              batch->GetColumnByName("argument"),
              batch->GetColumnByName("key"),
          },
          {
              batch->GetColumnByName("key"),
          },
          {},
          {
              {"hash_count", count_opts, "agg_0", "hash_count"},
              {"hash_count", count_nulls_opts, "agg_1", "hash_count"},
              {"hash_count", count_all_opts, "agg_2", "hash_count"},
              {"hash_count_all", "hash_count_all"},
              {"hash_sum", "agg_3", "hash_sum"},
              {"hash_sum", min_count_opts, "agg_4", "hash_sum"},
              {"hash_sum", "agg_5", "hash_sum"},
          }));

  AssertDatumsEqual(
      ArrayFromJSON(struct_({
                        field("key_0", int64()),
                        field("hash_count", int64()),
                        field("hash_count", int64()),
                        field("hash_count", int64()),
                        field("hash_count_all", int64()),
                        // NB: summing a float32 array results in float64 sums
                        field("hash_sum", float64()),
                        field("hash_sum", float64()),
                        field("hash_sum", int64()),
                    }),
                    R"([
    [1,    2, 1, 3, 3, 4.25,   null,   3   ],
    [2,    3, 0, 3, 3, -0.125, -0.125, 6   ],
    [3,    0, 2, 2, 2, null,   null,   6   ],
    [null, 2, 0, 2, 2, 4.75,   null,   null]
  ])"),
      aggregated_and_grouped,
      /*verbose=*/true);
}

TEST_P(GroupBy, StandAloneNullaryCount) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [1.0,   1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
  ])");

  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy(
                           // zero arguments for aggregations because only the
                           // nullary hash_count_all aggregation is present
                           {},
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_count_all", "hash_count_all"},
                           }));

  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_count_all", int64()),
                                  }),
                                  R"([
    [1, 3   ],
    [2, 3   ],
    [3, 2   ],
    [null, 2]
  ])"),
                    aggregated_and_grouped,
                    /*verbose=*/true);
}

TEST_P(GroupBy, Product) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [-1.0,  1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
  ])");

  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy(
                           {
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("key"),
                               batch->GetColumnByName("argument"),
                           },
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_product", nullptr, "agg_0", "hash_product"},
                               {"hash_product", nullptr, "agg_1", "hash_product"},
                               {"hash_product", min_count, "agg_2", "hash_product"},
                           }));

  AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                            field("key_0", int64()),
                                            field("hash_product", float64()),
                                            field("hash_product", int64()),
                                            field("hash_product", float64()),
                                        }),
                                        R"([
    [1,    -3.25, 1,    null],
    [2,    -0.0,  8,    -0.0],
    [3,    null,  9,    null],
    [null, 3.0,   null, null]
  ])"),
                          aggregated_and_grouped,
                          /*verbose=*/true);

  // Overflow should wrap around
  batch = RecordBatchFromJSON(schema({field("argument", int64()), field("key", int64())}),
                              R"([
    [8589934592, 1],
    [8589934593, 1]
  ])");

  ASSERT_OK_AND_ASSIGN(aggregated_and_grouped,
                       AltGroupBy(
                           {
                               batch->GetColumnByName("argument"),
                           },
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_product", nullptr, "agg_0", "hash_product"},
                           }));

  AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                            field("key_0", int64()),
                                            field("hash_product", int64()),
                                        }),
                                        R"([[1, 8589934592]])"),
                          aggregated_and_grouped,
                          /*verbose=*/true);
}

TEST_P(GroupBy, SumMeanProductKeepNulls) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [-1.0,  1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
  ])");

  auto keep_nulls = std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false);
  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy(
                           {
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                               batch->GetColumnByName("argument"),
                           },
                           {
                               batch->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_sum", keep_nulls, "agg_0", "hash_sum"},
                               {"hash_sum", min_count, "agg_1", "hash_sum"},
                               {"hash_mean", keep_nulls, "agg_2", "hash_mean"},
                               {"hash_mean", min_count, "agg_3", "hash_mean"},
                               {"hash_product", keep_nulls, "agg_4", "hash_product"},
                               {"hash_product", min_count, "agg_5", "hash_product"},
                           }));

  AssertDatumsApproxEqual(ArrayFromJSON(struct_({
                                            field("key_0", int64()),
                                            field("hash_sum", float64()),
                                            field("hash_sum", float64()),
                                            field("hash_mean", float64()),
                                            field("hash_mean", float64()),
                                            field("hash_product", float64()),
                                            field("hash_product", float64()),
                                        }),
                                        R"([
    [1,    null,   null,   null,       null,       null, null],
    [2,    -0.125, -0.125, -0.0416667, -0.0416667, -0.0, -0.0],
    [3,    null,   null,   null,       null,       null, null],
    [null, 4.75,   null,   2.375,      null,       3.0,  null]
  ])"),
                          aggregated_and_grouped,
                          /*verbose=*/true);
}

TEST_P(GroupBy, SumOnlyStringAndDictKeys) {
  for (auto key_type : {utf8(), dictionary(int32(), utf8())}) {
    SCOPED_TRACE("key type: " + key_type->ToString());

    auto batch = RecordBatchFromJSON(
        schema({field("agg_0", float64()), field("key", key_type)}), R"([
      [1.0,   "alfa"],
      [null,  "alfa"],
      [0.0,   "beta"],
      [null,  "gama"],
      [4.0,    null ],
      [3.25,  "alfa"],
      [0.125, "beta"],
      [-0.25, "beta"],
      [0.75,   null ],
      [null,  "gama"]
    ])");

    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        AltGroupBy({batch->GetColumnByName("agg_0")}, {batch->GetColumnByName("key")}, {},
                   {
                       {"hash_sum", nullptr, "agg_0", "hash_sum"},
                   }));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", key_type),
                                        field("hash_sum", float64()),
                                    }),
                                    R"([
    ["alfa", 4.25  ],
    ["beta", -0.125],
    ["gama", null  ],
    [null,   4.75  ]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, ConcreteCaseWithValidateGroupBy) {
  auto batch =
      RecordBatchFromJSON(schema({field("agg_0", float64()), field("key", utf8())}), R"([
    [1.0,   "alfa"],
    [null,  "alfa"],
    [0.0,   "beta"],
    [null,  "gama"],
    [4.0,    null ],
    [3.25,  "alfa"],
    [0.125, "beta"],
    [-0.25, "beta"],
    [0.75,   null ],
    [null,  "gama"]
  ])");

  std::shared_ptr<ScalarAggregateOptions> keepna =
      std::make_shared<ScalarAggregateOptions>(false, 1);
  std::shared_ptr<CountOptions> nulls =
      std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  std::shared_ptr<CountOptions> non_null =
      std::make_shared<CountOptions>(CountOptions::ONLY_VALID);

  for (auto agg : {
           Aggregate{"hash_sum", nullptr, "agg_0", "hash_sum"},
           Aggregate{"hash_count", non_null, "agg_0", "hash_count"},
           Aggregate{"hash_count", nulls, "agg_0", "hash_count"},
           Aggregate{"hash_min_max", nullptr, "agg_0", "hash_min_max"},
           Aggregate{"hash_min_max", keepna, "agg_0", "hash_min_max"},
       }) {
    SCOPED_TRACE(agg.function);
    ValidateGroupBy({agg}, {batch->GetColumnByName("agg_0")},
                    {batch->GetColumnByName("key")});
  }
}

// Count nulls/non_nulls from record batch with no nulls
TEST_P(GroupBy, CountNull) {
  auto batch =
      RecordBatchFromJSON(schema({field("agg_0", float64()), field("key", utf8())}), R"([
    [1.0, "alfa"],
    [2.0, "beta"],
    [3.0, "gama"]
  ])");

  std::shared_ptr<CountOptions> keepna =
      std::make_shared<CountOptions>(CountOptions::ONLY_NULL);
  std::shared_ptr<CountOptions> skipna =
      std::make_shared<CountOptions>(CountOptions::ONLY_VALID);

  for (auto agg : {
           Aggregate{"hash_count", keepna, "agg_0", "hash_count"},
           Aggregate{"hash_count", skipna, "agg_0", "hash_count"},
       }) {
    SCOPED_TRACE(agg.function);
    ValidateGroupBy({agg}, {batch->GetColumnByName("agg_0")},
                    {batch->GetColumnByName("key")});
  }
}

TEST_P(GroupBy, RandomArraySum) {
  std::shared_ptr<ScalarAggregateOptions> options =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
  for (int64_t length : {1 << 10, 1 << 12, 1 << 15}) {
    for (auto null_probability : {0.0, 0.01, 0.5, 1.0}) {
      auto batch = random::GenerateBatch(
          {
              field(
                  "agg_0", float32(),
                  key_value_metadata({{"null_probability", ToChars(null_probability)}})),
              field("key", int64(), key_value_metadata({{"min", "0"}, {"max", "100"}})),
          },
          length, 0xDEADBEEF);

      ValidateGroupBy(
          {
              {"hash_sum", options, "agg_0", "hash_sum"},
          },
          {batch->GetColumnByName("agg_0")}, {batch->GetColumnByName("key")},
          /*naive=*/false);
    }
  }
}

TEST_P(GroupBy, WithChunkedArray) {
  auto table =
      TableFromJSON(schema({field("argument", float64()), field("key", int64())}),
                    {R"([{"argument": 1.0,   "key": 1},
                         {"argument": null,  "key": 1}
                        ])",
                     R"([{"argument": 0.0,   "key": 2},
                         {"argument": null,  "key": 3},
                         {"argument": 4.0,   "key": null},
                         {"argument": 3.25,  "key": 1},
                         {"argument": 0.125, "key": 2},
                         {"argument": -0.25, "key": 2},
                         {"argument": 0.75,  "key": null},
                         {"argument": null,  "key": 3}
                        ])"});
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy(
                           {
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                           },
                           {
                               table->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_count", nullptr, "agg_0", "hash_count"},
                               {"hash_sum", nullptr, "agg_1", "hash_sum"},
                               {"hash_min_max", nullptr, "agg_2", "hash_min_max"},
                           }));

  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_count", int64()),
                                      field("hash_sum", float64()),
                                      field("hash_min_max", struct_({
                                                                field("min", float64()),
                                                                field("max", float64()),
                                                            })),
                                  }),
                                  R"([
    [1,    2, 4.25,   {"min": 1.0,   "max": 3.25} ],
    [2,    3, -0.125, {"min": -0.25, "max": 0.125}],
    [3,    0, null,   {"min": null,  "max": null} ],
    [null, 2, 4.75,   {"min": 0.75,  "max": 4.0}  ]
  ])"),
                    aggregated_and_grouped,
                    /*verbose=*/true);
}

TEST_P(GroupBy, MinMaxWithNewGroupsInChunkedArray) {
  auto table = TableFromJSON(
      schema({field("argument", int64()), field("key", int64())}),
      {R"([{"argument": 1, "key": 0}])", R"([{"argument": 0,   "key": 1}])"});
  ScalarAggregateOptions count_options;
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy(
                           {
                               table->GetColumnByName("argument"),
                           },
                           {
                               table->GetColumnByName("key"),
                           },
                           {},
                           {
                               {"hash_min_max", nullptr, "agg_0", "hash_min_max"},
                           }));

  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_min_max", struct_({
                                                                field("min", int64()),
                                                                field("max", int64()),
                                                            })),
                                  }),
                                  R"([
    [0, {"min": 1, "max": 1}],
    [1, {"min": 0, "max": 0}]
  ])"),
                    aggregated_and_grouped,
                    /*verbose=*/true);
}

TEST_P(GroupBy, FirstLastBasicTypes) {
  std::vector<std::shared_ptr<DataType>> types;
  types.insert(types.end(), boolean());
  types.insert(types.end(), NumericTypes().begin(), NumericTypes().end());
  types.insert(types.end(), TemporalTypes().begin(), TemporalTypes().end());

  const std::vector<std::string> numeric_table = {R"([
    [1,    1],
    [null, 5],
    [null, 1],
    [null, 7]
])",
                                                  R"([
    [0,    2],
    [null, 3],
    [3,    4],
    [5,    4],
    [4,    null],
    [3,    1],
    [6,    6],
    [5,    5],
    [0,    2],
    [7,    7]
])",
                                                  R"([
    [0,    2],
    [1,    null],
    [6,    5],
    [null, 5],
    [null, 6],
    [null, 3]
])"};

  const std::string numeric_expected =
      R"([
    [1,    1,    3,    1,   3],
    [2,    0,    0,    0,   0],
    [3,    null,  null,  null,  null],
    [4,    3,     5,    3,   5],
    [5,    5,     6,    null,   null],
    [6,    6,     6,    6,      null],
    [7,    7,     7,    null,   7],
    [null, 4,     1,    4,   1]
    ])";

  const std::vector<std::string> date64_table = {R"([
    [86400000,    1],
    [null, 1]
])",
                                                 R"([
    [0,    2],
    [null, 3],
    [259200000,    4],
    [432000000,    4],
    [345600000,    null],
    [259200000,    1],
    [0,    2]
])",
                                                 R"([
    [0,    2],
    [86400000,    null],
    [null, 3]
])"};

  const std::string date64_expected =
      R"([
    [1,    86400000,259200000,86400000,259200000],
    [2,    0,0,0,0],
    [3,    null,null,null,null],
    [4,    259200000,432000000,259200000,432000000],
    [null, 345600000,86400000,345600000,86400000]
    ])";

  const std::vector<std::string> boolean_table = {R"([
    [true,    1],
    [null, 1]
])",
                                                  R"([
    [false,    2],
    [null, 3],
    [false,    4],
    [true,    4],
    [true,    null],
    [false,    1],
    [false,    2]
])",
                                                  R"([
    [false,    2],
    [false,    null],
    [null, 3]
])"};

  const std::string boolean_expected =
      R"([
    [1,    true,false,true,false],
    [2,    false,false,false,false],
    [3,    null,null,null,null],
    [4,    false,true,false,true],
    [null, true,false,true,false]
    ])";

  auto keep_nulls = std::make_shared<ScalarAggregateOptions>(false, 1);

  for (const auto& ty : types) {
    SCOPED_TRACE(ty->ToString());
    auto in_schema = schema({field("argument0", ty), field("key", int64())});
    auto table = TableFromJSON(in_schema, (ty->name() == "date64") ? date64_table
                                          : (ty->name() == "bool") ? boolean_table
                                                                   : numeric_table);

    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument0"),
                                 table->GetColumnByName("argument0"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_first", nullptr},
                                 {"hash_last", nullptr},
                                 {"hash_first", keep_nulls},
                                 {"hash_last", keep_nulls},
                             },
                             /*use_threads=*/false));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_first", ty),
                                        field("hash_last", ty),
                                        field("hash_first", ty),
                                        field("hash_last", ty),
                                    }),
                                    (ty->name() == "date64") ? date64_expected
                                    : (ty->name() == "bool") ? boolean_expected
                                                             : numeric_expected),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, FirstLastBinary) {
  // First / last doesn't support multi threaded execution
  bool use_threads = false;
  for (const auto& ty : BaseBinaryTypes()) {
    auto table = TableFromJSON(schema({
                                   field("argument0", ty),
                                   field("key", int64()),
                               }),
                               {R"([
    ["aaaa", 1],
    [null,   5],
    [null,   1]
])",
                                R"([
    ["bcd",  2],
    [null,   3],
    ["2",    null],
    ["d",    1],
    ["ee",   5],
    ["bc",   2]
])",
                                R"([
    ["babcd", 2],
    ["123",   null],
    [null,    5],
    [null,    3]
])"});

    auto keep_nulls = std::make_shared<ScalarAggregateOptions>(false, 1);
    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        GroupByTest(
            {table->GetColumnByName("argument0"), table->GetColumnByName("argument0"),
             table->GetColumnByName("argument0"), table->GetColumnByName("argument0")},
            {table->GetColumnByName("key")},
            {{"hash_first", nullptr},
             {"hash_last", nullptr},
             {"hash_first", keep_nulls},
             {"hash_last", keep_nulls}},
            use_threads));
    ValidateOutput(aggregated_and_grouped);
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(
        ArrayFromJSON(struct_({field("key_0", int64()), field("hash_first", ty),
                               field("hash_last", ty), field("hash_first", ty),
                               field("hash_last", ty)}),
                      R"([
      [1,    "aaaa",    "d", "aaaa", "d"],
      [2,    "bcd",    "babcd", "bcd", "babcd"],
      [3,    null,    null, null, null],
      [5,    "ee",    "ee", null, null],
      [null, "2",    "123", "2", "123"]
    ])"),
        aggregated_and_grouped,
        /*verbose=*/true);
  }
}

TEST_P(GroupBy, FirstLastFixedSizeBinary) {
  const auto ty = fixed_size_binary(3);
  bool use_threads = false;

  auto table = TableFromJSON(schema({
                                 field("argument0", ty),
                                 field("key", int64()),
                             }),
                             {R"([
    ["aaa", 1],
    [null,  1]
])",
                              R"([
    ["bac", 2],
    [null,  3],
    ["234", null],
    ["ddd", 1],
    ["bcd", 2]
])",
                              R"([
    ["bab", 2],
    ["123", null],
    [null,  3]
])"});

  ASSERT_OK_AND_ASSIGN(
      Datum aggregated_and_grouped,
      GroupByTest(
          {table->GetColumnByName("argument0"), table->GetColumnByName("argument0")},
          {table->GetColumnByName("key")},
          {{"hash_first", nullptr}, {"hash_last", nullptr}}, use_threads));
  ValidateOutput(aggregated_and_grouped);
  SortBy({"key_0"}, &aggregated_and_grouped);

  AssertDatumsEqual(
      ArrayFromJSON(struct_({field("key_0", int64()), field("hash_first", ty),
                             field("hash_last", ty)}),
                    R"([
    [1,    "aaa", "ddd"],
    [2,    "bac", "bab"],
    [3,    null,  null],
    [null, "234", "123"]
  ])"),
      aggregated_and_grouped,
      /*verbose=*/true);
}

TEST_P(GroupBy, SmallChunkSizeSumOnly) {
  auto batch = RecordBatchFromJSON(
      schema({field("argument", float64()), field("key", int64())}), R"([
    [1.0,   1],
    [null,  1],
    [0.0,   2],
    [null,  3],
    [4.0,   null],
    [3.25,  1],
    [0.125, 2],
    [-0.25, 2],
    [0.75,  null],
    [null,  3]
  ])");
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       AltGroupBy({batch->GetColumnByName("argument")},
                                  {batch->GetColumnByName("key")}, {},
                                  {
                                      {"hash_sum", nullptr, "agg_0", "hash_sum"},
                                  },
                                  small_chunksize_context()));
  AssertDatumsEqual(ArrayFromJSON(struct_({
                                      field("key_0", int64()),
                                      field("hash_sum", float64()),
                                  }),
                                  R"([
    [1,    4.25  ],
    [2,    -0.125],
    [3,    null  ],
    [null, 4.75  ]
  ])"),
                    aggregated_and_grouped,
                    /*verbose=*/true);
}

TEST_P(GroupBy, CountWithNullType) {
  auto table =
      TableFromJSON(schema({field("argument", null()), field("key", int64())}), {R"([
    [null,  1],
    [null,  1]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, 3],
    [null, null],
    [null, 1],
    [null, 2]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, null],
    [null, 3]
                        ])"});

  auto all = std::make_shared<CountOptions>(CountOptions::ALL);
  auto only_valid = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  auto only_null = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_count", all},
                                 {"hash_count", only_valid},
                                 {"hash_count", only_null},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_count", int64()),
                                        field("hash_count", int64()),
                                        field("hash_count", int64()),
                                    }),
                                    R"([
    [1,    3, 0, 3],
    [2,    3, 0, 3],
    [3,    2, 0, 2],
    [null, 2, 0, 2]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, CountWithNullTypeEmptyTable) {
  auto table = TableFromJSON(schema({field("argument", null()), field("key", int64())}),
                             {R"([])"});

  auto all = std::make_shared<CountOptions>(CountOptions::ALL);
  auto only_valid = std::make_shared<CountOptions>(CountOptions::ONLY_VALID);
  auto only_null = std::make_shared<CountOptions>(CountOptions::ONLY_NULL);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_count", all},
                                 {"hash_count", only_valid},
                                 {"hash_count", only_null},
                             },
                             use_threads));
    auto struct_arr = aggregated_and_grouped.array_as<StructArray>();
    for (auto& field : struct_arr->fields()) {
      AssertDatumsEqual(ArrayFromJSON(int64(), "[]"), field, /*verbose=*/true);
    }
  }
}

TEST_P(GroupBy, SingleNullTypeKey) {
  auto table =
      TableFromJSON(schema({field("argument", int64()), field("key", null())}), {R"([
    [1,    null],
    [1,    null]
                        ])",
                                                                                 R"([
    [2,    null],
    [3,    null],
    [null, null],
    [1,    null],
    [2,    null]
                        ])",
                                                                                 R"([
    [2,    null],
    [null, null],
    [3,    null]
                        ])"});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_count", nullptr},
                                 {"hash_sum", nullptr},
                                 {"hash_mean", nullptr},
                                 {"hash_min_max", nullptr},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", null()),
                                        field("hash_count", int64()),
                                        field("hash_sum", int64()),
                                        field("hash_mean", float64()),
                                        field("hash_min_max", struct_({
                                                                  field("min", int64()),
                                                                  field("max", int64()),
                                                              })),
                                    }),
                                    R"([
    [null, 8, 15, 1.875, {"min": 1, "max": 3}]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, MultipleKeysIncludesNullType) {
  auto table = TableFromJSON(schema({field("argument", float64()), field("key_0", utf8()),
                                     field("key_1", null())}),
                             {R"([
    [1.0,   "a",      null],
    [null,  "a",      null]
                        ])",
                              R"([
    [0.0,   "bcdefg", null],
    [null,  "aa",     null],
    [4.0,   null,     null],
    [3.25,  "a",      null],
    [0.125, "bcdefg", null]
                        ])",
                              R"([
    [-0.25, "bcdefg", null],
    [0.75,  null,     null],
    [null,  "aa",     null]
                        ])"});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        GroupByTest(
            {
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
                table->GetColumnByName("argument"),
            },
            {table->GetColumnByName("key_0"), table->GetColumnByName("key_1")},
            {
                {"hash_count", nullptr},
                {"hash_sum", nullptr},
                {"hash_min_max", nullptr},
            },
            use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", utf8()),
                                        field("key_1", null()),
                                        field("hash_count", int64()),
                                        field("hash_sum", float64()),
                                        field("hash_min_max", struct_({
                                                                  field("min", float64()),
                                                                  field("max", float64()),
                                                              })),
                                    }),
                                    R"([
    ["a",      null, 2, 4.25,   {"min": 1,     "max": 3.25} ],
    ["aa",     null, 0, null,   {"min": null,  "max": null} ],
    ["bcdefg", null, 3, -0.125, {"min": -0.25, "max": 0.125}],
    [null,     null, 2, 4.75,   {"min": 0.75,  "max": 4}    ]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, SumNullType) {
  auto table =
      TableFromJSON(schema({field("argument", null()), field("key", int64())}), {R"([
    [null,  1],
    [null,  1]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, 3],
    [null, null],
    [null, 1],
    [null, 2]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, null],
    [null, 3]
                        ])"});

  auto no_min =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  auto keep_nulls =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
  auto keep_nulls_min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_sum", no_min},
                                 {"hash_sum", keep_nulls},
                                 {"hash_sum", min_count},
                                 {"hash_sum", keep_nulls_min_count},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_sum", int64()),
                                        field("hash_sum", int64()),
                                        field("hash_sum", int64()),
                                        field("hash_sum", int64()),
                                    }),
                                    R"([
    [1,    0, null, null, null],
    [2,    0, null, null, null],
    [3,    0, null, null, null],
    [null, 0, null, null, null]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, ProductNullType) {
  auto table =
      TableFromJSON(schema({field("argument", null()), field("key", int64())}), {R"([
    [null,  1],
    [null,  1]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, 3],
    [null, null],
    [null, 1],
    [null, 2]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, null],
    [null, 3]
                        ])"});

  auto no_min =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  auto keep_nulls =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
  auto keep_nulls_min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_product", no_min},
                                 {"hash_product", keep_nulls},
                                 {"hash_product", min_count},
                                 {"hash_product", keep_nulls_min_count},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_product", int64()),
                                        field("hash_product", int64()),
                                        field("hash_product", int64()),
                                        field("hash_product", int64()),
                                    }),
                                    R"([
    [1,    1, null, null, null],
    [2,    1, null, null, null],
    [3,    1, null, null, null],
    [null, 1, null, null, null]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, MeanNullType) {
  auto table =
      TableFromJSON(schema({field("argument", null()), field("key", int64())}), {R"([
    [null,  1],
    [null,  1]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, 3],
    [null, null],
    [null, 1],
    [null, 2]
                        ])",
                                                                                 R"([
    [null, 2],
    [null, null],
    [null, 3]
                        ])"});

  auto no_min =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  auto keep_nulls =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
  auto keep_nulls_min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_mean", no_min},
                                 {"hash_mean", keep_nulls},
                                 {"hash_mean", min_count},
                                 {"hash_mean", keep_nulls_min_count},
                             },
                             use_threads));
    SortBy({"key_0"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("hash_mean", float64()),
                                        field("hash_mean", float64()),
                                        field("hash_mean", float64()),
                                        field("hash_mean", float64()),
                                    }),
                                    R"([
    [1,    0, null, null, null],
    [2,    0, null, null, null],
    [3,    0, null, null, null],
    [null, 0, null, null, null]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, NullTypeEmptyTable) {
  auto table = TableFromJSON(schema({field("argument", null()), field("key", int64())}),
                             {R"([])"});

  auto no_min =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/0);
  auto min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/true, /*min_count=*/3);
  auto keep_nulls =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/0);
  auto keep_nulls_min_count =
      std::make_shared<ScalarAggregateOptions>(/*skip_nulls=*/false, /*min_count=*/3);

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                         GroupByTest(
                             {
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                                 table->GetColumnByName("argument"),
                             },
                             {table->GetColumnByName("key")},
                             {
                                 {"hash_sum", no_min},
                                 {"hash_product", min_count},
                                 {"hash_mean", keep_nulls},
                             },
                             use_threads));
    auto struct_arr = aggregated_and_grouped.array_as<StructArray>();
    AssertDatumsEqual(ArrayFromJSON(int64(), "[]"), struct_arr->field(1),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(int64(), "[]"), struct_arr->field(2),
                      /*verbose=*/true);
    AssertDatumsEqual(ArrayFromJSON(float64(), "[]"), struct_arr->field(3),
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, OnlyKeys) {
  auto table =
      TableFromJSON(schema({field("key_0", int64()), field("key_1", utf8())}), {R"([
    [1,    "a"],
    [null, "a"]
                        ])",
                                                                                R"([
    [0,    "bcdefg"],
    [null, "aa"],
    [3,    null],
    [1,    "a"],
    [2,    "bcdefg"]
                        ])",
                                                                                R"([
    [0,    "bcdefg"],
    [1,    null],
    [null, "a"]
                        ])"});

  for (bool use_threads : {true, false}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(
        Datum aggregated_and_grouped,
        GroupByTest({},
                    {table->GetColumnByName("key_0"), table->GetColumnByName("key_1")},
                    {}, use_threads));
    SortBy({"key_0", "key_1"}, &aggregated_and_grouped);

    AssertDatumsEqual(ArrayFromJSON(struct_({
                                        field("key_0", int64()),
                                        field("key_1", utf8()),
                                    }),
                                    R"([
    [0,    "bcdefg"],
    [1,    "a"],
    [1,    null],
    [2,    "bcdefg"],
    [3,    null],
    [null, "a"],
    [null, "aa"]
  ])"),
                      aggregated_and_grouped,
                      /*verbose=*/true);
  }
}

TEST_P(GroupBy, PivotBasics) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width",  10.5],
      [2, "width",  11.5]
      ])",
                                         R"([
      [2, "height", 12.5]
      ])",
                                         R"([
      [3, "width",  13.5],
      [1, "height", 14.5]
      ])"};
  std::string expected_json = R"([
      [1, {"height": 14.5, "width": 10.5} ],
      [2, {"height": 12.5, "width": 11.5} ],
      [3, {"height": null, "width": 13.5} ]
      ])";
  for (auto unexpected_key_behavior :
       {PivotWiderOptions::kIgnore, PivotWiderOptions::kRaise}) {
    PivotWiderOptions options(/*key_names=*/{"height", "width"}, unexpected_key_behavior);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }
}

TEST_P(GroupBy, PivotBinaryKeyTypes) {
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10.5],
      [2, "width", 11.5]
      ])",
                                         R"([
      [2, "height", 12.5],
      [3, "width",  13.5],
      [1, "height", 14.5]
      ])"};
  std::string expected_json = R"([
      [1, {"height": 14.5, "width": 10.5} ],
      [2, {"height": 12.5, "width": 11.5} ],
      [3, {"height": null, "width": 13.5} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"height", "width"});

  for (const auto& key_type : BaseBinaryTypes()) {
    ARROW_SCOPED_TRACE("key_type = ", *key_type);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }

  auto key_type = fixed_size_binary(3);
  table_json = {R"([
      [1, "wid", 10.5],
      [2, "wid", 11.5]
      ])",
                R"([
      [2, "hei", 12.5],
      [3, "wid",  13.5],
      [1, "hei", 14.5]
      ])"};
  expected_json = R"([
      [1, {"hei": 14.5, "wid": 10.5} ],
      [2, {"hei": 12.5, "wid": 11.5} ],
      [3, {"hei": null, "wid": 13.5} ]
      ])";
  options.key_names = {"hei", "wid"};
  ARROW_SCOPED_TRACE("key_type = ", *key_type);
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotIntegerKeyTypes) {
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, 78, 10.5],
      [2, 78, 11.5]
      ])",
                                         R"([
      [2, 56, 12.5],
      [3, 78, 13.5],
      [1, 56, 14.5]
      ])"};
  std::string expected_json = R"([
      [1, {"56": 14.5, "78": 10.5} ],
      [2, {"56": 12.5, "78": 11.5} ],
      [3, {"56": null, "78": 13.5} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"56", "78"});

  for (const auto& key_type : IntTypes()) {
    ARROW_SCOPED_TRACE("key_type = ", *key_type);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }
}

TEST_P(GroupBy, PivotNumericValues) {
  auto key_type = utf8();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10],
      [2, "width", 11]
      ])",
                                         R"([
      [2, "height", 12],
      [3, "width",  13],
      [1, "height", 14]
      ])"};
  std::string expected_json = R"([
      [1, {"height": 14,   "width": 10} ],
      [2, {"height": 12,   "width": 11} ],
      [3, {"height": null, "width": 13} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"height", "width"});

  for (const auto& value_type : NumericTypes()) {
    ARROW_SCOPED_TRACE("value_type = ", *value_type);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }
}

TEST_P(GroupBy, PivotBinaryLikeValues) {
  auto key_type = utf8();
  std::vector<std::string> table_json = {R"([
      [1, "name",      "Bob"],
      [2, "eye_color", "brown"]
      ])",
                                         R"([
      [2, "name",      "Alice"],
      [1, "eye_color", "gray"],
      [3, "name",      "Mallaury"]
      ])"};
  std::string expected_json = R"([
      [1, {"name": "Bob",      "eye_color": "gray"} ],
      [2, {"name": "Alice",    "eye_color": "brown"} ],
      [3, {"name": "Mallaury", "eye_color": null} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"name", "eye_color"});

  for (const auto& value_type : BaseBinaryTypes()) {
    ARROW_SCOPED_TRACE("value_type = ", *value_type);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }
}

TEST_P(GroupBy, PivotDecimalValues) {
  auto key_type = utf8();
  auto value_type = decimal128(9, 1);
  std::vector<std::string> table_json = {R"([
      [1, "width", "10.1"],
      [2, "width", "11.1"]
      ])",
                                         R"([
      [2, "height", "12.1"],
      [3, "width",  "13.1"],
      [1, "height", "14.1"]
      ])"};
  std::string expected_json = R"([
      [1, {"height": "14.1", "width": "10.1"} ],
      [2, {"height": "12.1", "width": "11.1"} ],
      [3, {"height": null,   "width": "13.1"} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"height", "width"});
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotStructValues) {
  auto key_type = utf8();
  auto value_type = struct_({{"value", float32()}});
  std::vector<std::string> table_json = {R"([
      [1, "width", [10.1]],
      [2, "width", [11.1]]
      ])",
                                         R"([
      [2, "height", [12.1]],
      [3, "width",  [13.1]],
      [1, "height", [14.1]]
      ])"};
  std::string expected_json = R"([
      [1, {"height": [14.1], "width": [10.1]} ],
      [2, {"height": [12.1], "width": [11.1]} ],
      [3, {"height": null,   "width": [13.1]} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"height", "width"});
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotListValues) {
  auto key_type = utf8();
  auto value_type = list(float32());
  std::vector<std::string> table_json = {R"([
      [1, "foo", [10.5, 11.5]],
      [2, "bar", [12.5]]
      ])",
                                         R"([
      [2, "foo", []],
      [3, "bar", [13.5]],
      [1, "foo", null]
      ])"};
  std::string expected_json = R"([
      [1, {"foo": [10.5, 11.5], "bar": null}   ],
      [2, {"foo": [],           "bar": [12.5]} ],
      [3, {"foo": null,         "bar": [13.5]} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"foo", "bar"});
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotNullValueType) {
  auto key_type = utf8();
  auto value_type = null();
  std::vector<std::string> table_json = {R"([
      [1, "foo", null],
      [2, "bar", null]
      ])",
                                         R"([
      [2, "foo", null],
      [3, "bar", null],
      [1, "foo", null]
      ])"};
  std::string expected_json = R"([
      [1, {"foo": null, "bar": null} ],
      [2, {"foo": null, "bar": null} ],
      [3, {"foo": null, "bar": null} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"foo", "bar"});
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotNullValues) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10.5],
      [2, "width", null]
      ])",
                                         R"([
      [2, "height", 12.5],
      [2, "width",  13.5],
      [1, "width",  null],
      [2, "height", null]
      ])",
                                         R"([
      [1, "width",  null],
      [2, "height", null]
      ])"};
  std::string expected_json = R"([
      [1, {"height": null, "width": 10.5} ],
      [2, {"height": 12.5, "width": 13.5} ]
      ])";
  PivotWiderOptions options(/*key_names=*/{"height", "width"}, PivotWiderOptions::kRaise);
  TestPivot(key_type, value_type, options, table_json, expected_json);
}

TEST_P(GroupBy, PivotScalarKey) {
  BatchesWithSchema input;
  std::vector<TypeHolder> types = {int32(), utf8(), float32()};
  std::vector<ArgShape> shapes = {ArgShape::ARRAY, ArgShape::SCALAR, ArgShape::ARRAY};
  input.batches = {
      ExecBatchFromJSON(types, shapes,
                        R"([
        [1, "width",  10.5],
        [2, "width",  11.5]
        ])"),
      ExecBatchFromJSON(types, shapes,
                        R"([
        [2, "width",  null]
        ])"),
      ExecBatchFromJSON(types, shapes,
                        R"([
        [3, "height", null],
        [3, "height", null]
        ])"),
      ExecBatchFromJSON(types, shapes,
                        R"([
        [3, "height", 12.5],
        [1, "height", 13.5]
        ])"),
  };
  input.schema = schema({field("group_key", int32()), field("pivot_key", utf8()),
                         field("pivot_value", float32())});
  Datum expected = ArrayFromJSON(
      struct_({field("group_key", int32()),
               field("pivoted",
                     struct_({field("height", float32()), field("width", float32())}))}),
      R"([
      [1, {"height": 13.5, "width": 10.5} ],
      [2, {"height": null, "width": 11.5} ],
      [3, {"height": 12.5, "width": null} ]
      ])");
  auto options = std::make_shared<PivotWiderOptions>(
      PivotWiderOptions(/*key_names=*/{"height", "width"}));
  Aggregate aggregate{"hash_pivot_wider", options,
                      std::vector<FieldRef>{"pivot_key", "pivot_value"}, "pivoted"};
  for (bool use_threads : {false, true}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    ASSERT_OK_AND_ASSIGN(Datum actual,
                         RunGroupBy(input, {"group_key"}, {aggregate}, use_threads));
    ValidateOutput(actual);
    AssertDatumsApproxEqual(expected, actual, /*verbose=*/true);
  }
}

TEST_P(GroupBy, PivotUnusedKeyName) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10.5],
      [2, "width", 11.5]
      ])",
                                         R"([
      [2, "height", 12.5],
      [3, "width",  13.5],
      [1, "height", 14.5]
      ])"};
  std::string expected_json = R"([
      [1, {"height": 14.5, "depth": null, "width": 10.5} ],
      [2, {"height": 12.5, "depth": null, "width": 11.5} ],
      [3, {"height": null, "depth": null, "width": 13.5} ]
      ])";
  for (auto unexpected_key_behavior :
       {PivotWiderOptions::kIgnore, PivotWiderOptions::kRaise}) {
    PivotWiderOptions options(/*key_names=*/{"height", "depth", "width"},
                              unexpected_key_behavior);
    TestPivot(key_type, value_type, options, table_json, expected_json);
  }
}

TEST_P(GroupBy, PivotUnexpectedKeyName) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10.5],
      [2, "width", 11.5]
      ])",
                                         R"([
      [2, "height", 12.5],
      [3, "width",  13.5],
      [1, "depth",  15.5],
      [1, "height", 14.5]
      ])"};
  PivotWiderOptions options(/*key_names=*/{"height", "width"});
  std::string expected_json = R"([
      [1, {"height": 14.5, "width": 10.5} ],
      [2, {"height": 12.5, "width": 11.5} ],
      [3, {"height": null, "width": 13.5} ]
      ])";
  TestPivot(key_type, value_type, options, table_json, expected_json);
  options.unexpected_key_behavior = PivotWiderOptions::kRaise;
  for (bool use_threads : {false, true}) {
    ARROW_SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        KeyError, HasSubstr("Unexpected pivot key: depth"),
        RunPivot(key_type, value_type, options, table_json, use_threads));
  }
}
TEST_P(GroupBy, PivotNullKeys) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([
      [1, "width", 10.5],
      [2, null,    11.5]
      ])"};
  PivotWiderOptions options(/*key_names=*/{"height", "width"});
  for (bool use_threads : {false, true}) {
    ARROW_SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        KeyError, HasSubstr("pivot key name cannot be null"),
        RunPivot(key_type, value_type, options, table_json, use_threads));
  }
}

TEST_P(GroupBy, PivotDuplicateKeys) {
  auto key_type = utf8();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([])"};
  PivotWiderOptions options(/*key_names=*/{"height", "width", "height"});
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      KeyError, HasSubstr("Duplicate key name 'height' in PivotWiderOptions"),
      RunPivot(key_type, value_type, options, table_json));
}

TEST_P(GroupBy, PivotInvalidKeys) {
  // Integer key type, but key names cannot be converted to int
  auto key_type = int32();
  auto value_type = float32();
  std::vector<std::string> table_json = {R"([])"};
  PivotWiderOptions options(/*key_names=*/{"123", "width"});
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, HasSubstr("Failed to parse string: 'width' as a scalar of type int32"),
      RunPivot(key_type, value_type, options, table_json));
  options.key_names = {"12.3", "45"};
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, HasSubstr("Failed to parse string: '12.3' as a scalar of type int32"),
      RunPivot(key_type, value_type, options, table_json));
}

TEST_P(GroupBy, PivotDuplicateValues) {
  auto key_type = utf8();
  auto value_type = float32();
  PivotWiderOptions options(/*key_names=*/{"height", "width"});

  for (bool use_threads : {false, true}) {
    ARROW_SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");

    // Duplicate values in same chunk
    std::vector<std::string> table_json = {R"([
        [1, "width", 10.5],
        [2, "width", 11.5],
        [1, "width", 11.5]
        ])"};
    EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid,
                                    HasSubstr("Encountered more than one non-null value"),
                                    RunPivot(key_type, value_type, options, table_json));

    // Duplicate values in different chunks
    table_json = {R"([
        [1, "width", 10.5],
        [2, "width", 11.5]
        ])",
                  R"([
        [1, "width", 11.5]
        ])"};
    EXPECT_RAISES_WITH_MESSAGE_THAT(Invalid,
                                    HasSubstr("Encountered more than one non-null value"),
                                    RunPivot(key_type, value_type, options, table_json));
  }
}

TEST_P(GroupBy, PivotScalarKeyWithDuplicateValues) {
  BatchesWithSchema input;
  std::vector<TypeHolder> types = {int32(), utf8(), float32()};
  std::vector<ArgShape> shapes = {ArgShape::ARRAY, ArgShape::SCALAR, ArgShape::ARRAY};
  input.schema = schema({field("group_key", int32()), field("pivot_key", utf8()),
                         field("pivot_value", float32())});
  auto options = std::make_shared<PivotWiderOptions>(
      PivotWiderOptions(/*key_names=*/{"height", "width"}));
  Aggregate aggregate{"hash_pivot_wider", options,
                      std::vector<FieldRef>{"pivot_key", "pivot_value"}, "pivoted"};

  // Duplicate values in same chunk
  input.batches = {
      ExecBatchFromJSON(types, shapes,
                        R"([
        [1, "width",  10.5],
        [1, "width",  11.5]
        ])"),
  };
  for (bool use_threads : {false, true}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Encountered more than one non-null value"),
        RunGroupBy(input, {"group_key"}, {aggregate}, use_threads));
  }

  // Duplicate values in different chunks
  input.batches = {
      ExecBatchFromJSON(types, shapes,
                        R"([
        [1, "width",  10.5],
        [2, "width",  11.5]
        ])"),
      ExecBatchFromJSON(types, shapes,
                        R"([
        [2, "width",  12.5]
        ])"),
  };
  for (bool use_threads : {false, true}) {
    SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid, HasSubstr("Encountered more than one non-null value"),
        RunGroupBy(input, {"group_key"}, {aggregate}, use_threads));
  }
}

struct RandomPivotTestCase {
  PivotWiderOptions options;
  std::shared_ptr<RecordBatch> input;
  std::shared_ptr<Array> expected_output;
};

Result<RandomPivotTestCase> MakeRandomPivot(int64_t length) {
  constexpr double kKeyPresenceProbability = 0.8;
  constexpr double kValueValidityProbability = 0.7;

  const std::vector<std::string> key_names = {"height", "width", "depth"};
  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> value_dist(0.0f, 1.0f);
  std::bernoulli_distribution key_presence_dist(kKeyPresenceProbability);
  std::bernoulli_distribution value_validity_dist(kValueValidityProbability);

  Int64Builder group_key_builder;
  StringBuilder key_builder;
  FloatBuilder value_builder;
  RETURN_NOT_OK(group_key_builder.Reserve(length));
  RETURN_NOT_OK(key_builder.Reserve(length));
  RETURN_NOT_OK(value_builder.Reserve(length));

  // The last input key name will not be part of the result
  PivotWiderOptions options(
      std::vector<std::string>(key_names.begin(), key_names.end() - 1));
  Int64Builder pivoted_group_builder;
  std::vector<FloatBuilder> pivoted_value_builders(options.key_names.size());

  auto finish_group = [&](int64_t group_key) -> Status {
    // First check if *any* pivoted column was populated (otherwise there was
    // no valid value at all in this group, and no output row should be generated).
    RETURN_NOT_OK(pivoted_group_builder.Append(group_key));
    // Make sure all pivoted columns are populated and in sync with the group key column
    for (auto& pivoted_value_builder : pivoted_value_builders) {
      if (pivoted_value_builder.length() < pivoted_group_builder.length()) {
        RETURN_NOT_OK(pivoted_value_builder.AppendNull());
      }
      EXPECT_EQ(pivoted_value_builder.length(), pivoted_group_builder.length());
    }
    return Status::OK();
  };

  int64_t group_key = 1000;
  bool group_started = false;
  int key_id = 0;
  while (group_key_builder.length() < length) {
    // For the current group_key and key_id we can either:
    // 1. not add a row
    // 2. add a row with a null value
    // 3. add a row with a non-null value
    //    3a. the row will end up in the pivoted data iff the key is part of
    //        the PivotWiderOptions.key_names
    if (key_presence_dist(gen)) {
      group_key_builder.UnsafeAppend(group_key);
      group_started = true;
      RETURN_NOT_OK(key_builder.Append(key_names[key_id]));
      if (value_validity_dist(gen)) {
        const auto value = value_dist(gen);
        value_builder.UnsafeAppend(value);
        if (key_id < static_cast<int>(pivoted_value_builders.size())) {
          RETURN_NOT_OK(pivoted_value_builders[key_id].Append(value));
        }
      } else {
        value_builder.UnsafeAppendNull();
      }
    }
    if (++key_id >= static_cast<int>(key_names.size())) {
      // We've considered all keys for this group.
      // Emit a pivoted row only if any key was emitted in the input.
      if (group_started) {
        RETURN_NOT_OK(finish_group(group_key));
      }
      // Initiate new group
      ++group_key;
      group_started = false;
      key_id = 0;
    }
  }
  if (group_started) {
    // We've started this group, finish it
    RETURN_NOT_OK(finish_group(group_key));
  }
  ARROW_ASSIGN_OR_RAISE(auto group_keys, group_key_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto keys, key_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto values, value_builder.Finish());
  auto input_schema =
      schema({{"group_key", int64()}, {"key", utf8()}, {"value", float32()}});
  auto input = RecordBatch::Make(input_schema, length, {group_keys, keys, values});
  RETURN_NOT_OK(input->Validate());

  ARROW_ASSIGN_OR_RAISE(auto pivoted_groups, pivoted_group_builder.Finish());
  ArrayVector pivoted_value_columns;
  for (auto& pivoted_value_builder : pivoted_value_builders) {
    ARROW_ASSIGN_OR_RAISE(pivoted_value_columns.emplace_back(),
                          pivoted_value_builder.Finish());
  }
  ARROW_ASSIGN_OR_RAISE(
      auto pivoted_values,
      StructArray::Make(std::move(pivoted_value_columns), options.key_names));
  ARROW_ASSIGN_OR_RAISE(auto output,
                        StructArray::Make({pivoted_groups, pivoted_values},
                                          std::vector<std::string>{"key_0", "out"}));
  RETURN_NOT_OK(output->Validate());

  return RandomPivotTestCase{std::move(options), std::move(input), std::move(output)};
}

TEST_P(GroupBy, PivotRandom) {
  constexpr int64_t kLength = 900;
  // Larger than 256 to exercise take-index dispatch in pivot implementation
  constexpr int64_t kChunkLength = 300;
  ASSERT_OK_AND_ASSIGN(auto pivot_case, MakeRandomPivot(kLength));

  for (bool shuffle : {false, true}) {
    ARROW_SCOPED_TRACE("shuffle = ", shuffle);
    auto input = Datum(pivot_case.input);
    if (shuffle) {
      // Since the "value" column is random-generated, sorting on it produces
      // a random shuffle.
      ASSERT_OK_AND_ASSIGN(
          auto shuffle_indices,
          SortIndices(pivot_case.input, SortOptions({SortKey("value")})));
      ASSERT_OK_AND_ASSIGN(input, Take(input, shuffle_indices));
    }
    ASSERT_EQ(input.kind(), Datum::RECORD_BATCH);
    RecordBatchVector chunks;
    for (int64_t start = 0; start < kLength; start += kChunkLength) {
      const auto chunk_length = std::min(kLength - start, kChunkLength);
      chunks.push_back(input.record_batch()->Slice(start, chunk_length));
    }
    ASSERT_OK_AND_ASSIGN(auto table, Table::FromRecordBatches(chunks));

    for (bool use_threads : {false, true}) {
      ARROW_SCOPED_TRACE(use_threads ? "parallel/merged" : "serial");
      ASSERT_OK_AND_ASSIGN(auto pivoted, RunPivot(utf8(), float32(), pivot_case.options,
                                                  table, use_threads));
      // XXX For some reason this works even in the shuffled case
      // (I would expect the test to require sorting of the output).
      // This might depend on implementation details of group id generation
      // by the hash-aggregate logic (the pivot implementation implicitly
      // orders the output by ascending group id).
      AssertDatumsEqual(pivot_case.expected_output, pivoted, /*verbose=*/true);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GroupBy, GroupBy, ::testing::Values(RunGroupByImpl));

class SegmentedScalarGroupBy : public GroupBy {};

class SegmentedKeyGroupBy : public GroupBy {};

void TestSegment(GroupByFunction group_by, const std::shared_ptr<Table>& table,
                 Datum output, const std::vector<Datum>& keys,
                 const std::vector<Datum>& segment_keys, bool is_scalar_aggregate) {
  const char* names[] = {
      is_scalar_aggregate ? "count" : "hash_count",
      is_scalar_aggregate ? "sum" : "hash_sum",
      is_scalar_aggregate ? "min_max" : "hash_min_max",
      is_scalar_aggregate ? "first_last" : "hash_first_last",
      is_scalar_aggregate ? "first" : "hash_first",
      is_scalar_aggregate ? "last" : "hash_last",
  };
  ASSERT_OK_AND_ASSIGN(Datum aggregated_and_grouped,
                       group_by(
                           {
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                               table->GetColumnByName("argument"),
                           },
                           keys, segment_keys,
                           {
                               {names[0], nullptr, "agg_0", names[0]},
                               {names[1], nullptr, "agg_1", names[1]},
                               {names[2], nullptr, "agg_2", names[2]},
                               {names[3], nullptr, "agg_3", names[3]},
                               {names[4], nullptr, "agg_4", names[4]},
                               {names[5], nullptr, "agg_5", names[5]},
                           },
                           /*use_threads=*/false, /*naive=*/false));

  AssertDatumsEqual(output, aggregated_and_grouped, /*verbose=*/true);
}

// test with empty keys, covering code in ScalarAggregateNode
void TestSegmentScalar(GroupByFunction group_by, const std::shared_ptr<Table>& table,
                       Datum output, const std::vector<Datum>& segment_keys) {
  TestSegment(group_by, table, output, {}, segment_keys, /*scalar=*/true);
}

// test with given segment-keys and keys set to `{"key"}`, covering code in GroupByNode
void TestSegmentKey(GroupByFunction group_by, const std::shared_ptr<Table>& table,
                    Datum output, const std::vector<Datum>& segment_keys) {
  TestSegment(group_by, table, output, {table->GetColumnByName("key")}, segment_keys,
              /*scalar=*/false);
}

Result<std::shared_ptr<Table>> GetSingleSegmentInputAsChunked() {
  auto table = TableFromJSON(schema({field("segment_key", int64()), field("key", int64()),
                                     field("argument", float64())}),
                             {R"([{"argument": 1.0,   "key": 1,    "segment_key": 1},
                         {"argument": null,  "key": 1,    "segment_key": 1}
                        ])",
                              R"([
                          {"argument": 0.0,   "key": 2,    "segment_key": 1},
                          {"argument": null,  "key": 3,    "segment_key": 1},
                          {"argument": 4.0,   "key": null, "segment_key": 1},
                          {"argument": 3.25,  "key": 1,    "segment_key": 1},
                          {"argument": 0.125, "key": 2,    "segment_key": 1},
                          {"argument": -0.25, "key": 2,    "segment_key": 1},
                          {"argument": 0.75,  "key": null, "segment_key": 1},
                          {"argument": null,  "key": 3,    "segment_key": 1}
                        ])",
                              R"([
                          {"argument": 1.0,   "key": 1,    "segment_key": 0},
                          {"argument": null,  "key": 1,    "segment_key": 0}
                        ])",
                              R"([
                          {"argument": 0.0,   "key": 2,    "segment_key": 0},
                          {"argument": null,  "key": 3,    "segment_key": 0},
                          {"argument": 4.0,   "key": null, "segment_key": 0},
                          {"argument": 3.25,  "key": 1,    "segment_key": 0},
                          {"argument": 0.125, "key": 2,    "segment_key": 0},
                          {"argument": -0.25, "key": 2,    "segment_key": 0},
                          {"argument": 0.75,  "key": null, "segment_key": 0},
                          {"argument": null,  "key": 3,    "segment_key": 0}
                        ])"});
  return table;
}

Result<std::shared_ptr<Table>> GetSingleSegmentInputAsCombined() {
  ARROW_ASSIGN_OR_RAISE(auto table, GetSingleSegmentInputAsChunked());
  return table->CombineChunks();
}

Result<std::shared_ptr<ChunkedArray>> GetSingleSegmentScalarOutput() {
  return ChunkedArrayFromJSON(
      struct_({
          field("key_0", int64()),
          field("count", int64()),
          field("sum", float64()),
          field("min_max", struct_({
                               field("min", float64()),
                               field("max", float64()),
                           })),
          field("first_last",
                struct_({field("first", float64()), field("last", float64())})),
          field("first", float64()),
          field("last", float64()),
      }),
      {R"([
    [1, 7, 8.875, {"min": -0.25, "max": 4.0}, {"first": 1.0, "last": 0.75}, 1.0, 0.75]
  ])",
       R"([
    [0, 7, 8.875, {"min": -0.25, "max": 4.0}, {"first": 1.0, "last": 0.75}, 1.0, 0.75]

  ])"});
}

Result<std::shared_ptr<ChunkedArray>> GetSingleSegmentKeyOutput() {
  return ChunkedArrayFromJSON(struct_({
                                  field("key_1", int64()),
                                  field("key_0", int64()),
                                  field("hash_count", int64()),
                                  field("hash_sum", float64()),
                                  field("hash_min_max", struct_({
                                                            field("min", float64()),
                                                            field("max", float64()),
                                                        })),
                                  field("hash_first_last", struct_({
                                                               field("first", float64()),
                                                               field("last", float64()),
                                                           })),
                                  field("hash_first", float64()),
                                  field("hash_last", float64()),
                              }),
                              {R"([
    [1,    1, 2, 4.25,   {"min": 1.0,   "max": 3.25}, {"first": 1.0, "last": 3.25}, 1.0, 3.25 ],
    [1,    2, 3, -0.125, {"min": -0.25, "max": 0.125}, {"first": 0.0, "last": -0.25}, 0.0, -0.25],
    [1,    3, 0, null,   {"min": null,  "max": null}, {"first": null, "last": null}, null, null],
    [1, null, 2, 4.75,   {"min": 0.75,  "max": 4.0},  {"first": 4.0, "last": 0.75}, 4.0, 0.75]
  ])",
                               R"([
    [0,    1, 2, 4.25,   {"min": 1.0,   "max": 3.25}, {"first": 1.0, "last": 3.25}, 1.0, 3.25 ],
    [0,    2, 3, -0.125, {"min": -0.25, "max": 0.125}, {"first": 0.0, "last": -0.25}, 0.0, -0.25],
    [0,    3, 0, null,   {"min": null,  "max": null}, {"first": null, "last": null}, null, null],
    [0, null, 2, 4.75,   {"min": 0.75,  "max": 4.0}, {"first": 4.0, "last": 0.75}, 4.0, 0.75]
  ])"});
}

void TestSingleSegmentScalar(GroupByFunction group_by,
                             std::function<Result<std::shared_ptr<Table>>()> get_table) {
  ASSERT_OK_AND_ASSIGN(auto table, get_table());
  ASSERT_OK_AND_ASSIGN(auto output, GetSingleSegmentScalarOutput());
  TestSegmentScalar(group_by, table, output, {table->GetColumnByName("segment_key")});
}

void TestSingleSegmentKey(GroupByFunction group_by,
                          std::function<Result<std::shared_ptr<Table>>()> get_table) {
  ASSERT_OK_AND_ASSIGN(auto table, get_table());
  ASSERT_OK_AND_ASSIGN(auto output, GetSingleSegmentKeyOutput());
  TestSegmentKey(group_by, table, output, {table->GetColumnByName("segment_key")});
}

TEST_P(SegmentedScalarGroupBy, SingleSegmentScalarChunked) {
  TestSingleSegmentScalar(GetParam(), GetSingleSegmentInputAsChunked);
}

TEST_P(SegmentedScalarGroupBy, SingleSegmentScalarCombined) {
  TestSingleSegmentScalar(GetParam(), GetSingleSegmentInputAsCombined);
}

TEST_P(SegmentedKeyGroupBy, SingleSegmentKeyChunked) {
  TestSingleSegmentKey(GetParam(), GetSingleSegmentInputAsChunked);
}

TEST_P(SegmentedKeyGroupBy, SingleSegmentKeyCombined) {
  TestSingleSegmentKey(GetParam(), GetSingleSegmentInputAsCombined);
}

// extracts one segment of the obtained (single-segment-key) table
Result<std::shared_ptr<Table>> GetEmptySegmentKeysInput(
    std::function<Result<std::shared_ptr<Table>>()> get_table) {
  ARROW_ASSIGN_OR_RAISE(auto table, get_table());
  auto sliced = table->Slice(0, 10);
  ARROW_ASSIGN_OR_RAISE(auto batch, sliced->CombineChunksToBatch());
  ARROW_ASSIGN_OR_RAISE(auto array, batch->ToStructArray());
  ARROW_ASSIGN_OR_RAISE(auto chunked, ChunkedArray::Make({array}, array->type()));
  return Table::FromChunkedStructArray(chunked);
}

Result<std::shared_ptr<Table>> GetEmptySegmentKeysInputAsChunked() {
  return GetEmptySegmentKeysInput(GetSingleSegmentInputAsChunked);
}

Result<std::shared_ptr<Table>> GetEmptySegmentKeysInputAsCombined() {
  return GetEmptySegmentKeysInput(GetSingleSegmentInputAsCombined);
}

// extracts the expected output for one segment
Result<std::shared_ptr<Array>> GetEmptySegmentKeyOutput() {
  ARROW_ASSIGN_OR_RAISE(auto chunked, GetSingleSegmentKeyOutput());
  ARROW_ASSIGN_OR_RAISE(auto table, Table::FromChunkedStructArray(chunked));
  ARROW_ASSIGN_OR_RAISE(auto removed, table->RemoveColumn(0));
  auto sliced = removed->Slice(0, 4);
  ARROW_ASSIGN_OR_RAISE(auto batch, sliced->CombineChunksToBatch());
  return batch->ToStructArray();
}

void TestEmptySegmentKey(GroupByFunction group_by,
                         std::function<Result<std::shared_ptr<Table>>()> get_table) {
  ASSERT_OK_AND_ASSIGN(auto table, get_table());
  ASSERT_OK_AND_ASSIGN(auto output, GetEmptySegmentKeyOutput());
  TestSegmentKey(group_by, table, output, {});
}

TEST_P(SegmentedKeyGroupBy, EmptySegmentKeyChunked) {
  TestEmptySegmentKey(GetParam(), GetEmptySegmentKeysInputAsChunked);
}

TEST_P(SegmentedKeyGroupBy, EmptySegmentKeyCombined) {
  TestEmptySegmentKey(GetParam(), GetEmptySegmentKeysInputAsCombined);
}

// adds a named copy of the first (single-segment-key) column to the obtained table
Result<std::shared_ptr<Table>> GetMultiSegmentInput(
    std::function<Result<std::shared_ptr<Table>>()> get_table,
    const std::string& add_name) {
  ARROW_ASSIGN_OR_RAISE(auto table, get_table());
  auto add_field = field(add_name, table->schema()->field(0)->type());
  return table->AddColumn(table->num_columns(), add_field, table->column(0));
}

Result<std::shared_ptr<Table>> GetMultiSegmentInputAsChunked(
    const std::string& add_name) {
  return GetMultiSegmentInput(GetSingleSegmentInputAsChunked, add_name);
}

Result<std::shared_ptr<Table>> GetMultiSegmentInputAsCombined(
    const std::string& add_name) {
  return GetMultiSegmentInput(GetSingleSegmentInputAsCombined, add_name);
}

// adds a named copy of the first(single-segment-key) column to the expected output table
Result<std::shared_ptr<ChunkedArray>> GetMultiSegmentKeyOutput(
    const std::string& add_name) {
  ARROW_ASSIGN_OR_RAISE(auto chunked, GetSingleSegmentKeyOutput());
  ARROW_ASSIGN_OR_RAISE(auto table, Table::FromChunkedStructArray(chunked));
  int existing_key_field_idx = 0;
  auto add_field =
      field(add_name, table->schema()->field(existing_key_field_idx)->type());
  ARROW_ASSIGN_OR_RAISE(auto added,
                        table->AddColumn(existing_key_field_idx + 1, add_field,
                                         table->column(existing_key_field_idx)));
  ARROW_ASSIGN_OR_RAISE(auto batch, added->CombineChunksToBatch());
  ARROW_ASSIGN_OR_RAISE(auto array, batch->ToStructArray());
  return ChunkedArray::Make({array->Slice(0, 4), array->Slice(4, 4)}, array->type());
}

void TestMultiSegmentKey(
    GroupByFunction group_by,
    std::function<Result<std::shared_ptr<Table>>(const std::string&)> get_table) {
  std::string add_name = "segment_key2";
  ASSERT_OK_AND_ASSIGN(auto table, get_table(add_name));
  ASSERT_OK_AND_ASSIGN(auto output, GetMultiSegmentKeyOutput("key_2"));
  TestSegmentKey(
      group_by, table, output,
      {table->GetColumnByName("segment_key"), table->GetColumnByName(add_name)});
}

TEST_P(SegmentedKeyGroupBy, MultiSegmentKeyChunked) {
  TestMultiSegmentKey(GetParam(), GetMultiSegmentInputAsChunked);
}

TEST_P(SegmentedKeyGroupBy, MultiSegmentKeyCombined) {
  TestMultiSegmentKey(GetParam(), GetMultiSegmentInputAsCombined);
}

TEST_P(SegmentedKeyGroupBy, PivotSegmentKey) {
  auto group_by = GetParam();
  auto key_type = utf8();
  auto value_type = float32();

  std::vector<std::string> table_json = {R"([
      [1, "width",  10.5],
      [1, "height", 11.5]
      ])",
                                         R"([
      [2, "height", 12.5],
      [2, "width",  13.5],
      [3, "width",  14.5]
      ])",
                                         R"([
      [3, "width",  null],
      [4, "height", 15.5]
      ])"};
  std::vector<std::string> expected_json = {
      R"([[1, {"height": 11.5, "width": 10.5}]])",
      R"([[2, {"height": 12.5, "width": 13.5}]])",
      R"([[3, {"height": null, "width": 14.5}]])",
      R"([[4, {"height": 15.5, "width": null}]])",
  };

  auto table =
      TableFromJSON(schema({field("segment_key", int64()), field("pivot_key", key_type),
                            field("pivot_value", value_type)}),
                    table_json);

  auto options = std::make_shared<PivotWiderOptions>(
      PivotWiderOptions(/*key_names=*/{"height", "width"}));
  Aggregate aggregate{"pivot_wider", options, std::vector<FieldRef>{"agg_0", "agg_1"},
                      "pivoted"};
  ASSERT_OK_AND_ASSIGN(Datum actual,
                       group_by(
                           {
                               table->GetColumnByName("pivot_key"),
                               table->GetColumnByName("pivot_value"),
                           },
                           {}, {table->GetColumnByName("segment_key")}, {aggregate},
                           /*use_threads=*/false, /*naive=*/false));
  ValidateOutput(actual);
  auto expected = ChunkedArrayFromJSON(
      struct_({field("key_0", int64()),
               field("pivoted", struct_({field("height", value_type),
                                         field("width", value_type)}))}),
      expected_json);
  AssertDatumsEqual(expected, actual, /*verbose=*/true);
}

TEST_P(SegmentedKeyGroupBy, PivotSegmentKeyDuplicateValues) {
  // NOTE: besides testing "pivot_wider" behavior, this test also checks that errors
  // produced when consuming or merging an aggregate don't corrupt
  // execution engine internals.
  auto group_by = GetParam();
  auto key_type = utf8();
  auto value_type = float32();
  auto options = std::make_shared<PivotWiderOptions>(
      PivotWiderOptions(/*key_names=*/{"height", "width"}));
  auto table_schema = schema({field("segment_key", int64()), field("pivot_key", key_type),
                              field("pivot_value", value_type)});

  auto test_duplicate_values = [&](const std::vector<std::string>& table_json) {
    auto table = TableFromJSON(table_schema, table_json);
    Aggregate aggregate{"pivot_wider", options, std::vector<FieldRef>{"agg_0", "agg_1"},
                        "pivoted"};
    EXPECT_RAISES_WITH_MESSAGE_THAT(
        Invalid,
        HasSubstr("Encountered more than one non-null value for the same pivot key"),
        group_by(
            {
                table->GetColumnByName("pivot_key"),
                table->GetColumnByName("pivot_value"),
            },
            {}, {table->GetColumnByName("segment_key")}, {aggregate},
            /*use_threads=*/false, /*naive=*/false));
  };

  // Duplicate values in the same chunk
  test_duplicate_values({R"([
      [1, "width",  10.5],
      [2, "width",  11.5],
      [2, "width",  12.5]
      ])"});
  // Duplicate values in two different chunks
  test_duplicate_values({R"([
      [1, "width",  10.5],
      [2, "width",  11.5]
      ])",
                         R"([
      [2, "width",  12.5]
      ])"});
}

INSTANTIATE_TEST_SUITE_P(SegmentedScalarGroupBy, SegmentedScalarGroupBy,
                         ::testing::Values(RunSegmentedGroupByImpl));

INSTANTIATE_TEST_SUITE_P(SegmentedKeyGroupBy, SegmentedKeyGroupBy,
                         ::testing::Values(RunSegmentedGroupByImpl));

}  // namespace acero
}  // namespace arrow
