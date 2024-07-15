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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "arrow/array/array_binary.h"
#include "arrow/array/array_dict.h"
#include "arrow/array/array_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/concatenate.h"
#include "arrow/buffer_builder.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/kernels/common_internal.h"
#include "arrow/compute/kernels/util_internal.h"
#include "arrow/extension_type.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/bitmap_reader.h"
#include "arrow/util/int_util.h"

namespace arrow {

using internal::BinaryBitBlockCounter;
using internal::BitBlockCount;
using internal::BitBlockCounter;
using internal::CheckIndexBounds;
using internal::CopyBitmap;
using internal::CountSetBits;
using internal::OptionalBitBlockCounter;
using internal::OptionalBitIndexer;

namespace compute {
namespace internal {

namespace {

struct ScatterKernelData {
  InputType value_type;
  InputType selection_type;
  ArrayKernelExec exec;
};

const FunctionDoc array_scatter_doc(
    "Scatter with a boolean positional mask",
    ("The values of the input `array` will be placed into the output at positions where "
     "the `positional_mask` is non-zero.  The rest positions of the output will be "
     "populated by `null`s.\n"),
    {"array", "positional_mask"});

// ----------------------------------------------------------------------
// Optimized and streamlined scatter for primitive types

Status PrimitiveScatterExec(KernelContext* ctx, const ExecSpan& batch, ExecResult* out) {
  const ArraySpan& values = batch[0].array;
  const ArraySpan& filter = batch[1].array;
  const bool is_ree_filter = filter.type->id() == Type::RUN_END_ENCODED;

  int64_t output_length = GetFilterOutputSize(filter, null_selection);

  ArrayData* out_arr = out->array_data().get();

  const bool filter_null_count_is_zero =
      is_ree_filter ? filter.child_data[1].null_count == 0 : filter.null_count == 0;

  // The output precomputed null count is unknown except in the narrow
  // condition that all the values are non-null and the filter will not cause
  // any new nulls to be created.
  if (values.null_count == 0 &&
      (null_selection == FilterOptions::DROP || filter_null_count_is_zero)) {
    out_arr->null_count = 0;
  } else {
    out_arr->null_count = kUnknownNullCount;
  }

  // When neither the values nor filter is known to have any nulls, we will
  // elect the optimized non-null path where there is no need to populate a
  // validity bitmap.
  const bool allocate_validity = values.null_count != 0 || !filter_null_count_is_zero;

  DCHECK(util::IsFixedWidthLike(values));
  const int64_t bit_width = util::FixedWidthInBits(*values.type);
  RETURN_NOT_OK(util::internal::PreallocateFixedWidthArrayData(
      ctx, output_length, /*source=*/values, allocate_validity, out_arr));

  switch (bit_width) {
    case 1:
      PrimitiveFilterImpl<1, /*kIsBoolean=*/true>(values, filter, null_selection, out_arr)
          .Exec();
      break;
    case 8:
      PrimitiveFilterImpl<1>(values, filter, null_selection, out_arr).Exec();
      break;
    case 16:
      PrimitiveFilterImpl<2>(values, filter, null_selection, out_arr).Exec();
      break;
    case 32:
      PrimitiveFilterImpl<4>(values, filter, null_selection, out_arr).Exec();
      break;
    case 64:
      PrimitiveFilterImpl<8>(values, filter, null_selection, out_arr).Exec();
      break;
    case 128:
      // For INTERVAL_MONTH_DAY_NANO, DECIMAL128
      PrimitiveFilterImpl<16>(values, filter, null_selection, out_arr).Exec();
      break;
    case 256:
      // For DECIMAL256
      PrimitiveFilterImpl<32>(values, filter, null_selection, out_arr).Exec();
      break;
    default:
      // Non-specializing on byte width
      PrimitiveFilterImpl<-1>(values, filter, null_selection, out_arr).Exec();
      break;
  }
  return Status::OK();
}

// ----------------------------------------------------------------------
// Implement Scatter metafunction

Result<std::shared_ptr<RecordBatch>> ScatterRecordBatch(const RecordBatch& batch,
                                                        const Datum& filter,
                                                        const FunctionOptions* options,
                                                        ExecContext* ctx) {
  if (batch.num_rows() != filter.length()) {
    return Status::Invalid("Filter inputs must all be the same length");
  }

  // Fetch filter
  const auto& filter_opts = *static_cast<const FilterOptions*>(options);
  ArrayData filter_array;
  switch (filter.kind()) {
    case Datum::ARRAY:
      filter_array = *filter.array();
      break;
    case Datum::CHUNKED_ARRAY: {
      ARROW_ASSIGN_OR_RAISE(auto combined, Concatenate(filter.chunked_array()->chunks()));
      filter_array = *combined->data();
      break;
    }
    default:
      return Status::TypeError("Filter should be array-like");
  }

  // Convert filter to selection vector/indices and use Take
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<ArrayData> indices,
                        GetTakeIndices(filter_array, filter_opts.null_selection_behavior,
                                       ctx->memory_pool()));
  std::vector<std::shared_ptr<Array>> columns(batch.num_columns());
  for (int i = 0; i < batch.num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(Datum out, Take(batch.column(i)->data(), Datum(indices),
                                          TakeOptions::NoBoundsCheck(), ctx));
    columns[i] = out.make_array();
  }
  return RecordBatch::Make(batch.schema(), indices->length, std::move(columns));
}

Result<std::shared_ptr<Table>> ScatterTable(const Table& table, const Datum& filter,
                                            const FunctionOptions* options,
                                            ExecContext* ctx) {
  if (table.num_rows() != filter.length()) {
    return Status::Invalid("Filter inputs must all be the same length");
  }
  if (table.num_rows() == 0) {
    return Table::Make(table.schema(), table.columns(), 0);
  }

  // Last input element will be the filter array
  const int num_columns = table.num_columns();
  std::vector<ArrayVector> inputs(num_columns + 1);

  // Fetch table columns
  for (int i = 0; i < num_columns; ++i) {
    inputs[i] = table.column(i)->chunks();
  }
  // Fetch filter
  const auto& filter_opts = *static_cast<const FilterOptions*>(options);
  switch (filter.kind()) {
    case Datum::ARRAY:
      inputs.back().push_back(filter.make_array());
      break;
    case Datum::CHUNKED_ARRAY:
      inputs.back() = filter.chunked_array()->chunks();
      break;
    default:
      return Status::TypeError("Filter should be array-like");
  }

  // Rechunk inputs to allow consistent iteration over their respective chunks
  inputs = arrow::internal::RechunkArraysConsistently(inputs);

  // Instead of filtering each column with the boolean filter
  // (which would be slow if the table has a large number of columns: ARROW-10569),
  // convert each filter chunk to indices, and take() the column.
  const int64_t num_chunks = static_cast<int64_t>(inputs.back().size());
  std::vector<ArrayVector> out_columns(num_columns);
  int64_t out_num_rows = 0;

  for (int64_t i = 0; i < num_chunks; ++i) {
    const ArrayData& filter_chunk = *inputs.back()[i]->data();
    ARROW_ASSIGN_OR_RAISE(
        const auto indices,
        GetTakeIndices(filter_chunk, filter_opts.null_selection_behavior,
                       ctx->memory_pool()));

    if (indices->length > 0) {
      // Take from all input columns
      Datum indices_datum{std::move(indices)};
      for (int col = 0; col < num_columns; ++col) {
        const auto& column_chunk = inputs[col][i];
        ARROW_ASSIGN_OR_RAISE(Datum out, Take(column_chunk, indices_datum,
                                              TakeOptions::NoBoundsCheck(), ctx));
        out_columns[col].push_back(std::move(out).make_array());
      }
      out_num_rows += indices->length;
    }
  }

  ChunkedArrayVector out_chunks(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    out_chunks[i] = std::make_shared<ChunkedArray>(std::move(out_columns[i]),
                                                   table.column(i)->type());
  }
  return Table::Make(table.schema(), std::move(out_chunks), out_num_rows);
}

const FunctionDoc scatter_doc(
    "Scatter with a boolean selection filter",
    ("The output is populated with values from the input at positions\n"
     "where the selection filter is non-zero.  Nulls in the selection filter\n"
     "are handled based on FilterOptions."),
    {"input", "selection_filter"}, "FilterOptions");

class ScatterMetaFunction : public MetaFunction {
 public:
  ScatterMetaFunction()
      : MetaFunction("scatter", Arity::Binary(), scatter_doc, NULLPTR) {}

  Result<Datum> ExecuteImpl(const std::vector<Datum>& args,
                            const FunctionOptions* options,
                            ExecContext* ctx) const override {
    if (args[1].kind() != Datum::ARRAY && args[1].kind() != Datum::CHUNKED_ARRAY) {
      return Status::TypeError("Filter should be array-like");
    }

    const auto& filter_type = *args[1].type();
    const bool filter_is_plain_bool = filter_type.id() == Type::BOOL;
    const bool filter_is_ree_bool =
        filter_type.id() == Type::RUN_END_ENCODED &&
        checked_cast<const arrow::RunEndEncodedType&>(filter_type).value_type()->id() ==
            Type::BOOL;
    if (!filter_is_plain_bool && !filter_is_ree_bool) {
      return Status::NotImplemented("Filter argument must be boolean type");
    }

    if (args[0].kind() == Datum::RECORD_BATCH) {
      ARROW_ASSIGN_OR_RAISE(
          std::shared_ptr<RecordBatch> out_batch,
          ScatterRecordBatch(*args[0].record_batch(), args[1], options, ctx));
      return Datum(out_batch);
    } else if (args[0].kind() == Datum::TABLE) {
      ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Table> out_table,
                            ScatterTable(*args[0].table(), args[1], options, ctx));
      return Datum(out_table);
    } else {
      return CallFunction("array_scatter", args, options, ctx);
    }
  }
};

// ----------------------------------------------------------------------

}  // namespace

void RegisterVectorScatter(FunctionRegistry* registry) {
  // array_scatter
  {
    auto plain_mask = InputType(Type::BOOL);
    // auto ree_mask = InputType(match::RunEndEncoded(Type::BOOL));
    std::vector<ScatterKernelData> array_scatter_kernels{
        // * x Boolean
        {InputType(match::Primitive()), plain_mask, PrimitiveScatterExec},
        // {InputType(match::BinaryLike()), plain_filter, BinaryFilterExec},
        // {InputType(match::LargeBinaryLike()), plain_filter, BinaryFilterExec},
        // {InputType(null()), plain_filter, NullFilterExec},
        // {InputType(Type::FIXED_SIZE_BINARY), plain_filter, PrimitiveFilterExec},
        // {InputType(Type::DECIMAL128), plain_filter, PrimitiveFilterExec},
        // {InputType(Type::DECIMAL256), plain_filter, PrimitiveFilterExec},
        // {InputType(Type::DICTIONARY), plain_filter, DictionaryFilterExec},
        // {InputType(Type::EXTENSION), plain_filter, ExtensionFilterExec},
        // {InputType(Type::LIST), plain_filter, ListFilterExec},
        // {InputType(Type::LARGE_LIST), plain_filter, LargeListFilterExec},
        // {InputType(Type::LIST_VIEW), plain_filter, ListViewFilterExec},
        // {InputType(Type::LARGE_LIST_VIEW), plain_filter, LargeListViewFilterExec},
        // {InputType(Type::FIXED_SIZE_LIST), plain_filter, FSLFilterExec},
        // {InputType(Type::DENSE_UNION), plain_filter, DenseUnionFilterExec},
        // {InputType(Type::SPARSE_UNION), plain_filter, SparseUnionFilterExec},
        // {InputType(Type::STRUCT), plain_filter, StructFilterExec},
        // {InputType(Type::MAP), plain_filter, MapFilterExec},

        // * x REE(Boolean)
        // {InputType(match::Primitive()), ree_filter, PrimitiveFilterExec},
        // {InputType(match::BinaryLike()), ree_filter, BinaryFilterExec},
        // {InputType(match::LargeBinaryLike()), ree_filter, BinaryFilterExec},
        // {InputType(null()), ree_filter, NullFilterExec},
        // {InputType(Type::FIXED_SIZE_BINARY), ree_filter, PrimitiveFilterExec},
        // {InputType(Type::DECIMAL128), ree_filter, PrimitiveFilterExec},
        // {InputType(Type::DECIMAL256), ree_filter, PrimitiveFilterExec},
        // {InputType(Type::DICTIONARY), ree_filter, DictionaryFilterExec},
        // {InputType(Type::EXTENSION), ree_filter, ExtensionFilterExec},
        // {InputType(Type::LIST), ree_filter, ListFilterExec},
        // {InputType(Type::LARGE_LIST), ree_filter, LargeListFilterExec},
        // {InputType(Type::LIST_VIEW), ree_filter, ListViewFilterExec},
        // {InputType(Type::LARGE_LIST_VIEW), ree_filter, LargeListViewFilterExec},
        // {InputType(Type::FIXED_SIZE_LIST), ree_filter, FSLFilterExec},
        // {InputType(Type::DENSE_UNION), ree_filter, DenseUnionFilterExec},
        // {InputType(Type::SPARSE_UNION), ree_filter, SparseUnionFilterExec},
        // {InputType(Type::STRUCT), ree_filter, StructFilterExec},
        // {InputType(Type::MAP), ree_filter, MapFilterExec},
    };

    VectorKernel kernal_base;
    kernal_base.can_execute_chunkwise = false;
    kernal_base.output_chunked = false;
    auto array_scatter_func = std::make_shared<VectorFunction>(
        "array_scatter", Arity::Binary(), std::move(array_scatter_doc), NULLPTR);
    for (auto&& kernel_data : array_scatter_kernels) {
      kernal_base.signature = KernelSignature::Make(
          {std::move(kernel_data.value_type), std::move(kernel_data.selection_type)},
          OutputType(FirstType));
      kernal_base.exec = kernel_data.exec;
      DCHECK_OK(array_scatter_func->AddKernel(kernal_base));
    }
    DCHECK_OK(registry->AddFunction(std::move(array_scatter_func)));
  }

  // scatter metafunction.
  DCHECK_OK(registry->AddFunction(std::make_shared<ScatterMetaFunction>()));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
