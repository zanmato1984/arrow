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

#include "arrow/compute/special/conditional_special_internal.h"

#include "arrow/array/array_primitive.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/visit_data_inline.h"

namespace arrow::compute::internal {

namespace {

template <typename CheckAllNullFn, typename CheckAllTrueFn, typename CheckAllFalseFn>
std::shared_ptr<const BodyMask> MakeTrivialBodyMask(
    CheckAllNullFn&& check_all_null, CheckAllTrueFn&& check_all_true,
    CheckAllFalseFn&& check_all_false, std::shared_ptr<const BranchMask> branch_mask) {
  if (check_all_null()) {
    return std::make_shared<AllNullBodyMask>();
  } else if (check_all_true()) {
    return std::make_shared<AllPassBodyMask>(std::move(branch_mask));
  } else if (check_all_false()) {
    return std::make_shared<AllFailBodyMask>(std::move(branch_mask));
  }

  return nullptr;
}

}  // namespace

Result<std::shared_ptr<const BodyMask>> BranchMask::MakeBodyMask(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK(datum.type()->id() == Type::BOOL);
  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    auto body_mask = MakeTrivialBodyMask(
        [&]() { return !scalar.is_valid; }, [&]() { return scalar.value; },
        [&]() { return !scalar.value; }, shared_from_this());
    DCHECK_NE(body_mask, nullptr);
    return body_mask;
  }

  if (datum.is_array()) {
    auto boolean_array = datum.array_as<BooleanArray>();
    if (auto body_mask = MakeTrivialBodyMask(
            [&]() { return boolean_array->null_count() == boolean_array->length(); },
            [&]() {
              return boolean_array->null_count() == 0 &&
                     boolean_array->true_count() == boolean_array->length();
            },
            [&]() {
              return boolean_array->null_count() == 0 &&
                     boolean_array->false_count() == boolean_array->length();
            },
            shared_from_this());
        body_mask) {
      return body_mask;
    }
    return MakeBodyMaskFromBitmap(std::move(boolean_array), exec_context);
  }

  DCHECK(datum.is_chunked_array());
  auto chunked_array = datum.chunked_array();
  DCHECK(std::all_of(chunked_array->chunks().begin(), chunked_array->chunks().end(),
                     [](const std::shared_ptr<Array>& chunk) {
                       return chunk->type()->id() == Type::BOOL;
                     }));
  if (auto body_mask = MakeTrivialBodyMask(
          [&]() {
            return std::all_of(chunked_array->chunks().begin(),
                               chunked_array->chunks().end(),
                               [&](const std::shared_ptr<Array>& chunk) {
                                 return chunk->null_count() == chunk->length();
                               });
          },
          [&]() {
            return std::all_of(
                chunked_array->chunks().begin(), chunked_array->chunks().end(),
                [&](const std::shared_ptr<Array>& chunk) {
                  auto boolean_array = checked_cast<BooleanArray*>(chunk.get());
                  return boolean_array->null_count() == 0 &&
                         boolean_array->true_count() == boolean_array->length();
                });
          },
          [&]() {
            return std::all_of(
                chunked_array->chunks().begin(), chunked_array->chunks().end(),
                [&](const std::shared_ptr<Array>& chunk) {
                  auto boolean_array = checked_cast<BooleanArray*>(chunk.get());
                  return boolean_array->null_count() == 0 &&
                         boolean_array->false_count() == boolean_array->length();
                });
          },
          shared_from_this());
      body_mask) {
    return body_mask;
  }
  return MakeBodyMaskFromBitmap(std::move(chunked_array), exec_context);
}

Result<std::shared_ptr<const BranchMask>> BranchMask::FromSelectionVector(
    std::shared_ptr<SelectionVector> selection, int64_t length) {
  DCHECK_NE(selection, nullptr);

#ifndef NDEBUG
  RETURN_NOT_OK(selection->Validate(length));
#endif

  if (selection->length() == 0) {
    return std::make_shared<AllFailBranchMask>();
  }

  if (selection->length() == length) {
    return std::make_shared<AllPassBranchMask>(length);
  }

  return std::make_shared<ConditionalBranchMask>(std::move(selection), length);
}

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder remainder_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(remainder_builder.Reserve(length_));

  ArraySpan span(*bitmap->data());
  int32_t i = 0;
  VisitArraySpanInline<BooleanType>(
      span,
      [&](bool mask) {
        if (mask) {
          body_builder.UnsafeAppend(i);
        } else {
          remainder_builder.UnsafeAppend(i);
        }
        ++i;
      },
      [&]() { ++i; });

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto remainder_arr, remainder_builder.Finish());
  auto body = std::make_shared<SelectionVector>(*body_arr);
  auto remainder = std::make_shared<SelectionVector>(*remainder_arr);
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(remainder),
                                               length_);
}

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder remainder_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(remainder_builder.Reserve(length_));

  int32_t i = 0;
  for (const auto& chunk : bitmap->chunks()) {
    DCHECK_EQ(chunk->type()->id(), Type::BOOL);
    ArraySpan span(*chunk->data());
    VisitArraySpanInline<BooleanType>(
        span,
        [&](bool mask) {
          if (mask) {
            body_builder.UnsafeAppend(i);
          } else {
            remainder_builder.UnsafeAppend(i);
          }
          ++i;
        },
        [&]() { ++i; });
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto remainder_arr, remainder_builder.Finish());
  auto body = std::make_shared<SelectionVector>(*body_arr);
  auto remainder = std::make_shared<SelectionVector>(*remainder_arr);
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(remainder),
                                               length_);
}

Result<std::shared_ptr<const BodyMask>> ConditionalBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder remainder_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(remainder_builder.Reserve(length_));

  for (int64_t i = 0; i < selection_vector_->length(); ++i) {
    auto index = selection_vector_->indices()[i];
    if (!bitmap->IsNull(index)) {
      if (bitmap->Value(index)) {
        body_builder.UnsafeAppend(index);
      } else {
        remainder_builder.UnsafeAppend(index);
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto remainder_arr, remainder_builder.Finish());
  auto body = std::make_shared<SelectionVector>(*body_arr);
  auto remainder = std::make_shared<SelectionVector>(*remainder_arr);
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(remainder),
                                               length_);
}

Result<std::shared_ptr<const BodyMask>> ConditionalBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  std::vector<const BooleanArray*> boolean_arrays(bitmap->num_chunks());
  std::transform(bitmap->chunks().begin(), bitmap->chunks().end(), boolean_arrays.begin(),
                 [](const auto& chunk) {
                   DCHECK_EQ(chunk->type()->id(), Type::BOOL);
                   return checked_cast<const BooleanArray*>(chunk.get());
                 });

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder remainder_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(remainder_builder.Reserve(length_));

  ChunkResolver resolver(bitmap->chunks());
  ChunkLocation location;
  for (int64_t i = 0; i < selection_vector_->length(); ++i) {
    auto index = selection_vector_->indices()[i];
    location = resolver.ResolveWithHint(index, location);
    if (boolean_arrays[location.chunk_index]->IsValid(location.index_in_chunk)) {
      if (boolean_arrays[location.chunk_index]->Value(location.index_in_chunk)) {
        body_builder.UnsafeAppend(index);
      } else {
        remainder_builder.UnsafeAppend(index);
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto remainder_arr, remainder_builder.Finish());
  auto body = std::make_shared<SelectionVector>(body_arr->data());
  auto remainder = std::make_shared<SelectionVector>(remainder_arr->data());
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(remainder),
                                               length_);
}

Result<Datum> ConditionalExec::Execute(const ExecBatch& input,
                                       ExecContext* exec_context) const&& {
  DCHECK(!branches.empty());

  BranchResults results;
  results.Reserve(branches.size());
  ARROW_ASSIGN_OR_RAISE(auto branch_mask, InitBranchMask(input, exec_context));
  for (const auto& branch : branches) {
    if (branch_mask->empty()) {
      // No more rows to evaluate.
      break;
    }
    ARROW_ASSIGN_OR_RAISE(auto body_mask,
                          EvaluateCond(branch_mask, branch.cond, input, exec_context));
    if (body_mask->empty()) {
      // No rows taken for this branch.
      ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask());
      continue;
    }
    ARROW_ASSIGN_OR_RAISE(auto body_result,
                          EvaluateBody(body_mask, branch.body, input, exec_context));
    DCHECK(body_result.type()->Equals(*result_type));
    ARROW_ASSIGN_OR_RAISE(auto selection_vector, body_mask->GetSelectionVector());
    results.Emplace(std::move(body_result), std::move(selection_vector));
    ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask());
  }
  // Should have no remaining rows.
  DCHECK(branch_mask->empty());
  return MultiplexResults(input, results, exec_context);
}

namespace {

/// @brief Results multiplexing is done by invoking a "choose" function to choose values
/// from each branch result based on the selection vectors. This function prepares the
/// choose indices from the branch selection vectors. For the example in
/// ConditionalExec::MultiplexResults's doc string, the choose indices will be:
///   [0, 1, 2, 0, 1, 2, 0]
Result<Datum> ChooseIndices(
    const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors,
    int64_t length, ExecContext* exec_context) {
  const int64_t validity_bytes = bit_util::BytesForBits(length);
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<ResizableBuffer> validity_buf,
      AllocateResizableBuffer(validity_bytes, exec_context->memory_pool()));
  auto validity_data = validity_buf->mutable_data_as<uint8_t>();
  std::memset(validity_data, 0, validity_bytes);

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<ResizableBuffer> indices_buf,
      AllocateResizableBuffer(length * sizeof(int32_t), exec_context->memory_pool()));
  auto indices_data = indices_buf->mutable_data_as<int32_t>();
  for (int32_t index = 0; index < static_cast<int32_t>(selection_vectors.size());
       ++index) {
    DCHECK_NE(selection_vectors[index], nullptr);
    DCHECK_GT(selection_vectors[index]->length(), 0);
    auto row_ids = selection_vectors[index]->indices();
    for (int64_t i = 0; i < selection_vectors[index]->length(); ++i) {
      const int32_t row_id = row_ids[i];
      DCHECK_EQ(bit_util::GetBit(validity_data, row_id), false);
      bit_util::SetBitTo(validity_data, row_id, true);
      indices_data[row_id] = index;
    }
  }

  return ArrayData::Make(int32(), length,
                         {std::move(validity_buf), std::move(indices_buf)});
}

}  // namespace

Result<Datum> ConditionalExec::MultiplexResults(const ExecBatch& input,
                                                const BranchResults& results,
                                                ExecContext* exec_context) const {
  if (results.empty()) {
    // No branches were taken, return an array of nulls.
    return MakeArrayOfNull(result_type.GetSharedPtr(), input.length,
                           exec_context->memory_pool());
  }

  if (results.size() == 1) {
    // Single branch taken.
    const auto& result = results.body_results()[0];
    if (results.selection_vectors()[0] == nullptr) {
      // This branch has no selection vector, then this branch covers all rows, regardless
      // of the existence of outer selection vector, return as is.
      return result;
    }
    if (input.selection_vector == nullptr) {
      // This branch has a selection vector but there is no outer selection vector, then
      // this branch must not be covering all rows - other branches might just have all
      // failed. And we need to go through the choose path to fill in nulls for the rows
      // not covered.
      DCHECK_NE(results.selection_vectors()[0]->length(), input.length);
    } else {
      if (results.selection_vectors()[0]->length() == input.selection_vector->length()) {
        // This branch has a selection vector and there is outer selection vector, and
        // their lengths equal, then this branch must be covering all rows under the outer
        // selection vector, return as is.
        return result;
      }
    }
  }

  std::vector<Datum> choose_args;
  choose_args.reserve(results.size() + 1);
  ARROW_ASSIGN_OR_RAISE(auto indices, ChooseIndices(results.selection_vectors(),
                                                    input.length, exec_context));
  choose_args.emplace_back(std::move(indices));
  choose_args.insert(choose_args.end(), results.body_results().begin(),
                     results.body_results().end());
  return CallFunction("choose", choose_args, exec_context);
}

}  // namespace arrow::compute::internal
