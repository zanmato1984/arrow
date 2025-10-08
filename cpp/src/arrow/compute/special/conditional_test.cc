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

#include "arrow/compute/special/conditional_internal.h"
#include "arrow/compute/test_util_internal.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/checked_cast.h"

namespace arrow::compute::internal {

namespace {

template <typename BranchMaskType>
void CheckBranchMaskType(const std::shared_ptr<const BranchMask>& branch_mask) {
  auto casted = checked_cast<const BranchMaskType*>(branch_mask.get());
  EXPECT_NE(casted, nullptr);
}

template <typename BodyMaskType>
void CheckBodyMaskType(const std::shared_ptr<BranchMask>& branch_mask,
                       const Expression& expr, const ExecBatch& batch) {
  ASSERT_OK_AND_ASSIGN(auto body_mask,
                       branch_mask->EvaluateCond(expr, batch, default_exec_context()));
  auto casted = checked_cast<const BodyMaskType*>(body_mask.get());
  EXPECT_NE(casted, nullptr);
}

void AssertSelectionVectorsEqual(const std::shared_ptr<SelectionVector>& expected,
                                 const std::shared_ptr<SelectionVector>& actual) {
  if (expected == nullptr) {
    EXPECT_EQ(actual, nullptr);
    return;
  }

  if (actual == nullptr) {
    EXPECT_EQ(expected, nullptr);
    return;
  }

  ASSERT_EQ(expected->length(), actual->length());
  AssertArraysEqual(*MakeArray(expected->data()), *MakeArray(actual->data()));
}

template <typename BodyMaskType>
void CheckBodyMaskSelectionVector(const std::shared_ptr<BranchMask>& branch_mask,
                                  const Expression& expr, const ExecBatch& batch,
                                  const std::shared_ptr<SelectionVector>& expected) {
  ASSERT_OK_AND_ASSIGN(auto body_mask,
                       branch_mask->EvaluateCond(expr, batch, default_exec_context()));
  auto casted = checked_cast<const BodyMaskType*>(body_mask.get());
  EXPECT_NE(casted, nullptr);
  ASSERT_OK_AND_ASSIGN(auto selection, body_mask->GetSelectionVector());
  AssertSelectionVectorsEqual(expected, selection);
}

const auto kTrueScalar = MakeScalar(true);
const auto kFalseScalar = MakeScalar(false);
const auto kNullScalar = MakeNullScalar(boolean());

const auto kTrueLiteral = literal(true);
const auto kFalseLiteral = literal(false);
const auto kNullLiteral = literal(kNullScalar);

}  // namespace

TEST(BranchMask, FromSelectionVector) {
  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[]"), 0));
    CheckBranchMaskType<AllFailBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[]"), 42));
    CheckBranchMaskType<AllFailBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[0]"), 1));
    CheckBranchMaskType<AllPassBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask,
                         BranchMask::FromSelectionVector(MakeSelectionVectorTo(42), 42));
    CheckBranchMaskType<AllPassBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[0]"), 42));
    CheckBranchMaskType<ConditionalBranchMask>(branch_mask);
  }
}

TEST(AllPassBranchMask, Emptiness) {
  for (auto length : {0, 42}) {
    auto branch_mask = std::make_shared<AllPassBranchMask>(length);
    EXPECT_EQ(branch_mask->empty(), length == 0);
  }
}

TEST(AllPassBranchMask, EvaluateCond) {
  ASSERT_OK_AND_ASSIGN(auto f, field_ref(0).Bind(Schema({field("", boolean())})));

  for (auto length : {0, 42}) {
    CheckBodyMaskSelectionVector<AllPassBodyMask>(
        std::make_shared<AllPassBranchMask>(length), kTrueLiteral, ExecBatch({}, length),
        nullptr);
    CheckBodyMaskSelectionVector<AllPassBodyMask>(
        std::make_shared<AllPassBranchMask>(length), f, ExecBatch({kTrueScalar}, length),
        nullptr);

    CheckBodyMaskType<AllFailBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                       kFalseLiteral, ExecBatch({}, length));
    CheckBodyMaskType<AllFailBodyMask>(std::make_shared<AllPassBranchMask>(length), f,
                                       ExecBatch({kFalseScalar}, length));

    CheckBodyMaskType<AllNullBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                       kNullLiteral, ExecBatch({}, length));
    CheckBodyMaskType<AllNullBodyMask>(std::make_shared<AllPassBranchMask>(length), f,
                                       ExecBatch({kNullScalar}, length));

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kTrueScalar)->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length), f,
          ExecBatch({boolean_array}, length), MakeSelectionVectorTo(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          MakeSelectionVectorTo(length * 2));
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kFalseScalar)->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length), f,
          ExecBatch({boolean_array}, length), MakeSelectionVectorTo(0));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          SelectionVectorFromJSON("[]"));
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(MakeNullScalar(boolean()))->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length), f,
          ExecBatch({boolean_array}, length), MakeSelectionVectorTo(0));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<AllPassBranchMask>(length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          SelectionVectorFromJSON("[]"));
    }
  }

  {
    auto boolean_array = ArrayFromJSON(boolean(), "[true, false, null, true]");
    CheckBodyMaskSelectionVector<ConditionalBodyMask>(
        std::make_shared<AllPassBranchMask>(boolean_array->length()), f,
        ExecBatch({boolean_array}, boolean_array->length()),
        SelectionVectorFromJSON("[0, 3]"));
  }

  {
    auto boolean_chunked_array = ChunkedArrayFromJSON(
        boolean(), {"[true, false, null, true]", "[true, false, null, true]"});
    CheckBodyMaskSelectionVector<ConditionalBodyMask>(
        std::make_shared<AllPassBranchMask>(boolean_chunked_array->length()), f,
        ExecBatch({boolean_chunked_array}, boolean_chunked_array->length()),
        SelectionVectorFromJSON("[0, 3, 4, 7]"));
  }
}

TEST(AllFailBranchMask, Emptiness) {
  auto branch_mask = std::make_shared<AllFailBranchMask>();
  EXPECT_TRUE(branch_mask->empty());
}

TEST(ConditionalBranchMask, Emptiness) {
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(42)}) {
    auto branch_mask = std::make_shared<ConditionalBranchMask>(selection, /*length=*/42);
    EXPECT_EQ(branch_mask->empty(), selection->length() == 0);
  }
}

TEST(ConditionalBranchMask, EvaluateCond) {
  ASSERT_OK_AND_ASSIGN(auto f, field_ref(0).Bind(Schema({field("", boolean())})));

  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(42)}) {
    const int64_t length = 42;

    CheckBodyMaskSelectionVector<AllPassBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), kTrueLiteral,
        ExecBatch({}, length), selection);
    CheckBodyMaskSelectionVector<AllPassBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), f,
        ExecBatch({kTrueScalar}, length), selection);

    CheckBodyMaskType<AllFailBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), kFalseLiteral,
        ExecBatch({}, length));
    CheckBodyMaskType<AllFailBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), f,
        ExecBatch({kFalseScalar}, length));

    CheckBodyMaskType<AllNullBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), kNullLiteral,
        ExecBatch({}, length));
    CheckBodyMaskType<AllNullBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), f,
        ExecBatch({kNullScalar}, length));

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kTrueScalar)->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length), f,
          ExecBatch({boolean_array}, length), selection);
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          selection);
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kFalseScalar)->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length), f,
          ExecBatch({boolean_array}, length), SelectionVectorFromJSON("[]"));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          SelectionVectorFromJSON("[]"));
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kNullScalar)->Generate(length));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length), f,
          ExecBatch({boolean_array}, length), SelectionVectorFromJSON("[]"));
      CheckBodyMaskSelectionVector<ConditionalBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2), f,
          ExecBatch(
              {std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array})},
              length * 2),
          SelectionVectorFromJSON("[]"));
    }
  }

  {
    auto selection = SelectionVectorFromJSON("[2, 3]");
    auto boolean_array = ArrayFromJSON(boolean(), "[true, false, null, true]");
    CheckBodyMaskSelectionVector<ConditionalBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, boolean_array->length()), f,
        ExecBatch({boolean_array}, boolean_array->length()),
        SelectionVectorFromJSON("[3]"));
  }

  {
    auto selection = SelectionVectorFromJSON("[2, 4]");
    auto boolean_chunked_array = ChunkedArrayFromJSON(
        boolean(), {"[true, false, null, true]", "[true, false, null, true]"});
    CheckBodyMaskSelectionVector<ConditionalBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection,
                                                boolean_chunked_array->length()),
        f, ExecBatch({boolean_chunked_array}, boolean_chunked_array->length()),
        SelectionVectorFromJSON("[4]"));
  }
}

TEST(AllNullBodyMask, Emptiness) {
  auto body_mask = std::make_shared<AllNullBodyMask>();
  EXPECT_TRUE(body_mask->empty());
}

TEST(AllNullBodyMask, NextBranchMask) {
  auto body_mask = std::make_shared<AllNullBodyMask>();
  ASSERT_OK_AND_ASSIGN(auto next_branch_mask, body_mask->NextBranchMask());
  CheckBranchMaskType<AllFailBranchMask>(next_branch_mask);
}

TEST(AllPassBodyMask, Emptiness) {
  for (const auto& branch_masks : std::vector<std::shared_ptr<BranchMask>>{
           std::make_shared<AllPassBranchMask>(0),
           std::make_shared<AllPassBranchMask>(42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), 0),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), 42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0]"), 42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0, 41]"),
                                                   42),
           std::make_shared<ConditionalBranchMask>(MakeSelectionVectorTo(42), 42)}) {
    auto body_mask = std::make_shared<AllPassBodyMask>(branch_masks);
    EXPECT_EQ(body_mask->empty(), branch_masks->empty());
  }
}

template <typename OnSelectedFn, typename OnNonSelectedFn>
void VisitIndicesWithSelection(int64_t length, const SelectionVectorSpan& selection,
                               OnSelectedFn&& on_selected,
                               OnNonSelectedFn&& on_non_selected) {
  int64_t selected = 0;
  for (int64_t i = 0; i < length; ++i) {
    if (selected < selection.length() && i == selection[selected]) {
      on_selected(i);
      ++selected;
    } else {
      on_non_selected(i);
    }
  }
}

TEST(AllPassBodyMask, EvaluateBody) {
  const int64_t length = 42;
  for (const auto& branch_mask : std::vector<std::shared_ptr<BranchMask>>{
           std::make_shared<AllPassBranchMask>(length),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), length),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0]"),
                                                   length),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0, 41]"),
                                                   length),
           std::make_shared<ConditionalBranchMask>(MakeSelectionVectorTo(length),
                                                   length)}) {
    auto body_mask = std::make_shared<AllPassBodyMask>(branch_mask);

    // for (const auto& expr : {literal(true), literal(false), literal(kNullScalar)})
    // {
    //   CheckBodyMaskType<AllPassBodyMask>(body_mask, expr, ExecBatch({}, length));
    // }
  }
}

}  // namespace arrow::compute::internal
