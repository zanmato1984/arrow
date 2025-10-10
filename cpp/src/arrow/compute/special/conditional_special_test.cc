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

#include <gtest/gtest.h>

#include "arrow/compute/exec_internal.h"
#include "arrow/compute/test_util_internal.h"
#include "arrow/testing/builder.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/checked_cast.h"

namespace arrow::compute::internal {

namespace {

template <typename BranchMaskType>
void CheckBranchMask(const std::shared_ptr<const BranchMask>& branch_mask) {
  auto casted = checked_cast<const BranchMaskType*>(branch_mask.get());
  EXPECT_NE(casted, nullptr);
}

template <typename BodyMaskType>
void CheckMakeBodyMask(const std::shared_ptr<const BranchMask>& branch_mask,
                       const Datum& datum) {
  ASSERT_OK_AND_ASSIGN(auto body_mask,
                       branch_mask->MakeBodyMask(datum, default_exec_context()));
  auto casted = checked_cast<const BodyMaskType*>(body_mask.get());
  EXPECT_NE(casted, nullptr);
}

template <typename BranchMaskType>
void CheckNextBranchMask(const std::shared_ptr<BodyMask>& body_mask) {
  ASSERT_OK_AND_ASSIGN(auto branch_mask, body_mask->NextBranchMask());
  CheckBranchMask<BranchMaskType>(branch_mask);
}

template <typename BodyMaskType>
void CheckMakeBodyMaskAndSelection(
    const std::shared_ptr<const BranchMask>& branch_mask, const Datum& datum,
    const std::shared_ptr<SelectionVector>& expected_body_selection) {
  ASSERT_OK_AND_ASSIGN(auto body_mask,
                       branch_mask->MakeBodyMask(datum, default_exec_context()));
  auto casted = checked_cast<const BodyMaskType*>(body_mask.get());
  EXPECT_NE(casted, nullptr);
  ASSERT_OK_AND_ASSIGN(auto body_selection, body_mask->GetSelectionVector());
  AssertSelectionVectorsEqual(expected_body_selection, body_selection);
}

template <typename BranchMaskType>
void CheckNextBranchMaskAndSelection(
    const std::shared_ptr<BodyMask>& body_mask,
    const std::shared_ptr<SelectionVector>& expected_branch_selection) {
  ASSERT_OK_AND_ASSIGN(auto branch_mask, body_mask->NextBranchMask());
  auto casted = checked_cast<const BranchMaskType*>(branch_mask.get());
  EXPECT_NE(casted, nullptr);
  ASSERT_OK_AND_ASSIGN(auto branch_selection, branch_mask->GetSelectionVector());
  AssertSelectionVectorsEqual(expected_branch_selection, branch_selection);
}

const auto kNullScalar = MakeNullScalar(boolean());
const auto kTrueScalar = MakeScalar(true);
const auto kFalseScalar = MakeScalar(false);

const auto kNullLiteral = literal(kNullScalar);
const auto kTrueLiteral = literal(true);
const auto kFalseLiteral = literal(false);

Expression nullify_selected(Expression expr) {
  return call("nullify_selected", {std::move(expr)});
}

}  // namespace

TEST(BranchMask, FromSelectionVector) {
  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[]"), 0));
    CheckBranchMask<AllFailBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[]"), 42));
    CheckBranchMask<AllFailBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[0]"), 1));
    CheckBranchMask<AllPassBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask,
                         BranchMask::FromSelectionVector(MakeSelectionVectorTo(42), 42));
    CheckBranchMask<AllPassBranchMask>(branch_mask);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto branch_mask, BranchMask::FromSelectionVector(
                                               SelectionVectorFromJSON("[0]"), 42));
    CheckBranchMask<ConditionalBranchMask>(branch_mask);
  }
}

TEST(AllPassBranchMask, Emptiness) {
  for (auto length : {0, 42}) {
    auto branch_mask = std::make_shared<AllPassBranchMask>(length);
    EXPECT_EQ(branch_mask->empty(), length == 0);
  }
}

TEST(AllPassBranchMask, GetSelectionVector) {
  for (auto length : {0, 42}) {
    auto branch_mask = std::make_shared<AllPassBranchMask>(length);
    EXPECT_EQ(branch_mask->GetSelectionVector(), nullptr);
  }
}

TEST(AllPassBranchMask, MakeBodyMaskTrival) {
  const int64_t length = 42;

  CheckMakeBodyMask<AllNullBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                     Datum(kNullScalar));
  CheckMakeBodyMask<AllPassBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                     Datum(kTrueScalar));
  CheckMakeBodyMask<AllFailBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                     Datum(kFalseScalar));

  {
    ASSERT_OK_AND_ASSIGN(auto boolean_array,
                         gen::Constant(kNullScalar)->Generate(length));
    CheckMakeBodyMask<AllNullBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                       Datum(boolean_array));

    auto boolean_chunked_array =
        std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
    CheckMakeBodyMask<AllNullBodyMask>(std::make_shared<AllPassBranchMask>(length * 2),
                                       Datum(boolean_chunked_array));
  }

  {
    ASSERT_OK_AND_ASSIGN(auto boolean_array,
                         gen::Constant(kTrueScalar)->Generate(length));
    CheckMakeBodyMaskAndSelection<AllPassBodyMask>(
        std::make_shared<AllPassBranchMask>(length), Datum(boolean_array), nullptr);

    auto boolean_chunked_array =
        std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
    CheckMakeBodyMaskAndSelection<AllPassBodyMask>(
        std::make_shared<AllPassBranchMask>(length * 2), Datum(boolean_chunked_array),
        nullptr);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto boolean_array,
                         gen::Constant(kFalseScalar)->Generate(length));
    CheckMakeBodyMask<AllFailBodyMask>(std::make_shared<AllPassBranchMask>(length),
                                       Datum(boolean_array));

    auto boolean_chunked_array =
        std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
    CheckMakeBodyMask<AllFailBodyMask>(std::make_shared<AllPassBranchMask>(length * 2),
                                       Datum(boolean_chunked_array));
  }
}

TEST(AllPassBranchMask, MakeBodyMask) {
  {
    auto boolean_array = ArrayFromJSON(boolean(), "[true, false, null, true]");
    CheckMakeBodyMaskAndSelection<ConditionalBodyMask>(
        std::make_shared<AllPassBranchMask>(boolean_array->length()),
        Datum(boolean_array), SelectionVectorFromJSON("[0, 3]"));
  }

  {
    auto boolean_chunked_array = ChunkedArrayFromJSON(
        boolean(), {"[true, false, null, true]", "[true, false, null, true]"});
    CheckMakeBodyMaskAndSelection<ConditionalBodyMask>(
        std::make_shared<AllPassBranchMask>(boolean_chunked_array->length()),
        Datum(boolean_chunked_array), SelectionVectorFromJSON("[0, 3, 4, 7]"));
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

TEST(ConditionalBranchMask, GetSelectionVector) {
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(42)}) {
    auto branch_mask = std::make_shared<ConditionalBranchMask>(selection, /*length=*/42);
    ASSERT_OK_AND_ASSIGN(auto got, branch_mask->GetSelectionVector());
    AssertSelectionVectorsEqual(selection, got);
  }
}

TEST(ConditionalBranchMask, MakeBodyMaskTrivial) {
  const int64_t length = 42;

  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(length)}) {
    CheckMakeBodyMask<AllPassBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), Datum(kTrueScalar));
    CheckMakeBodyMask<AllFailBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), Datum(kFalseScalar));
    CheckMakeBodyMask<AllNullBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, length), Datum(kNullScalar));

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kNullScalar)->Generate(length));
      CheckMakeBodyMask<AllNullBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length),
          Datum(boolean_array));

      auto boolean_chunked_array =
          std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
      CheckMakeBodyMask<AllNullBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2),
          Datum(boolean_chunked_array));
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kTrueScalar)->Generate(length));
      CheckMakeBodyMaskAndSelection<AllPassBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length),
          Datum(boolean_array), selection);

      auto boolean_chunked_array =
          std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
      CheckMakeBodyMaskAndSelection<AllPassBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2),
          Datum(boolean_chunked_array), selection);
    }

    {
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(kFalseScalar)->Generate(length));
      CheckMakeBodyMask<AllFailBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length),
          Datum(boolean_array));

      auto boolean_chunked_array =
          std::make_shared<ChunkedArray>(ArrayVector{boolean_array, boolean_array});
      CheckMakeBodyMask<AllFailBodyMask>(
          std::make_shared<ConditionalBranchMask>(selection, length * 2),
          Datum(boolean_chunked_array));
    }
  }
}

TEST(ConditionalBranchMask, MakeBodyMask) {
  {
    auto selection = SelectionVectorFromJSON("[2, 3]");
    auto boolean_array = ArrayFromJSON(boolean(), "[true, false, null, true]");
    CheckMakeBodyMaskAndSelection<ConditionalBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection, boolean_array->length()),
        Datum(boolean_array), SelectionVectorFromJSON("[3]"));
  }

  {
    auto selection = SelectionVectorFromJSON("[2, 4]");
    auto boolean_chunked_array = ChunkedArrayFromJSON(
        boolean(), {"[true, false, null, true]", "[true, false, null, true]"});
    CheckMakeBodyMaskAndSelection<ConditionalBodyMask>(
        std::make_shared<ConditionalBranchMask>(selection,
                                                boolean_chunked_array->length()),
        Datum(boolean_chunked_array), SelectionVectorFromJSON("[4]"));
  }
}

TEST(AllNullBodyMask, Emptiness) {
  auto body_mask = std::make_shared<AllNullBodyMask>();
  EXPECT_TRUE(body_mask->empty());
}

TEST(AllNullBodyMask, NextBranchMask) {
  auto body_mask = std::make_shared<AllNullBodyMask>();
  ASSERT_OK_AND_ASSIGN(auto next_branch_mask, body_mask->NextBranchMask());
  CheckBranchMask<AllFailBranchMask>(next_branch_mask);
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

TEST(AllPassBodyMask, GetSelectionVector) {
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
    ASSERT_OK_AND_ASSIGN(auto body_selection, body_mask->GetSelectionVector());
    ASSERT_OK_AND_ASSIGN(auto branch_selection, branch_mask->GetSelectionVector());
    AssertSelectionVectorsEqual(branch_selection, body_selection);
  }
}

TEST(AllPassBodyMask, NextBranchMask) {
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
    CheckNextBranchMask<AllFailBranchMask>(body_mask);
  }
}

TEST(AllFailBodyMask, Emptiness) {
  for (const auto& branch_masks : std::vector<std::shared_ptr<BranchMask>>{
           std::make_shared<AllPassBranchMask>(0),
           std::make_shared<AllPassBranchMask>(42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), 0),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), 42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0]"), 42),
           std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0, 41]"),
                                                   42),
           std::make_shared<ConditionalBranchMask>(MakeSelectionVectorTo(42), 42)}) {
    auto body_mask = std::make_shared<AllFailBodyMask>(branch_masks);
    EXPECT_TRUE(body_mask->empty());
  }
}

TEST(AllFailBodyMask, NextBranchMask) {
  const int64_t length = 42;

  {
    auto branch_mask = std::make_shared<AllPassBranchMask>(length);
    ASSERT_OK_AND_ASSIGN(auto branch_selection, branch_mask->GetSelectionVector());
    auto body_mask = std::make_shared<AllFailBodyMask>(branch_mask);
    CheckNextBranchMaskAndSelection<AllPassBranchMask>(body_mask, branch_selection);
  }

  for (const auto& branch_mask :
       {std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[]"), length),
        std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0]"), length),
        std::make_shared<ConditionalBranchMask>(SelectionVectorFromJSON("[0, 41]"),
                                                length),
        std::make_shared<ConditionalBranchMask>(MakeSelectionVectorTo(length), length)}) {
    ASSERT_OK_AND_ASSIGN(auto branch_selection, branch_mask->GetSelectionVector());
    auto body_mask = std::make_shared<AllFailBodyMask>(branch_mask);
    CheckNextBranchMaskAndSelection<ConditionalBranchMask>(body_mask, branch_selection);
  }
}

TEST(ConditionalBodyMask, Emptiness) {
  const int64_t length = 42;
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(length)}) {
    auto body_mask = std::make_shared<ConditionalBodyMask>(
        selection, SelectionVectorFromJSON("[1]"), length);
    EXPECT_EQ(body_mask->empty(), selection->length() == 0);
  }
}

TEST(ConditionalBodyMask, GetSelectionVector) {
  const int64_t length = 42;
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]"), MakeSelectionVectorTo(length)}) {
    auto body_mask = std::make_shared<ConditionalBodyMask>(
        selection, SelectionVectorFromJSON("[1]"), length);
    ASSERT_OK_AND_ASSIGN(auto got, body_mask->GetSelectionVector());
    AssertSelectionVectorsEqual(selection, got);
  }
}

TEST(ConditionalBodyMask, NextBranchMask) {
  const int64_t length = 42;

  {
    auto body_mask = std::make_shared<ConditionalBodyMask>(
        SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[]"), length);
    CheckNextBranchMask<AllFailBranchMask>(body_mask);
  }

  for (const auto& remainder :
       {SelectionVectorFromJSON("[0]"), SelectionVectorFromJSON("[0, 41]")}) {
    auto body_mask = std::make_shared<ConditionalBodyMask>(SelectionVectorFromJSON("[]"),
                                                           remainder, length);
    CheckNextBranchMaskAndSelection<ConditionalBranchMask>(body_mask, remainder);
  }

  {
    auto remainder = MakeSelectionVectorTo(length);
    auto body_mask = std::make_shared<ConditionalBodyMask>(SelectionVectorFromJSON("[]"),
                                                           remainder, length);
    CheckNextBranchMaskAndSelection<AllPassBranchMask>(body_mask, nullptr);
  }
}

TEST(ConditionalSpecialExecutor, Basic) {
  ConditionalSpecialExecutor executor({}, utf8());
  EXPECT_EQ(executor.out_type().id(), Type::STRING);
  EXPECT_EQ(executor.options(), nullptr);
}

TEST(ConditionalSpecialExecutor, Execute) {
  auto schm = schema({field("", boolean()), field("", boolean()), field("", utf8()),
                      field("", utf8()), field("", utf8())});

  ASSERT_OK_AND_ASSIGN(auto b0, field_ref(0).Bind(*schm));
  ASSERT_OK_AND_ASSIGN(auto b1, field_ref(1).Bind(*schm));
  ASSERT_OK_AND_ASSIGN(auto sa, field_ref(2).Bind(*schm));
  ASSERT_OK_AND_ASSIGN(auto sb, field_ref(3).Bind(*schm));
  ASSERT_OK_AND_ASSIGN(auto sc, field_ref(4).Bind(*schm));
  auto unreachable = [&](Expression argument) -> Expression {
    EXPECT_OK_AND_ASSIGN(auto bound,
                         unreachable_special(std::move(argument)).Bind(*schm));
    return bound;
  };
  auto assert_empty_selection = [&](Expression argument) -> Expression {
    EXPECT_OK_AND_ASSIGN(auto bound,
                         assert_empty_selection_special(std::move(argument)).Bind(*schm));
    return bound;
  };
  auto assert_selection_eq =
      [&](Expression argument, std::shared_ptr<SelectionVector> selection) -> Expression {
    EXPECT_OK_AND_ASSIGN(
        auto bound, assert_selection_eq_special(std::move(argument), std::move(selection))
                        .Bind(*schm));
    return bound;
  };

  auto batch = ExecBatchFromJSON({boolean(), boolean(), utf8(), utf8(), utf8()},
                                 R"([[true,  true,  "a0", "b0", "c0"],
                                     [false, false, "a1", "b1", "c1"],
                                     [null,  true,  "a2", "b2", "c2"],
                                     [true,  true,  "a3", "b3", "c3"]])");

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kFalseLiteral), unreachable(sa)},
         Branch{assert_empty_selection(kTrueLiteral), assert_empty_selection(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "b1", "b2", "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kFalseLiteral), unreachable(sa)},
         Branch{assert_empty_selection(kNullLiteral), unreachable(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"([null, null, null, null])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kFalseLiteral), unreachable(sa)},
         Branch{assert_empty_selection(b0),
                assert_selection_eq(sb, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "c1", null, "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kFalseLiteral), unreachable(sa)},
         Branch{assert_empty_selection(b1),
                assert_selection_eq(sb, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "c1", "b2", "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kTrueLiteral), assert_empty_selection(sa)},
         Branch{unreachable(kTrueLiteral), unreachable(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "a1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(kNullLiteral), unreachable(sa)},
         Branch{unreachable(kTrueLiteral), unreachable(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"([null, null, null, null])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b0),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sb, SelectionVectorFromJSON("[1]"))},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "b1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b0),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq(kNullLiteral, SelectionVectorFromJSON("[1]")),
                unreachable(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", null, null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b0),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq(b0, SelectionVectorFromJSON("[1]")), unreachable(sb)},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b0),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq(b1, SelectionVectorFromJSON("[1]")), unreachable(sb)},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b1),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sb, SelectionVectorFromJSON("[1]"))},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "b1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b1),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq(kNullLiteral, SelectionVectorFromJSON("[1]")),
                unreachable(sb)},
         Branch{unreachable(kTrueLiteral), unreachable(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", null, "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b1),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq(b0, SelectionVectorFromJSON("[1]")), unreachable(sb)},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_empty_selection(b1),
                assert_selection_eq(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq(b1, SelectionVectorFromJSON("[1]")), unreachable(sb)},
         Branch{assert_selection_eq(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", "a2", "a3"])")),
                      result);
  }
}

class ConditionalSpecialExecutorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ASSERT_OK(RegisterNullifySelectedFunction()); }

 protected:
  static Status NullifySelectedExec(KernelContext* ctx, const ExecSpan& span,
                                    ExecResult* out) {
    return Status::NotImplemented("Not implemented");
  }

  static Status NullifySelectedSelectiveExec(KernelContext* ctx, const ExecSpan& batch,
                                             const SelectionVectorSpan& selection,
                                             ExecResult* out) {
    DCHECK_EQ(1, batch.num_values());
    int value_size = batch[0].type()->byte_width();
    const ArraySpan& arg0 = batch[0].array;
    ArraySpan* out_arr = out->array_span_mutable();
    uint8_t* dst_validity = out_arr->buffers[0].data;
    int64_t dst_validity_offset = out_arr->offset;
    uint8_t* dst = out_arr->buffers[1].data + out_arr->offset * value_size;
    uint8_t* src_validity = arg0.buffers[0].data;
    int64_t src_validity_offset = arg0.offset;
    const uint8_t* src = arg0.buffers[1].data + arg0.offset * value_size;
    bit_util::SetBitmap(dst_validity, dst_validity_offset, batch.length);
    std::memcpy(dst, src, batch.length * value_size);
    compute::detail::VisitSelectionVectorSpanInline(selection, [&](int64_t i) {
      DCHECK(!src_validity || bit_util::GetBit(src_validity, src_validity_offset + i));
      bit_util::ClearBit(dst_validity, dst_validity_offset + i);
    });
    return Status::OK();
  }

  static Status RegisterNullifySelectedFunction() {
    auto registry = GetFunctionRegistry();

    auto func = std::make_shared<ScalarFunction>(std::move("nullify_selected"),
                                                 Arity::Unary(), FunctionDoc::Empty());
    ScalarKernel kernel({InputType(int32())}, internal::FirstType, NullifySelectedExec,
                        NullifySelectedSelectiveExec);
    kernel.can_write_into_slices = false;
    kernel.null_handling = NullHandling::COMPUTED_PREALLOCATE;
    kernel.mem_allocation = MemAllocation::PREALLOCATE;
    RETURN_NOT_OK(func->AddKernel(kernel));
    RETURN_NOT_OK(registry->AddFunction(std::move(func)));

    return Status::OK();
  }
};

TEST_F(ConditionalSpecialExecutorTest, ExecuteWithSelection) {
  auto schm = schema({field("a", int32())});
  ASSERT_OK_AND_ASSIGN(auto body, nullify_selected(field_ref("a")).Bind(*schm));
  ConditionalSpecialExecutor executor({Branch{kTrueLiteral, body}}, int32());
  auto batch = ExecBatchFromJSON({int32()}, R"([[10], [11], [12], [13]])");

  {
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = SelectionVectorFromJSON("[]");
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
    // Empty selection short-circuits to all-null output - the kernel is not invoked.
    AssertDatumsEqual(Datum(ArrayFromJSON(int32(), R"([null, null, null, null])")),
                      result);
  }

  {
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = SelectionVectorFromJSON("[0]");
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(int32(), R"([null, 11, 12, 13])")), result);
  }

  {
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = SelectionVectorFromJSON("[0, 1, 2]");
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(int32(), R"([null, null, null, 13])")), result);
  }

  {
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = SelectionVectorFromJSON("[0, 1, 2, 3]");
    // Full selection short-circuits to non-selective execution.
    ASSERT_RAISES(NotImplemented,
                  executor.Execute(batch_with_selection, default_exec_context()));
  }
}

}  // namespace arrow::compute::internal
