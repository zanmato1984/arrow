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

namespace {

class TrivialSpecialExecutor : public SpecialExecutor {
 public:
  explicit TrivialSpecialExecutor(Expression argument)
      : SpecialExecutor(argument.type()), argument_(std::move(argument)) {}

  Result<Datum> Execute(const ExecBatch& input,
                        ExecContext* exec_context) const override {
    RETURN_NOT_OK(PreExecute(input, exec_context));
    return ExecuteScalarExpression(argument_, input, exec_context);
  }

 protected:
  virtual Status PreExecute(const ExecBatch& input, ExecContext* exec_context) const {
    return Status::OK();
  }

 protected:
  Expression argument_;
};

class UnreachableSpecialExecutor : public TrivialSpecialExecutor {
 public:
  explicit UnreachableSpecialExecutor(Expression argument)
      : TrivialSpecialExecutor(std::move(argument)) {}

 protected:
  Status PreExecute(const ExecBatch& input, ExecContext* exec_context) const override {
    return Status::Invalid("Unreachable");
  }
};

class UnreachableSpecialForm : public SpecialForm {
 public:
  UnreachableSpecialForm() : SpecialForm("unreachable") {}

 protected:
  Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, std::shared_ptr<FunctionOptions> options,
      ExecContext* exec_context) const override {
    DCHECK_EQ(arguments.size(), 1);
    return std::make_unique<UnreachableSpecialExecutor>(arguments[0]);
  }
};

class AssertEmptySelectionSpecialExecutor : public TrivialSpecialExecutor {
 public:
  explicit AssertEmptySelectionSpecialExecutor(Expression argument)
      : TrivialSpecialExecutor(std::move(argument)) {}

 protected:
  Status PreExecute(const ExecBatch& input, ExecContext* exec_context) const override {
    if (input.selection_vector) {
      return Status::Invalid("There shouldn't be a selection vector");
    }
    return Status::OK();
  }
};

class AssertEmptySelectionSpecialForm : public SpecialForm {
 public:
  AssertEmptySelectionSpecialForm() : SpecialForm("assert_selection_empty") {}

 protected:
  Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, std::shared_ptr<FunctionOptions> options,
      ExecContext* exec_context) const override {
    DCHECK_EQ(arguments.size(), 1);
    return std::make_unique<AssertEmptySelectionSpecialExecutor>(arguments[0]);
  }
};

class AssertSelectionEqualSpecialExecutor : public TrivialSpecialExecutor {
 public:
  explicit AssertSelectionEqualSpecialExecutor(Expression argument,
                                               std::shared_ptr<SelectionVector> expected)
      : TrivialSpecialExecutor(std::move(argument)), expected_(std::move(expected)) {
    DCHECK_NE(expected_, nullptr);
  }

 protected:
  Status PreExecute(const ExecBatch& input, ExecContext* exec_context) const override {
    if (input.selection_vector == nullptr) {
      return Status::Invalid("There should be a selection vector");
    }
    if (!SelectionVectorsEqual(expected_, input.selection_vector)) {
      return Status::Invalid("Selection vector does not match expected");
    }
    return Status::OK();
  }

 private:
  bool SelectionVectorsEqual(const std::shared_ptr<SelectionVector>& left,
                             const std::shared_ptr<SelectionVector>& right) const {
    DCHECK_NE(left, nullptr);
    DCHECK_NE(right, nullptr);
    return MakeArray(left->data())->Equals(MakeArray(right->data()));
  }

  std::shared_ptr<SelectionVector> expected_;
};

class AssertSelectionEqualSpecialForm : public SpecialForm {
 public:
  explicit AssertSelectionEqualSpecialForm(std::shared_ptr<SelectionVector> expected)
      : SpecialForm("assert_selection_equal"), expected_(std::move(expected)) {
    DCHECK_NE(expected_, nullptr);
  }

 protected:
  Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, std::shared_ptr<FunctionOptions> options,
      ExecContext* exec_context) const override {
    DCHECK_EQ(arguments.size(), 1);
    return std::make_unique<AssertSelectionEqualSpecialExecutor>(arguments[0], expected_);
  }

 private:
  std::shared_ptr<SelectionVector> expected_;
};

Expression unreachable_special(Expression argument) {
  Expression::Special special;
  special.special_form = std::make_shared<UnreachableSpecialForm>();
  special.arguments.push_back(std::move(argument));
  return Expression(std::move(special));
}

Expression assert_selection_empty_special(Expression argument) {
  Expression::Special special;
  special.special_form = std::make_shared<AssertEmptySelectionSpecialForm>();
  special.arguments.push_back(std::move(argument));
  return Expression(std::move(special));
}

Expression assert_selection_eq_special(Expression argument,
                                       std::shared_ptr<SelectionVector> expected) {
  Expression::Special special;
  special.special_form =
      std::make_shared<AssertSelectionEqualSpecialForm>(std::move(expected));
  special.arguments.push_back(std::move(argument));
  return Expression(std::move(special));
}

const auto kNullLiteral = literal(kNullScalar);
const auto kTrueLiteral = literal(true);
const auto kFalseLiteral = literal(false);

Expression unreachable(Expression argument, const Schema& schm) {
  EXPECT_OK_AND_ASSIGN(auto bound, unreachable_special(std::move(argument)).Bind(schm));
  return bound;
}

Expression assert_selection_empty(Expression argument, const Schema& schm) {
  EXPECT_OK_AND_ASSIGN(auto bound,
                       assert_selection_empty_special(std::move(argument)).Bind(schm));
  return bound;
}

Expression assert_selection_equal(Expression argument,
                                  std::shared_ptr<SelectionVector> selection,
                                  const Schema& schm) {
  EXPECT_OK_AND_ASSIGN(
      auto bound,
      assert_selection_eq_special(std::move(argument), std::move(selection)).Bind(schm));
  return bound;
}

}  // namespace

TEST(ConditionalSpecialExecutor, Basic) {
  ConditionalSpecialExecutor executor({}, utf8());
  EXPECT_EQ(executor.out_type().id(), Type::STRING);
  EXPECT_EQ(executor.options(), nullptr);
}

TEST(ConditionalSpecialExecutor, Execute) {
  auto schm = schema({field("", boolean()), field("", boolean()), field("", utf8()),
                      field("", utf8()), field("", utf8())});

  auto b0 = field_ref(0);
  auto b1 = field_ref(1);
  auto sa = field_ref(2);
  auto sb = field_ref(3);
  auto sc = field_ref(4);
  auto unreachable_sp = [&](Expression argument) -> Expression {
    return unreachable(std::move(argument), *schm);
  };
  auto assert_selection_empty_sp = [&](Expression argument) -> Expression {
    return assert_selection_empty(std::move(argument), *schm);
  };
  auto assert_selection_eq_sp =
      [&](Expression argument, std::shared_ptr<SelectionVector> selection) -> Expression {
    return assert_selection_equal(std::move(argument), std::move(selection), *schm);
  };

  auto batch = ExecBatchFromJSON({boolean(), boolean(), utf8(), utf8(), utf8()},
                                 R"([[true,  true,  "a0", "b0", "c0"],
                                     [false, false, "a1", "b1", "c1"],
                                     [null,  true,  "a2", "b2", "c2"],
                                     [true,  true,  "a3", "b3", "c3"]])");

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kFalseLiteral), unreachable_sp(sa)},
         Branch{assert_selection_empty_sp(kTrueLiteral), assert_selection_empty_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "b1", "b2", "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kFalseLiteral), unreachable_sp(sa)},
         Branch{assert_selection_empty_sp(kNullLiteral), unreachable_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"([null, null, null, null])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kFalseLiteral), unreachable_sp(sa)},
         Branch{assert_selection_empty_sp(b0),
                assert_selection_eq_sp(sb, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "c1", null, "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kFalseLiteral), unreachable_sp(sa)},
         Branch{assert_selection_empty_sp(b1),
                assert_selection_eq_sp(sb, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["b0", "c1", "b2", "b3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kTrueLiteral), assert_selection_empty_sp(sa)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "a1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kNullLiteral), unreachable_sp(sa)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"([null, null, null, null])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b0),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sb, SelectionVectorFromJSON("[1]"))},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "b1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b0),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq_sp(kNullLiteral, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", null, null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b0),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq_sp(b0, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b0),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 3]"))},
         Branch{assert_selection_eq_sp(b1, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", null, "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b1),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sb, SelectionVectorFromJSON("[1]"))},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "b1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b1),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq_sp(kNullLiteral, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{unreachable_sp(kTrueLiteral), unreachable_sp(sc)}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", null, "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b1),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq_sp(b0, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", "a2", "a3"])")),
                      result);
  }

  {
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(b1),
                assert_selection_eq_sp(sa, SelectionVectorFromJSON("[0, 2, 3]"))},
         Branch{assert_selection_eq_sp(b1, SelectionVectorFromJSON("[1]")),
                unreachable_sp(sb)},
         Branch{assert_selection_eq_sp(kTrueLiteral, SelectionVectorFromJSON("[1]")),
                assert_selection_eq_sp(sc, SelectionVectorFromJSON("[1]"))}},
        utf8());
    ASSERT_OK_AND_ASSIGN(auto result, executor.Execute(batch, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(utf8(), R"(["a0", "c1", "a2", "a3"])")),
                      result);
  }
}

TEST(ConditionalSpecialExecutor, ExecuteWithSelection) {
  auto schm = schema({field("a", int32())});

  auto a = field_ref("a");
  auto unreachable_sp = [&](Expression argument) -> Expression {
    return unreachable(std::move(argument), *schm);
  };
  auto assert_selection_empty_sp = [&](Expression argument) -> Expression {
    return assert_selection_empty(std::move(argument), *schm);
  };
  auto assert_selection_eq_sp =
      [&](Expression argument, std::shared_ptr<SelectionVector> selection) -> Expression {
    return assert_selection_equal(std::move(argument), std::move(selection), *schm);
  };

  auto batch = ExecBatchFromJSON({int32()}, R"([[10], [11], [12], [13]])");

  {
    // Empty selection short-circuits to all-null output - the kernel is not invoked.
    const auto selection = SelectionVectorFromJSON("[]");
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = selection;
    ConditionalSpecialExecutor executor(
        {Branch{unreachable_sp(kTrueLiteral), unreachable_sp(a)}}, int32());
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
    AssertDatumsEqual(Datum(ArrayFromJSON(int32(), R"([null, null, null, null])")),
                      result);
  }

  for (const auto& selection :
       {SelectionVectorFromJSON("[0]"), SelectionVectorFromJSON("[1]"),
        SelectionVectorFromJSON("[0, 1, 2]")}) {
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = selection;
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_eq_sp(kTrueLiteral, selection),
                assert_selection_eq_sp(a, selection)}},
        int32());
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
  }

  {
    // Full selection short-circuits to non-selective execution.
    const auto selection = SelectionVectorFromJSON("[0, 1, 2, 3]");
    auto batch_with_selection = batch;
    batch_with_selection.selection_vector = selection;
    ConditionalSpecialExecutor executor(
        {Branch{assert_selection_empty_sp(kTrueLiteral), assert_selection_empty_sp(a)}},
        int32());
    ASSERT_OK_AND_ASSIGN(auto result,
                         executor.Execute(batch_with_selection, default_exec_context()));
  }
}

}  // namespace arrow::compute::internal
