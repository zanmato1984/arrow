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

#include "arrow/compute/test_util_internal.h"

#include "arrow/array/array_base.h"
#include "arrow/array/validate.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/special_form.h"
#include "arrow/datum.h"
#include "arrow/record_batch.h"
#include "arrow/scalar.h"
#include "arrow/table.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/type.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/vector.h"

namespace arrow::compute {

using compute::ExecBatch;
using internal::MapVector;

ExecBatch ExecBatchFromJSON(const std::vector<TypeHolder>& types, std::string_view json) {
  auto fields = ::arrow::internal::MapVector(
      [](const TypeHolder& th) { return field("", th.GetSharedPtr()); }, types);

  ExecBatch batch{*RecordBatchFromJSON(schema(std::move(fields)), json)};

  return batch;
}

ExecBatch ExecBatchFromJSON(const std::vector<TypeHolder>& types,
                            const std::vector<ArgShape>& shapes, std::string_view json) {
  DCHECK_EQ(types.size(), shapes.size());

  ExecBatch batch = ExecBatchFromJSON(types, json);

  auto value_it = batch.values.begin();
  for (ArgShape shape : shapes) {
    if (shape == ArgShape::SCALAR) {
      if (batch.length == 0) {
        *value_it = MakeNullScalar(value_it->type());
      } else {
        *value_it = value_it->make_array()->GetScalar(0).ValueOrDie();
      }
    }
    ++value_it;
  }

  return batch;
}

namespace {

void ValidateOutputImpl(const ArrayData& output) {
  ASSERT_OK(::arrow::internal::ValidateArrayFull(output));
  TestInitialized(output);
}

void ValidateOutputImpl(const ChunkedArray& output) {
  ASSERT_OK(output.ValidateFull());
  for (const auto& chunk : output.chunks()) {
    TestInitialized(*chunk);
  }
}

void ValidateOutputImpl(const RecordBatch& output) {
  ASSERT_OK(output.ValidateFull());
  for (const auto& column : output.column_data()) {
    TestInitialized(*column);
  }
}

void ValidateOutputImpl(const Table& output) {
  ASSERT_OK(output.ValidateFull());
  for (const auto& column : output.columns()) {
    for (const auto& chunk : column->chunks()) {
      TestInitialized(*chunk);
    }
  }
}

void ValidateOutputImpl(const Scalar& output) { ASSERT_OK(output.ValidateFull()); }

}  // namespace

void ValidateOutput(const Datum& output) {
  switch (output.kind()) {
    case Datum::ARRAY:
      ValidateOutputImpl(*output.array());
      break;
    case Datum::CHUNKED_ARRAY:
      ValidateOutputImpl(*output.chunked_array());
      break;
    case Datum::RECORD_BATCH:
      ValidateOutputImpl(*output.record_batch());
      break;
    case Datum::TABLE:
      ValidateOutputImpl(*output.table());
      break;
    case Datum::SCALAR:
      ValidateOutputImpl(*output.scalar());
      break;
    default:
      break;
  }
}

std::shared_ptr<SelectionVector> SelectionVectorFromJSON(const std::string& json) {
  return std::make_shared<SelectionVector>(*ArrayFromJSON(int32(), json));
}

std::shared_ptr<SelectionVector> MakeSelectionVectorTo(int64_t length) {
  auto res = gen::Step<int32_t>()->Generate(length);
  DCHECK_OK(res.status());
  auto arr = res.ValueUnsafe();
  return std::make_shared<SelectionVector>(*arr);
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

}  // namespace

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

}  // namespace arrow::compute
