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

#include "arrow/compute/special_form.h"

#include <gtest/gtest.h>

#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::compute {

TEST(IfElseSpecialForm, Basic) {
  {
    ARROW_SCOPED_TRACE("if (b != 0) then a / b else b");
    auto cond = call("not_equal", {field_ref("b"), literal(0)});
    auto if_true = call("divide", {field_ref("a"), field_ref("b")});
    auto if_false = field_ref("b");
    auto schema = arrow::schema({field("a", int32()), field("b", int32())});
    auto rb = RecordBatchFromJSON(schema, R"([
        [1, 1],
        [2, 1],
        [3, 0],
        [4, 1],
        [5, 1]
      ])");
    auto input = ExecBatch(*rb);
    auto if_else_sp = if_else_special(cond, if_true, if_false);
    {
      auto expected = ArrayFromJSON(int32(), "[1, 2, 0, 4, 5]");
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
      ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
      AssertDatumsEqual(*expected, result);
    }
    {
      ARROW_SCOPED_TRACE("(if (b != 0) then a / b else b) + 1");
      auto plus_one = call("add", {if_else_sp, literal(1)});
      {
        auto expected = ArrayFromJSON(int32(), "[2, 3, 1, 5, 6]");
        ASSERT_OK_AND_ASSIGN(auto bound, plus_one.Bind(*schema));
        ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
        AssertDatumsEqual(*expected, result);
      }
      {
        ARROW_SCOPED_TRACE(
            "if ((if (b != 0) then a / b else b) + 1 != 1) then a / b else b");
        auto cond = call("not_equal", {plus_one, literal(1)});
        auto if_true = call("divide", {field_ref("a"), field_ref("b")});
        auto if_false = field_ref("b");
        auto if_else_sp = if_else_special(cond, if_true, if_false);
        auto expected = ArrayFromJSON(int32(), "[1, 2, 0, 4, 5]");
        ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
        ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
        AssertDatumsEqual(*expected, result);
      }
    }
  }
  {
    ARROW_SCOPED_TRACE("if (b != 0) then a else a");
    auto cond = call("not_equal", {field_ref("b"), literal(0)});
    auto if_true = field_ref("a");
    auto if_false = field_ref("a");
    auto schema = arrow::schema({field("a", int32()), field("b", int32())});
    auto rb = RecordBatchFromJSON(schema, R"([
        [1, 1],
        [2, 1],
        [3, 0],
        [4, 1],
        [5, 1]
      ])");
    auto input = ExecBatch(*rb);
    auto if_else_sp = if_else_special(cond, if_true, if_false);
    auto expected = ArrayFromJSON(int32(), "[1, 2, 3, 4, 5]");
    ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
    ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
    AssertDatumsEqual(*expected, result);
  }
}

namespace {

void AssertIfElseEqual(const Datum& expected, Expression cond, Expression if_true,
                       Expression if_false, const std::shared_ptr<Schema>& schema,
                       const ExecBatch& input) {
  auto if_else_sp = if_else_special(cond, if_true, if_false);
  ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
  AssertDatumsEqual(expected, result);
}

void AssertIfElseEqualWithExpr(Expression cond, Expression if_true, Expression if_false,
                               const std::shared_ptr<Schema>& schema,
                               const ExecBatch& input) {
  auto if_else = call("if_else", {cond, if_true, if_false});
  auto if_else_sp = if_else_special(cond, if_true, if_false);
  ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, input));
  ASSERT_OK_AND_ASSIGN(auto bound_sp, if_else_sp.Bind(*schema));
  ASSERT_OK_AND_ASSIGN(auto result_sp, ExecuteScalarExpression(bound_sp, input));
  AssertDatumsEqual(result, result_sp);
}

}  // namespace

TEST(IfElseSpecialForm, Shortcuts) {
  {
    ARROW_SCOPED_TRACE("if (null) then 1 else 0");
    AssertIfElseEqual(MakeNullScalar(int32()), literal(MakeNullScalar(boolean())),
                      literal(1), literal(0), arrow::schema({field("", int32())}),
                      ExecBatch({*ArrayFromJSON(int32(), "[]")}, 0));
  }
  {
    ARROW_SCOPED_TRACE("if (true) then 1 else 0");
    AssertIfElseEqual(MakeScalar(1), literal(true), literal(1), literal(0),
                      arrow::schema({field("", int32())}),
                      ExecBatch({*ArrayFromJSON(int32(), "[]")}, 0));
  }
  {
    ARROW_SCOPED_TRACE("if (false) then 1 else 0");
    AssertIfElseEqual(MakeScalar(0), literal(false), literal(1), literal(0),
                      arrow::schema({field("", int32())}),
                      ExecBatch({*ArrayFromJSON(int32(), "[]")}, 0));
  }
  {
    auto schema = arrow::schema({field("a", int32()), field("b", int32())});
    std::vector<ExecBatch> batches = {
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [1, 0],
            [1, 0],
            [1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([])")),
    };
    for (const auto& input : batches) {
      {
        ARROW_SCOPED_TRACE("if (null) then a else b");
        AssertIfElseEqual(MakeNullScalar(int32()), literal(MakeNullScalar(boolean())),
                          field_ref("a"), field_ref("b"), schema, input);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then 0 else b");
        AssertIfElseEqual(MakeScalar(0), literal(true), literal(0), field_ref("b"),
                          schema, input);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then a else b");
        AssertIfElseEqualWithExpr(literal(true), field_ref("a"), field_ref("b"), schema,
                                  input);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else 1");
        AssertIfElseEqual(MakeScalar(1), literal(false), field_ref("a"), literal(1),
                          schema, input);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else b");
        AssertIfElseEqualWithExpr(literal(false), field_ref("a"), field_ref("b"), schema,
                                  input);
      }
    }
  }
  {
    auto schema =
        arrow::schema({field("a", boolean()), field("b", int32()), field("c", int32())});
    std::vector<ExecBatch> batches = {
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [null, 1, 0],
            [null, 1, 0],
            [null, 1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [true, 1, 0],
            [true, 1, 0],
            [true, 1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [false, 1, 0],
            [false, 1, 0],
            [false, 1, 0]
          ])")),
    };
    for (const auto& input : batches) {
      {
        ARROW_SCOPED_TRACE("if (a) then b else c");
        AssertIfElseEqualWithExpr(field_ref("a"), field_ref("b"), field_ref("c"), schema,
                                  input);
      }
      {
        ARROW_SCOPED_TRACE("if (a) then 1 else 0");
        AssertIfElseEqualWithExpr(field_ref("a"), literal(1), literal(0), schema, input);
      }
    }
  }
}

}  // namespace arrow::compute
