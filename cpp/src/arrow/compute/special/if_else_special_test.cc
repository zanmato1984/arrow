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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unordered_set>

#include "arrow/array/concatenate.h"
#include "arrow/array/util.h"
#include "arrow/compute/api_special.h"
#include "arrow/compute/expression_test_internal.h"
#include "arrow/compute/special_form.h"
#include "arrow/compute/test_util_internal.h"
#include "arrow/util/logging_internal.h"

namespace arrow::compute {

using internal::add;
using internal::cast;
using internal::ExpectBindsTo;
using internal::kBoringSchema;
using internal::no_change;
using internal::sub;

TEST(IfElseSpecial, ToString) {
  EXPECT_EQ(
      if_else_special(field_ref("cond"), field_ref("if_true"), field_ref("if_false"))
          .ToString(),
      "if_else_special(cond, if_true, if_false)");
}

TEST(IfElseSpecial, Equality) {
  EXPECT_EQ(if_else_special(literal(true), field_ref("a"), field_ref("b")),
            if_else_special(literal(true), field_ref("a"), field_ref("b")));
  EXPECT_NE(if_else_special(literal(true), field_ref("a"), field_ref("b")),
            if_else_special(literal(false), field_ref("a"), field_ref("b")));
  EXPECT_NE(if_else_special(literal(true), field_ref("a"), field_ref("b")),
            if_else_special(literal(true), field_ref("b"), field_ref("b")));
  EXPECT_NE(if_else_special(literal(true), field_ref("a"), field_ref("b")),
            if_else_special(literal(true), field_ref("a"), field_ref("a")));
  EXPECT_NE(if_else_special(literal(true), field_ref("a"), field_ref("b")),
            call("if_else", {literal(true), field_ref("a"), field_ref("b")}));
}

TEST(IfElseSpecial, Hash) {
  std::unordered_set<Expression, Expression::Hash> set;

  EXPECT_TRUE(
      set.emplace(if_else_special(field_ref("cond"), field_ref("a"), field_ref("b")))
          .second);
  EXPECT_FALSE(
      set.emplace(if_else_special(field_ref("cond"), field_ref("a"), field_ref("b")))
          .second);
  EXPECT_TRUE(
      set.emplace(if_else_special(field_ref("cond"), field_ref("b"), field_ref("a")))
          .second);

  EXPECT_EQ(set.size(), 2);
}

TEST(IfElseSpecial, IsScalarExpression) {
  EXPECT_TRUE(if_else_special(field_ref("cond"), field_ref("a"), field_ref("b"))
                  .IsScalarExpression());
}

TEST(IfElseSpecial, IsSatisfiable) {
  auto Bind = [](Expression expr) { return expr.Bind(*kBoringSchema).ValueOrDie(); };

  EXPECT_TRUE(Bind(if_else_special(field_ref("bool"), field_ref("i32"), field_ref("i32")))
                  .IsSatisfiable());
}

TEST(IfElseSpecial, FieldsInExpression) {
  auto ExpectFieldsAre = [](Expression expr, std::vector<FieldRef> expected) {
    EXPECT_THAT(FieldsInExpression(expr), testing::ContainerEq(expected));
  };

  ExpectFieldsAre(if_else_special(literal(true), literal(1), literal(0)), {});
  ExpectFieldsAre(if_else_special(literal(true), field_ref("a"), field_ref("b")),
                  {"a", "b"});
  ExpectFieldsAre(if_else_special(field_ref("a"), field_ref("b"), field_ref("b")),
                  {"a", "b", "b"});
  ExpectFieldsAre(if_else_special(field_ref("a"), field_ref("b"), field_ref("c")),
                  {"a", "b", "c"});
  ExpectFieldsAre(
      if_else_special(call("not", {field_ref("a")}), call("not", {field_ref("b")}),
                      call("not", {field_ref("c")})),
      {"a", "b", "c"});
  ExpectFieldsAre(
      call("not", {if_else_special(field_ref("a"), field_ref("b"), field_ref("c"))}),
      {"a", "b", "c"});
}

TEST(IfElseSpecial, ExpressionHasFieldRefs) {
  EXPECT_FALSE(
      ExpressionHasFieldRefs(if_else_special(literal(true), literal(1), literal(0))));
  EXPECT_TRUE(
      ExpressionHasFieldRefs(if_else_special(field_ref("a"), literal(1), literal(0))));
  EXPECT_TRUE(
      ExpressionHasFieldRefs(if_else_special(literal(true), field_ref("a"), literal(0))));
  EXPECT_TRUE(
      ExpressionHasFieldRefs(if_else_special(literal(true), literal(0), field_ref("a"))));
}

TEST(IfElseSpecial, BindSpecialForm) {
  {
    auto expr = if_else_special(field_ref("bool"), field_ref("i8"), field_ref("i8"));
    EXPECT_FALSE(expr.IsBound());
    ExpectBindsTo(expr, no_change, &expr);
    EXPECT_TRUE(expr.IsBound());
    EXPECT_TRUE(expr.type()->Equals(*int8()));
  }

  // Implicit casts.
  {
    Expression bound;
    ExpectBindsTo(if_else_special(field_ref("bool"), field_ref("i8"), field_ref("i32")),
                  if_else_special(field_ref("bool"), cast(field_ref("i8"), int32()),
                                  field_ref("i32")),
                  &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }
  {
    Expression bound;
    ExpectBindsTo(if_else_special(field_ref("bool"), field_ref("i32"), field_ref("i8")),
                  if_else_special(field_ref("bool"), field_ref("i32"),
                                  cast(field_ref("i8"), int32())),
                  &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }

  // Nested call.
  {
    Expression bound;
    ExpectBindsTo(if_else_special(equal(field_ref("i8"), field_ref("i8")),
                                  add(field_ref("i8"), literal(42)),
                                  add(field_ref("i32"), literal(42))),
                  if_else_special(equal(field_ref("i8"), field_ref("i8")),
                                  cast(add(field_ref("i8"), literal(42)), int32()),
                                  add(field_ref("i32"), literal(42))),
                  &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }
  {
    Expression bound;
    ExpectBindsTo(if_else_special(equal(field_ref("i8"), field_ref("i32")),
                                  add(field_ref("i32"), field_ref("i8")),
                                  add(field_ref("i32"), literal(42))),
                  if_else_special(equal(cast(field_ref("i8"), int32()), field_ref("i32")),
                                  add(field_ref("i32"), cast(field_ref("i8"), int32())),
                                  add(field_ref("i32"), literal(42))),
                  &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }

  // Nesting call.
  {
    Expression bound;
    ExpectBindsTo(add(if_else_special(field_ref("bool"), field_ref("i32"), literal(42)),
                      field_ref("i8")),
                  add(if_else_special(field_ref("bool"), field_ref("i32"), literal(42)),
                      cast(field_ref("i8"), int32())),
                  &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }
  {
    Expression bound;
    ExpectBindsTo(
        add(if_else_special(field_ref("bool"), field_ref("i8"), literal(42)),
            field_ref("i32")),
        add(cast(if_else_special(field_ref("bool"), field_ref("i8"), literal(42)),
                 int32()),
            field_ref("i32")),
        &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int32()));
  }

  // Self-nested.
  {
    Expression bound;
    ExpectBindsTo(
        if_else_special(
            if_else_special(literal(true), literal(true), literal(false)),
            if_else_special(field_ref("bool"), field_ref("i8"), field_ref("i32")),
            if_else_special(field_ref("bool"), field_ref("i8"), field_ref("i64"))),
        if_else_special(
            if_else_special(literal(true), literal(true), literal(false)),
            cast(if_else_special(field_ref("bool"), cast(field_ref("i8"), int32()),
                                 field_ref("i32")),
                 int64()),
            if_else_special(field_ref("bool"), cast(field_ref("i8"), int64()),
                            field_ref("i64"))),
        &bound);
    EXPECT_TRUE(bound.IsBound());
    EXPECT_TRUE(bound.type()->Equals(*int64()));
  }
}

namespace {

Expression if_else(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Result<Datum> ExecuteExpr(Expression expr, const Schema& schema, const ExecBatch& batch,
                          ExecContext* exec_context = default_exec_context()) {
  ARROW_ASSIGN_OR_RAISE(auto bound, expr.Bind(schema, exec_context));
  return ExecuteScalarExpression(bound, batch, exec_context);
}

void AssertDatumsEqualIgnoreShape(const Datum& expected, const Datum& result) {
  DCHECK(expected.is_scalar() || expected.is_array() || expected.is_chunked_array());
  DCHECK(result.is_scalar() || result.is_array() || result.is_chunked_array());

  if (expected.kind() == result.kind()) {
    AssertDatumsEqual(expected, result);
    return;
  }

  int64_t length = expected.is_scalar() ? result.length() : expected.length();
  auto to_array = [&](const Datum& datum) -> Result<std::shared_ptr<Array>> {
    if (datum.is_scalar()) {
      return MakeArrayFromScalar(*datum.scalar(), length);
    }
    if (datum.is_array()) {
      return datum.make_array();
    }
    DCHECK(datum.is_chunked_array());
    return Concatenate(datum.chunked_array()->chunks(), default_memory_pool());
  };

  ASSERT_OK_AND_ASSIGN(auto expected_array, to_array(expected));
  ASSERT_OK_AND_ASSIGN(auto result_array, to_array(result));
}

using MakeIfElseFunc = std::function<Expression(Expression, Expression, Expression)>;

using MakeExprContainingIfElseFunc =
    std::function<Expression(MakeIfElseFunc make_if_else)>;

void CheckIfElseSpecial(MakeExprContainingIfElseFunc make_expr_containing_if_else,
                        const Schema& schema, const ExecBatch& batch,
                        ExecContext* exec_context = default_exec_context()) {
  auto if_else_expr = make_expr_containing_if_else(if_else);
  ASSERT_OK_AND_ASSIGN(auto expected,
                       ExecuteExpr(if_else_expr, schema, batch, exec_context));
  auto if_else_sp_expr = make_expr_containing_if_else(if_else_special);
  ASSERT_OK_AND_ASSIGN(auto result,
                       ExecuteExpr(if_else_sp_expr, schema, batch, exec_context));
  AssertDatumsEqualIgnoreShape(expected, result);
}

void CheckIfElseSpecial(Expression cond, Expression if_true, Expression if_false,
                        const Schema& schema, const ExecBatch& batch,
                        ExecContext* exec_context = default_exec_context()) {
  CheckIfElseSpecial(
      [=](MakeIfElseFunc make_if_else) {
        return make_if_else(std::move(cond), std::move(if_true), std::move(if_false));
      },
      schema, batch, exec_context);
}

}  // namespace

class TestExecuteIfElseSpecial : public ::testing::Test {
 protected:
  const int64_t length = 7;

  std::shared_ptr<Schema> schm =
      schema({field("boolean1", boolean()), field("boolean2", boolean()),
              field("int1", int32()), field("int2", int32())});
  Expression boolean1 = field_ref("boolean1");
  Expression boolean2 = field_ref("boolean2");
  Expression int1 = field_ref("int1");
  Expression int2 = field_ref("int2");

  std::vector<Expression> boolean_literals = {literal(MakeNullScalar(boolean())),
                                              literal(true), literal(false)};
  std::vector<Expression> boolean_fields = {boolean1, boolean2};
  std::vector<Expression> boolean_complex_exprs = {and_(boolean1, boolean2),
                                                   or_(boolean1, boolean2)};

  std::vector<Expression> int_literals = {literal(MakeNullScalar(int32())), literal(42)};
  std::vector<Expression> int_fields = {int1, int2};
  std::vector<Expression> int_complex_exprs = {add(int1, int2), sub(int1, int2)};

  std::vector<Datum> boolean_scalars = {Datum(MakeNullScalar(boolean())),
                                        Datum(MakeScalar(true)),
                                        Datum(MakeScalar(false))};
  std::shared_ptr<Array> boolean1_arr =
      ArrayFromJSON(boolean(), "[null, true, false, true, false, null, true]");
  std::shared_ptr<ChunkedArray> boolean1_chunked = ChunkedArrayFromJSON(
      boolean(), {"[null, true, false]", "[]", "[true, false, null, true]"});
  std::shared_ptr<Array> boolean2_arr =
      ArrayFromJSON(boolean(), "[true, false, true, true, null, false, true]");
  std::shared_ptr<ChunkedArray> boolean2_chunked = ChunkedArrayFromJSON(
      boolean(), {"[true, false]", "[true]", "[true]", "[null, false, true]"});
  std::vector<Datum> boolean_arrays = {Datum(boolean1_arr), Datum(boolean1_chunked),
                                       Datum(boolean2_arr), Datum(boolean2_chunked)};
  std::vector<Datum> boolean1_arrays = {Datum(boolean1_arr), Datum(boolean1_chunked)};
  std::vector<Datum> boolean2_arrays = {Datum(boolean2_arr), Datum(boolean2_chunked)};

  std::vector<Datum> int_scalars = {Datum(MakeNullScalar(int32())),
                                    Datum(MakeScalar(42))};
  std::shared_ptr<Array> int1_arr = ArrayFromJSON(int32(), "[0, 1, 2, 3, 4, 5, 6]");
  std::shared_ptr<ChunkedArray> int1_chunked =
      ChunkedArrayFromJSON(int32(), {"[0, 1]", "[2, 3, 4]", "[5, 6]"});
  std::shared_ptr<Array> int2_arr = ArrayFromJSON(int32(), "[0, 10, 20, 30, 40, 50, 60]");
  std::shared_ptr<ChunkedArray> int2_chunked =
      ChunkedArrayFromJSON(int32(), {"[0]", "[10, 20]", "[30, 40, 50]", "[]", "[60]"});
  std::vector<Datum> int_arrays = {Datum(int1_arr), Datum(int1_chunked), Datum(int2_arr),
                                   Datum(int2_chunked)};
  std::vector<Datum> int1_arrays = {Datum(int1_arr), Datum(int1_chunked)};
  std::vector<Datum> int2_arrays = {Datum(int2_arr), Datum(int2_chunked)};

 protected:
  void DoTestBasic(const std::vector<Expression>& cond_exprs,
                   const std::vector<Expression>& if_true_exprs,
                   const std::vector<Expression>& if_false_exprs, const ExecBatch& batch);

  void DoTestNestedCond(const std::vector<Expression>& nested_cond_exprs,
                        const std::vector<Expression>& nested_if_true_exprs,
                        const std::vector<Expression>& nested_if_false_exprs,
                        const std::vector<Expression>& outer_if_true_exprs,
                        const std::vector<Expression>& outer_if_false_exprs,
                        const ExecBatch& batch);

  void DoTestNestedBody(const std::vector<Expression>& cond_exprs,
                        const std::vector<Expression>& nested_cond_exprs,
                        const std::vector<Expression>& nested_if_true_exprs,
                        const std::vector<Expression>& nested_if_false_exprs,
                        const std::vector<Expression>& other_branch_exprs,
                        const ExecBatch& batch);

  void WithDatumCombinations(const std::vector<Datum>& boolean1_datums,
                             const std::vector<Datum>& boolean2_datums,
                             const std::vector<Datum>& int1_datums,
                             const std::vector<Datum>& int2_datums,
                             std::function<void(const ExecBatch& batch)> test_func) {
    for (const auto& b1_datum : boolean1_datums) {
      for (const auto& b2_datum : boolean2_datums) {
        for (const auto& i1_datum : int1_datums) {
          for (const auto& i2_datum : int2_datums) {
            ExecBatch batch({b1_datum, b2_datum, i1_datum, i2_datum}, length);
            ARROW_SCOPED_TRACE("batch: " + batch.ToString());
            test_func(batch);
          }
        }
      }
    }
  }
};

void TestExecuteIfElseSpecial::DoTestBasic(const std::vector<Expression>& cond_exprs,
                                           const std::vector<Expression>& if_true_exprs,
                                           const std::vector<Expression>& if_false_exprs,
                                           const ExecBatch& batch) {
  for (const auto& cond_expr : cond_exprs) {
    for (const auto& if_true_expr : if_true_exprs) {
      for (const auto& if_false_expr : if_false_exprs) {
        ARROW_SCOPED_TRACE(
            "if_else_special: " +
            if_else_special(cond_expr, if_true_expr, if_false_expr).ToString());
        CheckIfElseSpecial(cond_expr, if_true_expr, if_false_expr, *schm, batch);
      }
    }
  }
}

TEST_F(TestExecuteIfElseSpecial, BasicAllLiterals) {
  DoTestBasic(boolean_literals, int_literals, int_literals, ExecBatch({}, length));
}

TEST_F(TestExecuteIfElseSpecial, BasicAllScalars) {
  WithDatumCombinations(boolean_scalars, boolean_scalars, int_scalars, int_scalars,
                        [&](const ExecBatch& batch) {
                          DoTestBasic(boolean_fields, {int1}, {int2}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, BasicArrays) {
  WithDatumCombinations(boolean_arrays, boolean_arrays, int_arrays, int_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestBasic(boolean_fields, {int1}, {int2}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, BasicComplexExprs) {
  WithDatumCombinations(boolean1_arrays, boolean2_arrays, int1_arrays, int2_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestBasic(boolean_complex_exprs, int_complex_exprs,
                                      int_complex_exprs, batch);
                        });
}

void TestExecuteIfElseSpecial::DoTestNestedCond(
    const std::vector<Expression>& nested_cond_exprs,
    const std::vector<Expression>& nested_if_true_exprs,
    const std::vector<Expression>& nested_if_false_exprs,
    const std::vector<Expression>& outer_if_true_exprs,
    const std::vector<Expression>& outer_if_false_exprs, const ExecBatch& batch) {
  for (const auto& nested_cond_expr : nested_cond_exprs) {
    for (const auto& nested_if_true_expr : nested_if_true_exprs) {
      for (const auto& nested_if_false_expr : nested_if_false_exprs) {
        for (const auto& outer_if_true_expr : outer_if_true_exprs) {
          for (const auto& outer_if_false_expr : outer_if_false_exprs) {
            ARROW_SCOPED_TRACE(
                "if_else_special: " +
                if_else_special(if_else_special(nested_cond_expr, nested_if_true_expr,
                                                nested_if_false_expr),
                                outer_if_true_expr, outer_if_false_expr)
                    .ToString());
            CheckIfElseSpecial(
                [=](MakeIfElseFunc make_if_else) {
                  return make_if_else(make_if_else(nested_cond_expr, nested_if_true_expr,
                                                   nested_if_false_expr),
                                      outer_if_true_expr, outer_if_false_expr);
                },
                *schm, batch);
          }
        }
      }
    }
  }
}

TEST_F(TestExecuteIfElseSpecial, NestedCondAllLiterals) {
  DoTestNestedCond(boolean_literals, boolean_literals, boolean_literals, int_literals,
                   int_literals, ExecBatch({}, length));
}

TEST_F(TestExecuteIfElseSpecial, NestedCondAllScalars) {
  WithDatumCombinations(boolean_scalars, boolean_scalars, int_scalars, int_scalars,
                        [&](const ExecBatch& batch) {
                          DoTestNestedCond(boolean_fields, boolean_fields, boolean_fields,
                                           {int1}, {int2}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, NestedCondArrays) {
  WithDatumCombinations(boolean1_arrays, boolean2_arrays, int1_arrays, int2_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestNestedCond(boolean_fields, boolean_fields, boolean_fields,
                                           {int1}, {int2}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, NestedCondComplexExprs) {
  WithDatumCombinations(boolean1_arrays, boolean2_arrays, int1_arrays, int2_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestNestedCond(boolean_complex_exprs, boolean_complex_exprs,
                                           boolean_complex_exprs, {int1}, {int2}, batch);
                        });
}

void TestExecuteIfElseSpecial::DoTestNestedBody(
    const std::vector<Expression>& cond_exprs,
    const std::vector<Expression>& nested_cond_exprs,
    const std::vector<Expression>& nested_if_true_exprs,
    const std::vector<Expression>& nested_if_false_exprs,
    const std::vector<Expression>& other_branch_exprs, const ExecBatch& batch) {
  for (const auto& cond_expr : cond_exprs) {
    for (const auto& nested_cond_expr : nested_cond_exprs) {
      for (const auto& nested_if_true_expr : nested_if_true_exprs) {
        for (const auto& nested_if_false_expr : nested_if_false_exprs) {
          for (const auto& other_branch_expr : other_branch_exprs) {
            {
              ARROW_SCOPED_TRACE(
                  "if_else_special: " +
                  if_else_special(cond_expr,
                                  if_else_special(nested_cond_expr, nested_if_true_expr,
                                                  nested_if_false_expr),
                                  other_branch_expr)
                      .ToString());
              CheckIfElseSpecial(
                  [=](MakeIfElseFunc make_if_else) {
                    return make_if_else(
                        cond_expr,
                        make_if_else(nested_cond_expr, nested_if_true_expr,
                                     nested_if_false_expr),
                        other_branch_expr);
                  },
                  *schm, batch);
            }
            {
              ARROW_SCOPED_TRACE(
                  "if_else_special: " +
                  if_else_special(cond_expr, other_branch_expr,
                                  if_else_special(nested_cond_expr, nested_if_true_expr,
                                                  nested_if_false_expr))
                      .ToString());
              CheckIfElseSpecial(
                  [=](MakeIfElseFunc make_if_else) {
                    return make_if_else(
                        cond_expr, other_branch_expr,
                        make_if_else(nested_cond_expr, nested_if_true_expr,
                                     nested_if_false_expr));
                  },
                  *schm, batch);
            }
          }
        }
      }
    }
  }
}

TEST_F(TestExecuteIfElseSpecial, NestedBodyAllLiterals) {
  DoTestNestedBody(boolean_literals, boolean_literals, int_literals, int_literals,
                   int_literals, ExecBatch({}, length));
}

TEST_F(TestExecuteIfElseSpecial, NestedBodyAllScalars) {
  WithDatumCombinations(boolean_scalars, boolean_scalars, int_scalars, int_scalars,
                        [&](const ExecBatch& batch) {
                          DoTestNestedBody({boolean1}, boolean_fields, int_fields,
                                           int_fields, {int1}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, NestedBodyFieldWithArrays) {
  WithDatumCombinations(boolean1_arrays, boolean2_arrays, int1_arrays, int2_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestNestedBody({boolean1}, boolean_fields, int_fields,
                                           int_fields, {int1}, batch);
                        });
}

TEST_F(TestExecuteIfElseSpecial, NestedBodyComplexExprsWithArrays) {
  WithDatumCombinations(boolean1_arrays, boolean2_arrays, int1_arrays, int2_arrays,
                        [&](const ExecBatch& batch) {
                          DoTestNestedBody({boolean1}, boolean_complex_exprs,
                                           int_complex_exprs, int_complex_exprs, {int1},
                                           batch);
                        });
}

namespace {

Result<Datum> ExecuteIfElseSpecial(Expression cond, Expression if_true,
                                   Expression if_false, const Schema& schema,
                                   const ExecBatch& batch,
                                   ExecContext* exec_context = default_exec_context()) {
  return ExecuteExpr(
      if_else_special(std::move(cond), std::move(if_true), std::move(if_false)), schema,
      batch, exec_context);
}

}  // namespace

// GH-41094: Maskable execution of division that otherwise would error on division by
// zero.
TEST(IfElseSpecial, IfNotZeroThenDivide) {
  // if (b != 0) then (a / b) else b
  auto cond = call("not_equal", {field_ref("b"), literal(0)});
  auto if_true = call("divide", {field_ref("a"), field_ref("b")});
  auto if_false = field_ref("b");

  auto schm = schema({field("a", int32()), field("b", int32())});
  auto batch = ExecBatchFromJSON({int32(), int32()},
                                 R"([[1, 1],
                                     [2, 1],
                                     [3, 0],
                                     [4, 1],
                                     [5, 1]])");

  ASSERT_OK_AND_ASSIGN(auto result,
                       ExecuteIfElseSpecial(cond, if_true, if_false, *schm, batch));
  auto expected = ArrayFromJSON(int32(), "[1, 2, 0, 4, 5]");

  AssertDatumsEqual(expected, result);
}

}  // namespace arrow::compute
