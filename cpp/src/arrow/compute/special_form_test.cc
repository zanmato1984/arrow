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
#include "arrow/compute/function.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/logging.h"

namespace arrow::compute {

namespace {

Status UnreachableExec(KernelContext*, const ExecSpan&, ExecResult*) {
  return Status::Invalid("Unreachable");
}

Status IdentityExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
  DCHECK_EQ(span.num_values(), 1);
  const auto& arg = span[0];
  DCHECK(arg.is_array());
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

template <bool sv_existence>
Status AssertSelectionVectorExec(KernelContext* kernel_ctx, const ExecSpan& span,
                                 ExecResult* out) {
  if constexpr (sv_existence) {
    if (!span.selection_vector) {
      return Status::Invalid("There is no selection vector");
    }
  } else {
    if (span.selection_vector) {
      return Status::Invalid("There is a selection vector");
    }
  }
  return IdentityExec(kernel_ctx, span, out);
}

static Status RegisterAuxilaryFunctions() {
  auto registry = GetFunctionRegistry();

  {
    auto register_unreachable_func = [&](const std::string& name) -> Status {
      auto func =
          std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

      ArrayKernelExec exec = UnreachableExec;
      ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
      kernel.selection_vector_aware = true;
      kernel.can_write_into_slices = false;
      kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
      kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
      RETURN_NOT_OK(func->AddKernel(kernel));
      RETURN_NOT_OK(registry->AddFunction(std::move(func)));
      return Status::OK();
    };

    RETURN_NOT_OK(register_unreachable_func("unreachable"));
  }

  {
    auto register_sv_awareness_func = [&](const std::string& name,
                                          bool sv_awareness) -> Status {
      auto func =
          std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

      ArrayKernelExec exec = IdentityExec;
      ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
      kernel.selection_vector_aware = sv_awareness;
      kernel.can_write_into_slices = false;
      kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
      kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
      RETURN_NOT_OK(func->AddKernel(kernel));
      RETURN_NOT_OK(registry->AddFunction(std::move(func)));
      return Status::OK();
    };

    RETURN_NOT_OK(register_sv_awareness_func("sv_aware", true));
    RETURN_NOT_OK(register_sv_awareness_func("sv_suppress", false));
  }

  {
    auto register_assert_sv_func = [&](const std::string& name,
                                       bool sv_existence) -> Status {
      auto func =
          std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

      ArrayKernelExec exec;
      if (sv_existence) {
        exec = AssertSelectionVectorExec<true>;
      } else {
        exec = AssertSelectionVectorExec<false>;
      }
      ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
      kernel.selection_vector_aware = true;
      kernel.can_write_into_slices = false;
      kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
      kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
      RETURN_NOT_OK(func->AddKernel(kernel));
      RETURN_NOT_OK(registry->AddFunction(std::move(func)));
      return Status::OK();
    };

    RETURN_NOT_OK(register_assert_sv_func("assert_sv_exist", /*sv_existence=*/true));
    RETURN_NOT_OK(register_assert_sv_func("assert_sv_empty", /*sv_existence=*/false));
  }

  return Status::OK();
}

auto boolean_null = literal(MakeNullScalar(boolean()));

Expression if_else_expr(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Expression unreachable(Expression arg) { return call("unreachable", {std::move(arg)}); }

Expression sv_aware(Expression arg) { return call("sv_aware", {std::move(arg)}); }

Expression sv_suppress(Expression arg) { return call("sv_suppress", {std::move(arg)}); }

Expression assert_sv_exist(Expression arg) {
  return call("assert_sv_exist", {std::move(arg)});
}

Expression assert_sv_empty(Expression arg) {
  return call("assert_sv_empty", {std::move(arg)});
}

std::vector<Expression> SuppressSelectionVectorAwareForIfElse(
    const Expression& cond, const Expression& if_true, const Expression& if_false) {
  auto suppress_if_else_recursive =
      [&](const Expression& expr) -> std::vector<Expression> {
    if (const auto& sp = expr.special(); sp && sp->special_form->name == "if_else") {
      const auto& cond = sp->arguments[0];
      const auto& if_true = sp->arguments[1];
      const auto& if_false = sp->arguments[2];
      return SuppressSelectionVectorAwareForIfElse(cond, if_true, if_false);
    } else {
      return {expr};
    }
  };
  auto suppressed_conds = suppress_if_else_recursive(cond);
  auto suppressed_if_trues = suppress_if_else_recursive(if_true);
  auto suppressed_if_falses = suppress_if_else_recursive(if_false);
  std::vector<Expression> result;
  for (const auto& suppressed_cond : suppressed_conds) {
    for (const auto& suppressed_if_true : suppressed_if_trues) {
      for (const auto& suppressed_if_false : suppressed_if_falses) {
        result.emplace_back(
            if_else_special(suppressed_cond, suppressed_if_true, suppressed_if_false));
        result.emplace_back(if_else_special(suppressed_cond, suppressed_if_true,
                                            sv_suppress(suppressed_if_false)));
      }
    }
  }
  return result;
}

void AssertEqualIgnoreShape(const Datum& expected, const Datum& result) {
  if (expected.kind() == result.kind()) {
    AssertDatumsEqual(expected, result);
    return;
  }
  if (expected.is_scalar()) {
    ASSERT_OK_AND_ASSIGN(auto expected_array,
                         MakeArrayFromScalar(*expected.scalar(), result.length()));
    AssertDatumsEqual(expected_array, result);
    return;
  }
  if (result.is_scalar()) {
    ASSERT_OK_AND_ASSIGN(auto result_array,
                         MakeArrayFromScalar(*result.scalar(), expected.length()));
    AssertDatumsEqual(expected, result_array);
    return;
  }
}

Result<Datum> ExecuteExpr(const Expression& expr, const std::shared_ptr<Schema>& schema,
                          const ExecBatch& batch) {
  ARROW_ASSIGN_OR_RAISE(auto bound, expr.Bind(*schema));
  return ExecuteScalarExpression(bound, batch);
}

void AssertExprEqualIgnoreShape(const Expression& expr,
                                const std::shared_ptr<Schema>& schema,
                                const ExecBatch& batch, const Datum& expected) {
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(expr, schema, batch));
  AssertEqualIgnoreShape(expected, result);
}

#define AssertExprRaisesWithMessage(expr, schema, batch, ENUM, message) \
  { ASSERT_RAISES_WITH_MESSAGE(ENUM, message, ExecuteExpr(expr, schema, batch)); }

void AssertExprEqualExprs(const Expression& expr, const std::vector<Expression>& exprs,
                          const std::shared_ptr<Schema>& schema, const ExecBatch& batch) {
  ASSERT_OK_AND_ASSIGN(auto expected, ExecuteExpr(expr, schema, batch));
  for (const auto& e : exprs) {
    ARROW_SCOPED_TRACE(e.ToString());
    AssertExprEqualIgnoreShape(e, schema, batch, expected);
  }
}

void CheckIfElse(const Expression& cond, const Expression& if_true,
                 const Expression& if_false, const std::shared_ptr<Schema>& schema,
                 const ExecBatch& batch) {
  auto if_else = if_else_expr(cond, if_true, if_false);
  auto exprs = SuppressSelectionVectorAwareForIfElse(cond, if_true, if_false);
  AssertExprEqualExprs(if_else, exprs, schema, batch);
}

}  // namespace

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
    auto batch = ExecBatch(*rb);
    auto if_else_sp = if_else_special(cond, if_true, if_false);
    {
      auto expected = ArrayFromJSON(int32(), "[1, 2, 0, 4, 5]");
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
    {
      ARROW_SCOPED_TRACE("(if (b != 0) then a / b else b) + 1");
      auto plus_one = call("add", {if_else_sp, literal(1)});
      {
        auto expected = ArrayFromJSON(int32(), "[2, 3, 1, 5, 6]");
        AssertExprEqualIgnoreShape(plus_one, schema, batch, expected);
      }
      {
        ARROW_SCOPED_TRACE(
            "if ((if (b != 0) then a / b else b) + 1 != 1) then a / b else b");
        auto cond = call("not_equal", {plus_one, literal(1)});
        auto if_true = call("divide", {field_ref("a"), field_ref("b")});
        auto if_false = field_ref("b");
        auto if_else_sp = if_else_special(cond, if_true, if_false);
        auto expected = ArrayFromJSON(int32(), "[1, 2, 0, 4, 5]");
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
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
    auto batch = ExecBatch(*rb);
    auto if_else_sp = if_else_special(cond, if_true, if_false);
    auto expected = ArrayFromJSON(int32(), "[1, 2, 3, 4, 5]");
    AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
  }
}

class IfElseSpecialFormTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ASSERT_OK(RegisterAuxilaryFunctions()); }
};

TEST_F(IfElseSpecialFormTest, AuxilaryFunction) {
  auto schema = arrow::schema({field("a", boolean())});
  auto a = field_ref("a");
  auto batch = ExecBatch({*ArrayFromJSON(boolean(), "[null, true, false]")}, 3);
  {
    ARROW_SCOPED_TRACE("unreachable");
    AssertExprRaisesWithMessage(unreachable(a), schema, batch, Invalid,
                                "Invalid: Unreachable");
  }
  {
    ARROW_SCOPED_TRACE("selection vector awareness");
    {
      ASSERT_OK_AND_ASSIGN(auto bound, sv_aware(a).Bind(*schema));
      ASSERT_TRUE(bound.selection_vector_aware());
    }
    {
      ASSERT_OK_AND_ASSIGN(auto bound, sv_suppress(a).Bind(*schema));
      ASSERT_FALSE(bound.selection_vector_aware());
    }
    {
      ASSERT_OK_AND_ASSIGN(auto bound, sv_aware(sv_aware(a)).Bind(*schema));
      ASSERT_TRUE(bound.selection_vector_aware());
    }
    {
      ASSERT_OK_AND_ASSIGN(auto bound, sv_aware(sv_suppress(a)).Bind(*schema));
      ASSERT_FALSE(bound.selection_vector_aware());
    }
    {
      ASSERT_OK_AND_ASSIGN(auto bound, sv_suppress(sv_aware(a)).Bind(*schema));
      ASSERT_FALSE(bound.selection_vector_aware());
    }
  }
  {
    ARROW_SCOPED_TRACE("assert selection vector existence");
    {
      ARROW_SCOPED_TRACE("if (a) then 1 else 0");
      auto cond = a;
      auto if_true = literal(1);
      auto if_false = literal(0);
      auto expected = ArrayFromJSON(int32(), "[null, 1, 0]");
      {
        auto if_else_sp =
            if_else_special(cond, assert_sv_exist(if_true), assert_sv_exist(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp =
            if_else_special(sv_suppress(cond), assert_sv_exist(if_true), if_false);
        AssertExprRaisesWithMessage(if_else_sp, schema, batch, Invalid,
                                    "Invalid: There is no selection vector");
      }
      {
        auto if_else_sp = if_else_special(cond, assert_sv_empty(if_true), if_false);
        AssertExprRaisesWithMessage(if_else_sp, schema, batch, Invalid,
                                    "Invalid: There is a selection vector");
      }
      {
        auto if_else_sp = if_else_special(sv_suppress(cond), assert_sv_empty(if_true),
                                          assert_sv_empty(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
    }
    {
      ARROW_SCOPED_TRACE("if (true) then 1 else 0");
      auto cond = literal(true);
      auto if_true = a;
      auto if_false = literal(false);
      auto expected = ArrayFromJSON(boolean(), "[null, true, false]");
      {
        auto if_else_sp = if_else_special(cond, assert_sv_exist(if_true), if_false);
        AssertExprRaisesWithMessage(if_else_sp, schema, batch, Invalid,
                                    "Invalid: There is no selection vector");
      }
      {
        auto if_else_sp = if_else_special(cond, assert_sv_empty(if_true), if_false);
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
    }
  }
}

TEST_F(IfElseSpecialFormTest, SelectionVectorAwareness) {
  auto schema = arrow::schema({});
  auto cond = literal(true);
  auto if_true = literal(1);
  auto if_false = literal(0);
  for (const auto& if_else_sp : {if_else_special(cond, if_true, if_false),
                                 if_else_special(sv_suppress(cond), if_true, if_false),
                                 if_else_special(cond, sv_suppress(if_true), if_false),
                                 if_else_special(cond, if_true, sv_suppress(if_false))}) {
    ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
    ASSERT_TRUE(bound.selection_vector_aware());
  }
}

TEST_F(IfElseSpecialFormTest, SelectionVectorExistence) {
  auto schema = arrow::schema({field("b_null", boolean()), field("b_true", boolean()),
                               field("b_false", boolean()), field("b", boolean()),
                               field("i1", int32()), field("i2", int32())});
  auto b_null = field_ref("b_null");
  auto b_true = field_ref("b_true");
  auto b_false = field_ref("b_false");
  auto b = field_ref("b");
  auto i1 = field_ref("i1");
  auto i2 = field_ref("i2");
  auto batch = ExecBatch(
      {
          *ArrayFromJSON(boolean(), "[null, null, null]"),
          *ArrayFromJSON(boolean(), "[true, true, true]"),
          *ArrayFromJSON(boolean(), "[false, false, false]"),
          *ArrayFromJSON(boolean(), "[null, true, false]"),
          *ArrayFromJSON(int32(), "[0, 1, -1]"),
          *ArrayFromJSON(int32(), "[0, -1, 1]"),
      },
      3);

  {
    ARROW_SCOPED_TRACE("all null condition");
    auto expected = ArrayFromJSON(int32(), "[null, null, null]");
    for (const auto& cond :
         {boolean_null, b_null, assert_sv_empty(boolean_null), assert_sv_empty(b_null)}) {
      ARROW_SCOPED_TRACE(cond.ToString());
      auto if_else_sp = if_else_special(cond, unreachable(i1), unreachable(i2));
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
  {
    ARROW_SCOPED_TRACE("all true condition");
    auto expected = ArrayFromJSON(int32(), "[0, 1, -1]");
    for (const auto& if_else_sp :
         {if_else_special(literal(true), i1, unreachable(i2)),
          if_else_special(b_true, i1, unreachable(i2)),
          if_else_special(assert_sv_empty(literal(true)), assert_sv_empty(i1),
                          unreachable(i2)),
          if_else_special(assert_sv_empty(b_true), assert_sv_exist(i1), unreachable(i2)),
          if_else_special(assert_sv_empty(b_true), assert_sv_empty(i1),
                          sv_suppress(unreachable(i2)))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
  {
    ARROW_SCOPED_TRACE("all false condition");
    auto expected = ArrayFromJSON(int32(), "[0, -1, 1]");
    for (const auto& if_else_sp :
         {if_else_special(literal(false), unreachable(i1), i2),
          if_else_special(b_false, unreachable(i1), i2),
          if_else_special(assert_sv_empty(literal(false)), unreachable(i1),
                          assert_sv_empty(i2)),
          if_else_special(assert_sv_empty(b_false), unreachable(i1), assert_sv_exist(i2)),
          if_else_special(assert_sv_empty(b_false), sv_suppress(unreachable(i1)),
                          assert_sv_empty(i2))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
  {
    ARROW_SCOPED_TRACE("even condition");
    auto expected = ArrayFromJSON(int32(), "[null, 1, 1]");
    for (const auto& if_else_sp :
         {if_else_special(b, i1, i2),
          if_else_special(assert_sv_empty(b), assert_sv_exist(i1), assert_sv_exist(i2)),
          // TODO: This may not hold in the future.
          if_else_special(sv_suppress(b), assert_sv_empty(i1), assert_sv_empty(i2)),
          if_else_special(b, sv_suppress(i1), assert_sv_empty(i2)),
          if_else_special(b, assert_sv_empty(i1), sv_suppress(i2))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
  {
    ARROW_SCOPED_TRACE("nested");
    auto expected = ArrayFromJSON(int32(), "[null, 1, 1]");
    for (const auto& if_else_sp :
         {if_else_special(b, if_else_special(b, i1, i2), if_else_special(b, i1, i2)),
          // The nested if_else_special will see a selection vector.
          if_else_special(b, assert_sv_exist(if_else_special(b, i1, i2)),
                          assert_sv_exist(if_else_special(b, i1, i2))),
          // The arguments of nested if_else_special will see a selection vector.
          if_else_special(b,
                          if_else_special(assert_sv_exist(literal(true)),
                                          assert_sv_exist(i1), unreachable(i2)),
                          if_else_special(assert_sv_exist(literal(false)),
                                          unreachable(i1), assert_sv_exist(i2))),
          if_else_special(b,
                          if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                          assert_sv_exist(i2)),
                          if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                          assert_sv_exist(i2))),
          // Selection vector existences with some argument of the nested
          // if_else_special being selection vector unaware.
          if_else_special(
              b,
              assert_sv_exist(if_else_special(sv_suppress(literal(true)),
                                              assert_sv_empty(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(b,
                          assert_sv_exist(if_else_special(
                              sv_suppress(b), assert_sv_empty(i1), assert_sv_empty(i2))),
                          assert_sv_exist(if_else_special(
                              assert_sv_exist(b), unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              sv_suppress(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(b), sv_suppress(i1),
                                              assert_sv_empty(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              assert_sv_empty(i1),
                                              sv_suppress(unreachable(i2)))),
              assert_sv_exist(if_else_special(assert_sv_exist(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(b), assert_sv_empty(i1),
                                              sv_suppress(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(sv_suppress(literal(false)),
                                              unreachable(i1), assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(sv_suppress(b), assert_sv_empty(i1),
                                              assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              sv_suppress(unreachable(i1)),
                                              assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(b), sv_suppress(i1),
                                              assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              assert_sv_empty(unreachable(i1)),
                                              sv_suppress(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(b), assert_sv_empty(i1),
                                              sv_suppress(i2))))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
}

TEST_F(IfElseSpecialFormTest, Shortcut) {
  {
    ARROW_SCOPED_TRACE("scalar branch");
    auto schema = arrow::schema({field("b_null", boolean()), field("b_true", boolean()),
                                 field("b_false", boolean())});
    auto b_null = field_ref("b_null");
    auto b_true = field_ref("b_true");
    auto b_false = field_ref("b_false");
    auto batch = ExecBatch(
        {
            *ArrayFromJSON(boolean(), "[null, null, null]"),
            *ArrayFromJSON(boolean(), "[true, true, true]"),
            *ArrayFromJSON(boolean(), "[false, false, false]"),
        },
        3);
    {
      ARROW_SCOPED_TRACE("if (null) then 1 else 0");
      auto expected = ArrayFromJSON(int32(), "[null, null, null]");
      for (const auto& cond : {boolean_null, b_null}) {
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (true) then 1 else 0");
      auto expected = MakeScalar(1);
      for (const auto& cond : {literal(true), b_true}) {
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (false) then 1 else 0");
      auto expected = MakeScalar(0);
      for (const auto& cond : {literal(false), b_false}) {
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (if (null) true then false) then 1 else 0");
      auto expected = ArrayFromJSON(int32(), "[null, null, null]");
      for (const auto& nested_cond : {boolean_null, b_null}) {
        auto cond = if_else_special(nested_cond, literal(true), literal(false));
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (if (true) then true else false) then 1 else 0");
      auto expected = MakeScalar(1);
      for (const auto& nested_cond : {literal(true), b_true}) {
        auto cond = if_else_special(nested_cond, nested_cond, literal(false));
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (if (false) then true else false) then 1 else 0");
      auto expected = MakeScalar(0);
      for (const auto& nested_cond : {literal(false), b_false}) {
        auto cond = if_else_special(nested_cond, literal(true), nested_cond);
        for (const auto& if_else_sp :
             SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
          ARROW_SCOPED_TRACE(if_else_sp.ToString());
          ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
          AssertDatumsEqual(expected, result);
        }
      }
    }
  }
  {
    auto schema = arrow::schema({field("", int32())});
    std::vector<ExecBatch> batches = {
        ExecBatch({*ArrayFromJSON(int32(), "[]")}, 0),
        ExecBatch({*ArrayFromJSON(int32(), "[1]")}, 1),
    };
    for (const auto& batch : batches) {
      {
        ARROW_SCOPED_TRACE("if (null) then 1 else 0");
        CheckIfElse(literal(MakeNullScalar(boolean())), literal(1), literal(0), schema,
                    batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then 1 else 0");
        CheckIfElse(literal(true), literal(1), literal(0), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then 1 else 0");
        CheckIfElse(literal(false), literal(1), literal(0), schema, batch);
      }
    }
  }
  {
    auto schema = arrow::schema({field("a", int32()), field("b", int32())});
    auto a = field_ref("a");
    auto b = field_ref("b");
    std::vector<ExecBatch> batches = {
        ExecBatch(*RecordBatchFromJSON(schema, R"([])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [1, 0],
            [1, 0],
            [1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [1, 0],
            [null, 0],
            [1, null]
          ])")),
    };
    for (const auto& batch : batches) {
      {
        ARROW_SCOPED_TRACE("if (null) then a else b");
        CheckIfElse(boolean_null, a, b, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then 0 else b");
        CheckIfElse(literal(true), literal(0), b, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then a else b");
        CheckIfElse(literal(true), a, b, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else 1");
        CheckIfElse(literal(false), a, literal(1), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else b");
        CheckIfElse(literal(false), a, b, schema, batch);
      }
    }
  }
  {
    auto schema =
        arrow::schema({field("a", boolean()), field("b", int32()), field("c", int32())});
    auto a = field_ref("a");
    auto b = field_ref("b");
    auto c = field_ref("c");
    std::vector<ExecBatch> batches = {
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [null, 1, 0],
            [null, 1, 0],
            [null, 1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [null, 1, 0],
            [null, null, 0],
            [null, 1, null]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [true, 1, 0],
            [true, 1, 0],
            [true, 1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [true, 1, 0],
            [true, null, 0],
            [true, 1, null]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [false, 1, 0],
            [false, 1, 0],
            [false, 1, 0]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [false, 1, 0],
            [false, null, 0],
            [false, 1, null]
          ])")),
        ExecBatch(*RecordBatchFromJSON(schema, R"([
            [false, 1, 0],
            [true, null, 0],
            [null, 1, null]
          ])")),
    };
    for (const auto& batch : batches) {
      {
        ARROW_SCOPED_TRACE("if (a) then 1 else 0");
        CheckIfElse(a, literal(1), literal(0), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (a) then b else c");
        CheckIfElse(a, b, c, schema, batch);
      }
      auto boolean_datums = {boolean_null, literal(true), literal(false), a};
      for (const auto& nested_cond : boolean_datums) {
        for (const auto& nested_if_true : boolean_datums) {
          for (const auto& nested_if_false : boolean_datums) {
            auto nested_if_else_sp =
                if_else_special(nested_cond, nested_if_true, nested_if_false);
            for (const auto& if_true : {literal(0), literal(1), b}) {
              for (const auto& if_false : {literal(0), literal(1), c}) {
                ARROW_SCOPED_TRACE(
                    if_else_special(nested_if_else_sp, if_true, if_false).ToString());
                ASSERT_OK_AND_ASSIGN(
                    auto r,
                    ExecuteExpr(if_else_special(nested_if_else_sp, if_true, if_false),
                                schema, batch));
                CheckIfElse(nested_if_else_sp, if_true, if_false, schema, batch);
              }
            }
          }
        }
      }
    }
  }
}

// TODO: ChunkedArray.

namespace {
template <bool selection_vector_aware>
Status ConstantKernelExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
  DCHECK_EQ(span.num_values(), 1);
  DCHECK_EQ(span.length, 1);
  DCHECK(out->is_array_span());
  DCHECK_EQ(out->length(), 1);
  if constexpr (!selection_vector_aware) {
    if (span.selection_vector != nullptr) {
      return Status::Invalid("There is a selection vector");
    }
  }
  int32_t* out_values = out->array_span_mutable()->GetValues<int32_t>(1);
  *out_values = 0;
  return Status::OK();
}

static Status RegisterConstantFunctions() {
  auto registry = GetFunctionRegistry();

  auto register_test_func = [&](const std::string& name,
                                bool selection_vector_aware) -> Status {
    auto zero =
        std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

    ArrayKernelExec exec;
    if (selection_vector_aware) {
      exec = ConstantKernelExec<true>;
    } else {
      exec = ConstantKernelExec<false>;
    }
    ScalarKernel kernel({InputType::Any()}, OutputType{int32()}, std::move(exec));
    kernel.selection_vector_aware = selection_vector_aware;
    kernel.can_write_into_slices = true;
    kernel.null_handling = NullHandling::OUTPUT_NOT_NULL;
    kernel.mem_allocation = MemAllocation::PREALLOCATE;
    RETURN_NOT_OK(zero->AddKernel(kernel));
    RETURN_NOT_OK(registry->AddFunction(std::move(zero)));
    return Status::OK();
  };

  RETURN_NOT_OK(register_test_func("zero_panic", false));
  RETURN_NOT_OK(register_test_func("zero_calm", true));

  return Status::OK();
}

}  // namespace

TEST(IfElseSpecialForm, Reference) {
  ASSERT_OK(RegisterConstantFunctions());

  auto schema = arrow::schema({field("a", int32()), field("b", int32())});
  std::vector<ExecBatch> batches = {
      ExecBatch(*RecordBatchFromJSON(schema, R"([])")),
      ExecBatch(*RecordBatchFromJSON(schema, R"([
            [1, 0],
            [1, 0],
            [1, 0]
          ])")),
      ExecBatch(*RecordBatchFromJSON(schema, R"([
            [1, 0],
            [null, 0],
            [1, null]
          ])")),
  };
  for (const auto& batch : batches) {
    auto expr = call("zero_panic", {literal(42)});
    ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*schema));
    ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
    std::cout << result.ToString() << std::endl;
  }
}

}  // namespace arrow::compute
