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
#include "arrow/util/logging_internal.h"

namespace arrow::compute {

namespace {

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
                          const ExecBatch& batch,
                          ExecContext* exec_context = default_exec_context()) {
  ARROW_ASSIGN_OR_RAISE(auto bound, expr.Bind(*schema, exec_context));
  return ExecuteScalarExpression(bound, batch, exec_context);
}

#define AssertExprRaisesWithMessage(expr, schema, batch, ENUM, message, ...)     \
  {                                                                              \
    ASSERT_RAISES_WITH_MESSAGE(ENUM, message,                                    \
                               ExecuteExpr(expr, schema, batch, ##__VA_ARGS__)); \
  }

void AssertExprEqualIgnoreShape(const Expression& expr,
                                const std::shared_ptr<Schema>& schema,
                                const ExecBatch& batch, const Datum& expected,
                                ExecContext* exec_context = default_exec_context()) {
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(expr, schema, batch, exec_context));
  AssertEqualIgnoreShape(expected, result);
}

void AssertExprEqualExprsIgnoreShape(const Expression& expr,
                                     const std::vector<Expression>& exprs,
                                     const std::shared_ptr<Schema>& schema,
                                     const ExecBatch& batch,
                                     ExecContext* exec_context = default_exec_context()) {
  ASSERT_OK_AND_ASSIGN(auto expected, ExecuteExpr(expr, schema, batch, exec_context));
  for (const auto& e : exprs) {
    ARROW_SCOPED_TRACE(e.ToString());
    AssertExprEqualIgnoreShape(e, schema, batch, expected, exec_context);
  }
}

auto kBooleanNull = literal(MakeNullScalar(boolean()));
auto kIntNull = literal(MakeNullScalar(int32()));

Expression if_else_regular(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}
Expression unreachable(Expression arg) { return call("unreachable", {std::move(arg)}); }
Expression sv_suppress(Expression arg) { return call("sv_suppress", {std::move(arg)}); }
Expression assert_sv_exist(Expression arg) {
  return call("assert_sv_exist", {std::move(arg)});
}
Expression assert_sv_empty(Expression arg) {
  return call("assert_sv_empty", {std::move(arg)});
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

TEST(IfElseSpecialForm, ImplicitCast) {
  auto schema = arrow::schema({field("i8", int8()), field("i32", int32())});
  for (const auto& if_else_sp :
       {if_else_special(literal(true), field_ref("i8"), field_ref("i32")),
        if_else_special(literal(true), field_ref("i32"), field_ref("i8")),
        // Literal will be downcast.
        if_else_special(literal(true), field_ref("i32"),
                        literal(static_cast<int64_t>(0)))}) {
    ARROW_SCOPED_TRACE(if_else_sp.ToString());
    ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
    ASSERT_EQ(bound.type()->id(), Type::INT32);
  }
}

class IfElseSpecialFormTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ASSERT_OK(RegisterAuxilaryFunctions()); }

 protected:
  static Status UnreachableExec(KernelContext*, const ExecSpan&, ExecResult*) {
    return Status::Invalid("Unreachable");
  }

  static Status IdentityExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
    DCHECK_EQ(span.num_values(), 1);
    const auto& arg = span[0];
    DCHECK(arg.is_array());
    *out->array_data_mutable() = *arg.array.ToArrayData();
    return Status::OK();
  }

  template <bool sv_existence>
  static Status AssertSelectionVectorExec(KernelContext* kernel_ctx, const ExecSpan& span,
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

        ScalarKernel kernel({InputType::Any()}, internal::FirstType, UnreachableExec);
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

        ScalarKernel kernel({InputType::Any()}, internal::FirstType, IdentityExec);
        kernel.can_write_into_slices = false;
        kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
        kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
        RETURN_NOT_OK(func->AddKernel(kernel));
        RETURN_NOT_OK(registry->AddFunction(std::move(func)));
        return Status::OK();
      };

      RETURN_NOT_OK(register_sv_awareness_func("sv_suppress", false));
    }

    {
      auto register_assert_sv_func = [&](const std::string& name,
                                         bool sv_existence) -> Status {
        auto func =
            std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

        ArrayKernelExec exec = AssertSelectionVectorExec<false>;
        ArrayKernelSelectiveExec selective_exec = nullptr;
        if (sv_existence) {
          selective_exec = AssertSelectionVectorExec<true>;
        }
        ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec),
                            /*init=*/nullptr, std::move(selective_exec));
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

  static std::vector<Expression> SuppressSelectionVectorAwareForIfElse(
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
          result.emplace_back(if_else_special(sv_suppress(suppressed_cond),
                                              suppressed_if_true, suppressed_if_false));
          result.emplace_back(if_else_special(suppressed_cond,
                                              sv_suppress(suppressed_if_true),
                                              sv_suppress(suppressed_if_false)));
        }
      }
    }
    return result;
  }

  static void CheckIfElseIgnoreShape(const Expression& cond, const Expression& if_true,
                                     const Expression& if_false,
                                     const std::shared_ptr<Schema>& schema,
                                     const ExecBatch& batch,
                                     ExecContext* exec_context = default_exec_context()) {
    auto if_else = if_else_regular(cond, if_true, if_false);
    auto exprs = SuppressSelectionVectorAwareForIfElse(cond, if_true, if_false);
    AssertExprEqualExprsIgnoreShape(if_else, exprs, schema, batch, exec_context);
  }
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
    ARROW_SCOPED_TRACE("assert selection vector existence");
    {
      ARROW_SCOPED_TRACE("if (a) then a else a");
      auto cond = a;
      auto if_true = a;
      auto if_false = a;
      auto expected = ArrayFromJSON(boolean(), "[null, true, false]");
      {
        auto if_else_sp =
            if_else_special(cond, assert_sv_exist(if_true), assert_sv_exist(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp =
            if_else_special(cond, sv_suppress(if_true), assert_sv_exist(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp =
            if_else_special(cond, assert_sv_exist(if_true), sv_suppress(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp =
            if_else_special(cond, sv_suppress(if_true), assert_sv_empty(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp =
            if_else_special(cond, assert_sv_empty(if_true), sv_suppress(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp = if_else_special(cond, assert_sv_empty(if_true), if_false);
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp = if_else_special(cond, if_true, assert_sv_empty(if_false));
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
    }
    {
      ARROW_SCOPED_TRACE("if (true) then a else false");
      auto cond = literal(true);
      auto if_true = a;
      auto if_false = literal(false);
      auto expected = ArrayFromJSON(boolean(), "[null, true, false]");
      {
        auto if_else_sp = if_else_special(cond, assert_sv_exist(if_true), if_false);
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
      {
        auto if_else_sp = if_else_special(cond, assert_sv_empty(if_true), if_false);
        AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
      }
    }
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
         {kBooleanNull, b_null, assert_sv_empty(kBooleanNull), assert_sv_empty(b_null)}) {
      ARROW_SCOPED_TRACE("cond: " + cond.ToString());
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
          if_else_special(sv_suppress(b), assert_sv_exist(i1), assert_sv_exist(i2)),
          if_else_special(b, sv_suppress(i1), assert_sv_empty(i2)),
          if_else_special(b, assert_sv_empty(i1), sv_suppress(i2))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
  {
    ARROW_SCOPED_TRACE("literal bodies");
    auto expected = ArrayFromJSON(int32(), "[null, 1, 1]");
    for (const auto& if_else_sp :
         {if_else_special(assert_sv_empty(b), assert_sv_empty(literal(1)),
                          assert_sv_exist(i2)),
          if_else_special(assert_sv_empty(b), assert_sv_exist(i1),
                          assert_sv_empty(literal(1))),
          if_else_special(assert_sv_empty(b), assert_sv_empty(literal(1)),
                          assert_sv_empty(literal(1)))}) {
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
                          if_else_special(assert_sv_empty(literal(true)),
                                          assert_sv_exist(i1), unreachable(i2)),
                          if_else_special(assert_sv_empty(literal(false)),
                                          unreachable(i1), assert_sv_exist(i2))),
          if_else_special(b,
                          if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                          assert_sv_exist(i2)),
                          if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                          assert_sv_exist(i2))),
          // Selection vector existences with some argument of the nested if_else_special
          // being selection vector unaware.
          if_else_special(
              b,
              assert_sv_exist(if_else_special(sv_suppress(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(b,
                          assert_sv_exist(if_else_special(
                              sv_suppress(b), assert_sv_exist(i1), assert_sv_exist(i2))),
                          assert_sv_exist(if_else_special(
                              assert_sv_exist(b), unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              sv_suppress(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), sv_suppress(i1),
                                              assert_sv_empty(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              assert_sv_empty(i1),
                                              sv_suppress(unreachable(i2)))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_empty(i1),
                                              sv_suppress(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              unreachable(i1), assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(sv_suppress(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              sv_suppress(unreachable(i1)),
                                              assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), sv_suppress(i1),
                                              assert_sv_empty(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_empty(literal(true)),
                                              assert_sv_exist(i1), unreachable(i2))),
              assert_sv_exist(if_else_special(assert_sv_empty(literal(false)),
                                              assert_sv_empty(unreachable(i1)),
                                              sv_suppress(i2)))),
          if_else_special(
              b,
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i1),
                                              assert_sv_exist(i2))),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_empty(i1),
                                              sv_suppress(i2))))}) {
      ARROW_SCOPED_TRACE(if_else_sp.ToString());
      AssertExprEqualIgnoreShape(if_else_sp, schema, batch, expected);
    }
  }
}

TEST_F(IfElseSpecialFormTest, SelectionVectorExistenceExecChunkSize) {
  ExecContext exec_context;
  constexpr int64_t num_rows = 8;
  auto schema = arrow::schema({field("b", boolean()), field("i", int32())});
  auto b = field_ref("b");
  auto i = field_ref("i");
  auto batch = ExecBatch(
      {
          *ArrayFromJSON(boolean(), "[true, true, true, true, true, true, true, false]"),
          *ArrayFromJSON(int32(), "[42, 42, 42, 42, 42, 42, 42, 42]"),
      },
      num_rows);
  {
    ARROW_SCOPED_TRACE("exec_chunksize >= batch_size");
    for (auto chunksize : {num_rows, num_rows + 1}) {
      ARROW_SCOPED_TRACE("exec_chunksize: " + std::to_string(chunksize));
      exec_context.set_exec_chunksize(chunksize);
      {
        ARROW_SCOPED_TRACE("all literal bodies");
        auto if_else_sp =
            if_else_special(b, assert_sv_empty(literal(1)), assert_sv_empty(literal(0)));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
        ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
      }
      {
        ARROW_SCOPED_TRACE("array true body");
        auto if_else_sp =
            if_else_special(b, assert_sv_exist(i), assert_sv_empty(literal(0)));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
        ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
      }
      {
        ARROW_SCOPED_TRACE("array false body");
        auto if_else_sp =
            if_else_special(b, assert_sv_empty(literal(1)), assert_sv_exist(i));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
        ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
      }
      {
        ARROW_SCOPED_TRACE("all array bodies");
        auto if_else_sp = if_else_special(b, assert_sv_exist(i), assert_sv_exist(i));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
        ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
      }
    }
  }
  {
    ARROW_SCOPED_TRACE("exec_chunksize < batch_size");
    exec_context.set_exec_chunksize(num_rows - 1);
    {
      ARROW_SCOPED_TRACE("all literal bodies");
      auto if_else_sp =
          if_else_special(b, assert_sv_empty(literal(1)), assert_sv_empty(literal(0)));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
    {
      ARROW_SCOPED_TRACE("array true body");
      auto if_else_sp =
          if_else_special(b, assert_sv_exist(i), assert_sv_empty(literal(0)));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
    {
      ARROW_SCOPED_TRACE("array false body");
      auto if_else_sp =
          if_else_special(b, assert_sv_empty(literal(1)), assert_sv_exist(i));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
    {
      ARROW_SCOPED_TRACE("all array bodies");
      auto if_else_sp = if_else_special(b, assert_sv_exist(i), assert_sv_exist(i));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
  }
  {
    ARROW_SCOPED_TRACE("nested");
    {
      ARROW_SCOPED_TRACE("exec_chunksize == batch_size");
      exec_context.set_exec_chunksize(num_rows);
      auto if_else_sp = if_else_special(
          b,
          assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i),
                                          assert_sv_exist(i))),
          assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i),
                                          assert_sv_exist(i))));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
    {
      ARROW_SCOPED_TRACE("exec_chunksize < batch_size");
      exec_context.set_exec_chunksize(num_rows - 2);
      auto if_else_sp = if_else_special(
          b,
          assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i),
                                          assert_sv_exist(i))),
          assert_sv_exist(if_else_special(
              assert_sv_exist(b),
              assert_sv_exist(if_else_special(assert_sv_exist(b), assert_sv_exist(i),
                                              assert_sv_exist(i))),
              assert_sv_exist(i))));
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema, &exec_context));
      ASSERT_OK(ExecuteScalarExpression(bound, batch, &exec_context));
    }
  }
}

// TODO: Selection vector existence of chunked array.

TEST_F(IfElseSpecialFormTest, ResultShape) {
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
    for (const auto& cond : {kBooleanNull, b_null}) {
      ARROW_SCOPED_TRACE("cond: " + cond.ToString());
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
      ARROW_SCOPED_TRACE("cond: " + cond.ToString());
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
      ARROW_SCOPED_TRACE("cond: " + cond.ToString());
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
    for (const auto& nested_cond : {kBooleanNull, b_null}) {
      auto cond = if_else_special(nested_cond, literal(true), literal(false));
      ARROW_SCOPED_TRACE("nested cond: " + cond.ToString());
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
      ARROW_SCOPED_TRACE("nested cond: " + cond.ToString());
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
      ARROW_SCOPED_TRACE("nested cond: " + cond.ToString());
      for (const auto& if_else_sp :
           SuppressSelectionVectorAwareForIfElse(cond, literal(1), literal(0))) {
        ARROW_SCOPED_TRACE(if_else_sp.ToString());
        ASSERT_OK_AND_ASSIGN(auto result, ExecuteExpr(if_else_sp, schema, batch));
        AssertDatumsEqual(expected, result);
      }
    }
  }
  // TODO: Non-scalar branch bodies.
}

namespace {

auto kCanonicalSchema = arrow::schema({field("a", boolean()), field("b", int32())});

auto kCanonicalA = field_ref("a");
auto kCanonicalB = field_ref("b");

const auto kCanonicalBooleanCols = {kBooleanNull, literal(true), literal(false),
                                    kCanonicalA};
const auto kCanonicalIntCols = {kIntNull, literal(42), kCanonicalB};

const std::vector<ExecBatch> kCanonicalBatches = {
    ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
        [null, 0],
        [null, null],
        [null, 1]
      ])")),
    ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
        [true, 0],
        [true, null],
        [true, 1]
      ])")),
    ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
        [false, 0],
        [false, null],
        [false, 1]
      ])")),
    ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
        [false, 0],
        [true, null],
        [null, 1]
      ])")),
};

}  // namespace

TEST_F(IfElseSpecialFormTest, Simple) {
  const auto& schema = kCanonicalSchema;
  const auto& boolean_datums = kCanonicalBooleanCols;
  const auto& int_datums = kCanonicalIntCols;
  const auto& batches = kCanonicalBatches;
  for (const auto& cond : boolean_datums) {
    for (const auto& if_true : int_datums) {
      for (const auto& if_false : int_datums) {
        for (const auto& batch : batches) {
          CheckIfElseIgnoreShape(cond, if_true, if_false, schema, batch);
        }
      }
    }
  }
}

TEST_F(IfElseSpecialFormTest, NestedSimple) {
  const auto& schema = kCanonicalSchema;
  const auto& a = kCanonicalA;
  const auto& b = kCanonicalB;
  ExecBatch batch(*RecordBatchFromJSON(kCanonicalSchema, R"([
      [false, 0],
      [true, null],
      [null, 1]
    ])"));
  for (const auto& cond : {
           if_else_special(a, kBooleanNull, a),
           if_else_special(a, a, literal(true)),
       }) {
    for (const auto& if_true : {
             if_else_special(a, kIntNull, b),
             if_else_special(a, b, literal(42)),
         }) {
      for (const auto& if_false : {
               if_else_special(a, kIntNull, b),
               if_else_special(a, b, literal(42)),
           }) {
        CheckIfElseIgnoreShape(cond, if_true, if_false, schema, batch);
      }
    }
  }
}

// TODO: Deprecate this test due to slowness.
TEST_F(IfElseSpecialFormTest, NestedConditionComplex) {
  const auto& batches = kCanonicalBatches;
  const auto& schema = kCanonicalSchema;
  const auto& boolean_datums = kCanonicalBooleanCols;
  const auto& int_datums = kCanonicalIntCols;
  for (const auto& nested_cond : boolean_datums) {
    for (const auto& nested_if_true : boolean_datums) {
      for (const auto& nested_if_false : boolean_datums) {
        auto nested_if_else_sp =
            if_else_special(nested_cond, nested_if_true, nested_if_false);
        for (const auto& if_true : int_datums) {
          for (const auto& if_false : int_datums) {
            for (const auto& batch : batches) {
              CheckIfElseIgnoreShape(nested_if_else_sp, if_true, if_false, schema, batch);
            }
          }
        }
      }
    }
  }
}

// TODO: Deprecate this test due to slowness.
TEST_F(IfElseSpecialFormTest, NestedBodyComplex) {
  const auto& batches = kCanonicalBatches;
  const auto& schema = kCanonicalSchema;
  const auto& boolean_datums = kCanonicalBooleanCols;
  const auto& int_datums = kCanonicalIntCols;
  for (const auto& cond : boolean_datums) {
    for (const auto& nested_cond : boolean_datums) {
      for (const auto& nested_if_true : int_datums) {
        for (const auto& nested_if_false : int_datums) {
          auto nested_if_else_sp =
              if_else_special(nested_cond, nested_if_true, nested_if_false);
          for (const auto& batch : batches) {
            CheckIfElseIgnoreShape(cond, nested_if_else_sp, nested_if_else_sp, schema,
                                   batch);
          }
        }
      }
    }
  }
}

// TODO: Deprecate this test due to slowness.
TEST_F(IfElseSpecialFormTest, NestedComplex) {
  const auto& batches = kCanonicalBatches;
  const auto& schema = kCanonicalSchema;
  const auto& boolean_datums = kCanonicalBooleanCols;
  const auto& int_datums = kCanonicalIntCols;
  for (const auto& cond_nested_cond : boolean_datums) {
    for (const auto& cond_nested_if_true : boolean_datums) {
      for (const auto& cond_nested_if_false : boolean_datums) {
        auto cond =
            if_else_special(cond_nested_cond, cond_nested_if_true, cond_nested_if_false);
        for (const auto& nested_cond : boolean_datums) {
          for (const auto& nested_if_true : int_datums) {
            for (const auto& nested_if_false : int_datums) {
              auto nested_if_else_sp =
                  if_else_special(nested_cond, nested_if_true, nested_if_false);
              for (const auto& batch : batches) {
                CheckIfElseIgnoreShape(cond, nested_if_else_sp, nested_if_else_sp, schema,
                                       batch);
              }
            }
          }
        }
      }
    }
  }
}

// TODO: ChunkedArray.

// namespace {
// template <bool selection_vector_aware>
// Status ConstantKernelExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
//   DCHECK_EQ(span.num_values(), 1);
//   DCHECK_EQ(span.length, 1);
//   DCHECK(out->is_array_span());
//   DCHECK_EQ(out->length(), 1);
//   if constexpr (!selection_vector_aware) {
//     if (span.selection_vector->length() > 0) {
//       return Status::Invalid("There is a selection vector");
//     }
//   }
//   int32_t* out_values = out->array_span_mutable()->GetValues<int32_t>(1);
//   *out_values = 0;
//   return Status::OK();
// }

// static Status RegisterConstantFunctions() {
//   auto registry = GetFunctionRegistry();

//   auto register_test_func = [&](const std::string& name,
//                                 bool selection_vector_aware) -> Status {
//     auto zero =
//         std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

//     ArrayKernelExec exec;
//     if (selection_vector_aware) {
//       exec = ConstantKernelExec<true>;
//     } else {
//       exec = ConstantKernelExec<false>;
//     }
//     ScalarKernel kernel({InputType::Any()}, OutputType{int32()}, std::move(exec));
//     kernel.selection_vector_aware = selection_vector_aware;
//     kernel.can_write_into_slices = true;
//     kernel.null_handling = NullHandling::OUTPUT_NOT_NULL;
//     kernel.mem_allocation = MemAllocation::PREALLOCATE;
//     RETURN_NOT_OK(zero->AddKernel(kernel));
//     RETURN_NOT_OK(registry->AddFunction(std::move(zero)));
//     return Status::OK();
//   };

//   RETURN_NOT_OK(register_test_func("zero_panic", false));
//   RETURN_NOT_OK(register_test_func("zero_calm", true));

//   return Status::OK();
// }

// }  // namespace

// TEST(IfElseSpecialForm, Reference) {
//   ASSERT_OK(RegisterConstantFunctions());

//   auto schema = arrow::schema({field("a", int32()), field("b", int32())});
//   std::vector<ExecBatch> batches = {
//       ExecBatch(*RecordBatchFromJSON(schema, R"([])")),
//       ExecBatch(*RecordBatchFromJSON(schema, R"([
//             [1, 0],
//             [1, 0],
//             [1, 0]
//           ])")),
//       ExecBatch(*RecordBatchFromJSON(schema, R"([
//             [1, 0],
//             [null, 0],
//             [1, null]
//           ])")),
//   };
//   for (const auto& batch : batches) {
//     auto expr = call("zero_panic", {literal(42)});
//     ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*schema));
//     ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
//     std::cout << result.ToString() << std::endl;
//   }
//   // TODO: The result shape of exec_chunksize, chunked input, and preallocate.
// }

}  // namespace arrow::compute
