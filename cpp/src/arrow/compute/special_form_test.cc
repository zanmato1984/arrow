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
      ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
      ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
      AssertDatumsEqual(*expected, result);
    }
    {
      ARROW_SCOPED_TRACE("(if (b != 0) then a / b else b) + 1");
      auto plus_one = call("add", {if_else_sp, literal(1)});
      {
        auto expected = ArrayFromJSON(int32(), "[2, 3, 1, 5, 6]");
        ASSERT_OK_AND_ASSIGN(auto bound, plus_one.Bind(*schema));
        ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
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
        ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
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
    auto batch = ExecBatch(*rb);
    auto if_else_sp = if_else_special(cond, if_true, if_false);
    auto expected = ArrayFromJSON(int32(), "[1, 2, 3, 4, 5]");
    ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
    ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
    AssertDatumsEqual(*expected, result);
  }
}

namespace {

Expression if_else_expr(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Status HaltExec(KernelContext*, const ExecSpan&, ExecResult*) {
  return Status::Invalid("Halting");
}

static Status RegisterHaltFunctions() {
  auto registry = GetFunctionRegistry();

  auto register_halt_func = [&](const std::string& name,
                                bool selection_vector_aware) -> Status {
    auto func =
        std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

    ArrayKernelExec exec = HaltExec;
    ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
    kernel.selection_vector_aware = selection_vector_aware;
    kernel.can_write_into_slices = false;
    kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
    kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
    RETURN_NOT_OK(func->AddKernel(kernel));
    RETURN_NOT_OK(registry->AddFunction(std::move(func)));
    return Status::OK();
  };

  RETURN_NOT_OK(register_halt_func("halt_selection_vector_aware",
                                   /*selection_vector_aware=*/true));
  RETURN_NOT_OK(register_halt_func("halt_selection_vector_unaware",
                                   /*selection_vector_aware=*/false));

  return Status::OK();
}

Expression halt_selection_vector_aware(Expression arg) {
  return call("halt_selection_vector_aware", {std::move(arg)});
}

Expression halt_selection_vector_unaware(Expression arg) {
  return call("halt_selection_vector_unaware", {std::move(arg)});
}

template <bool selection_vector_aware, bool assert_selection_vector_existing>
Status SelectionVectorGuardExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
  DCHECK_EQ(span.num_values(), 1);
  if constexpr (!selection_vector_aware) {
    if (span.selection_vector) {
      return Status::Invalid("There is a selection vector");
    }
  } else if constexpr (assert_selection_vector_existing) {
    if (!span.selection_vector) {
      return Status::Invalid("There is no selection vector");
    }
  }
  const auto& arg = span[0];
  DCHECK(arg.is_array());
  *out->array_data_mutable() = *arg.array.ToArrayData();
  return Status::OK();
}

static Status RegisterSelectionVectorGuardFunctions() {
  auto registry = GetFunctionRegistry();

  auto register_selection_vector_guard_func =
      [&](const std::string& name, bool actual_selection_vector_aware,
          bool assert_selection_vector_existing,
          bool declare_selection_vector_aware) -> Status {
    auto func =
        std::make_shared<ScalarFunction>(name, Arity::Unary(), FunctionDoc::Empty());

    ArrayKernelExec exec;
    if (actual_selection_vector_aware) {
      if (assert_selection_vector_existing) {
        exec = SelectionVectorGuardExec<true, true>;
      } else {
        exec = SelectionVectorGuardExec<true, false>;
      }
    } else {
      if (assert_selection_vector_existing) {
        exec = SelectionVectorGuardExec<false, true>;
      } else {
        exec = SelectionVectorGuardExec<false, false>;
      }
    }
    ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec));
    kernel.selection_vector_aware = declare_selection_vector_aware;
    kernel.can_write_into_slices = false;
    kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
    kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
    RETURN_NOT_OK(func->AddKernel(kernel));
    RETURN_NOT_OK(registry->AddFunction(std::move(func)));
    return Status::OK();
  };

  RETURN_NOT_OK(
      register_selection_vector_guard_func("selection_vector_aware",
                                           /*actual_selection_vector_aware=*/true,
                                           /*assert_selection_vector_existing=*/false,
                                           /*declare_selection_vector_aware=*/true));
  RETURN_NOT_OK(
      register_selection_vector_guard_func("selection_vector_unaware",
                                           /*actual_selection_vector_aware=*/false,
                                           /*assert_selection_vector_existing=*/false,
                                           /*declare_selection_vector_aware=*/false));
  RETURN_NOT_OK(
      register_selection_vector_guard_func("selection_vector_mis_aware",
                                           /*actual_selection_vector_aware=*/false,
                                           /*assert_selection_vector_existing=*/false,
                                           /*declare_selection_vector_aware=*/true));
  RETURN_NOT_OK(
      register_selection_vector_guard_func("selection_vector_must_exist",
                                           /*actual_selection_vector_aware=*/true,
                                           /*assert_selection_vector_existing=*/true,
                                           /*declare_selection_vector_aware=*/true));

  return Status::OK();
}

Expression selection_vector_aware(Expression arg) {
  return call("selection_vector_aware", {std::move(arg)});
}

Expression selection_vector_unaware(Expression arg) {
  return call("selection_vector_unaware", {std::move(arg)});
}

Expression selection_vector_mis_aware(Expression arg) {
  return call("selection_vector_mis_aware", {std::move(arg)});
}

Expression selection_vector_must_exist(Expression arg) {
  return call("selection_vector_must_exist", {std::move(arg)});
}

Expression wrap_if_else_special_with_selection_vector_aware(Expression cond,
                                                            Expression if_true,
                                                            Expression if_false) {
  return selection_vector_aware(if_else_special(selection_vector_aware(cond),
                                                selection_vector_aware(if_true),
                                                selection_vector_aware(if_false)));
}

Expression wrap_if_else_special_with_selection_vector_unaware(Expression cond,
                                                              Expression if_true,
                                                              Expression if_false) {
  return selection_vector_unaware(if_else_special(selection_vector_unaware(cond),
                                                  selection_vector_unaware(if_true),
                                                  selection_vector_unaware(if_false)));
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

std::vector<Expression> WrapSelectionVectorGuard(Expression cond, Expression if_true,
                                                 Expression if_false) {
  return {
      if_else_special(cond, if_true, if_false),
      wrap_if_else_special_with_selection_vector_aware(cond, if_true, if_false),
      wrap_if_else_special_with_selection_vector_unaware(cond, if_true, if_false),
  };
}

void AssertExprEqualIgnoreShape(const Datum& expected, Expression expr,
                                const std::shared_ptr<Schema>& schema,
                                const ExecBatch& batch) {
  ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*schema));
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
  AssertEqualIgnoreShape(expected, result);
}

void AssertExprsEqualIgnoreShape(const Datum& expected,
                                 const std::vector<Expression>& exprs,
                                 const std::shared_ptr<Schema>& schema,
                                 const ExecBatch& batch) {
  for (const auto& expr : exprs) {
    AssertExprEqualIgnoreShape(expected, expr, schema, batch);
  }
}

void AssertIfElseEqualWithExpr(Expression cond, Expression if_true, Expression if_false,
                               const std::shared_ptr<Schema>& schema,
                               const ExecBatch& batch) {
  auto if_else = if_else_expr(cond, if_true, if_false);
  ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteScalarExpression(bound, batch));
  // Test using
  // original/selection_vector_aware(original)/selection_vector_unaware(original).
  auto exprs = WrapSelectionVectorGuard(cond, if_true, if_false);
  AssertExprsEqualIgnoreShape(result, exprs, schema, batch);
}

}  // namespace

class IfElseSpecialFormTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_OK(RegisterHaltFunctions());
    ASSERT_OK(RegisterSelectionVectorGuardFunctions());
  }
};

TEST_F(IfElseSpecialFormTest, SelectionVector) {
  {
    auto schema = arrow::schema({field("", int32())});
    auto batch = ExecBatch({*ArrayFromJSON(int32(), "[1]")}, 1);
    {
      ARROW_SCOPED_TRACE("halt");
      {
        auto if_else = if_else_expr(literal(MakeNullScalar(boolean())),
                                    halt_selection_vector_aware(literal(1)), literal(0));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
        ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                   ExecuteScalarExpression(bound, batch));
      }
      {
        auto if_else = if_else_expr(literal(MakeNullScalar(boolean())), literal(1),
                                    halt_selection_vector_aware(literal(0)));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
        ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                   ExecuteScalarExpression(bound, batch));
      }
    }
    {
      ARROW_SCOPED_TRACE("selection vector must exist");
      {
        auto if_else = if_else_expr(literal(MakeNullScalar(boolean())),
                                    selection_vector_must_exist(literal(1)), literal(0));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
        ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: There is no selection vector",
                                   ExecuteScalarExpression(bound, batch));
      }
      {
        auto if_else = if_else_expr(literal(MakeNullScalar(boolean())), literal(1),
                                    selection_vector_must_exist(literal(0)));
        ASSERT_OK_AND_ASSIGN(auto bound, if_else.Bind(*schema));
        ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: There is no selection vector",
                                   ExecuteScalarExpression(bound, batch));
      }
    }
    {
      ARROW_SCOPED_TRACE("if (null) then 1 else 0");
      auto expected = ArrayFromJSON(int32(), "[null]");
      {
        ARROW_SCOPED_TRACE("halt selection vector aware");
        auto if_else_sp = if_else_special(literal(MakeNullScalar(boolean())),
                                          halt_selection_vector_aware(literal(1)),
                                          halt_selection_vector_aware(literal(0)));
        AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("halt selection vector unaware");
        auto if_else_sp = if_else_special(literal(MakeNullScalar(boolean())),
                                          halt_selection_vector_unaware(literal(1)),
                                          halt_selection_vector_unaware(literal(0)));
        AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("selection vector mis-aware");
        auto if_else_sp = if_else_special(literal(MakeNullScalar(boolean())),
                                          selection_vector_mis_aware(literal(1)),
                                          selection_vector_mis_aware(literal(0)));
        AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("selection vector must exist");
        {
          ARROW_SCOPED_TRACE("if_else(special)");
          auto if_else_sp = if_else_special(literal(MakeNullScalar(boolean())),
                                            selection_vector_must_exist(literal(1)),
                                            selection_vector_mis_aware(literal(0)));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (true) then 1 else 0");
      auto expected = ArrayFromJSON(int32(), "[1]");
      {
        ARROW_SCOPED_TRACE("halt selection vector aware");
        {
          auto if_else_sp = if_else_special(
              literal(true), halt_selection_vector_aware(literal(1)), literal(0));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                     ExecuteScalarExpression(bound, batch));
        }
        {
          auto if_else_sp = if_else_special(literal(true), literal(1),
                                            halt_selection_vector_aware(literal(0)));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
      }
      {
        ARROW_SCOPED_TRACE("halt selection vector aware");
        {
          auto if_else_sp = if_else_special(
              literal(true), halt_selection_vector_unaware(literal(1)), literal(0));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                     ExecuteScalarExpression(bound, batch));
        }
        {
          auto if_else_sp = if_else_special(literal(true), literal(1),
                                            halt_selection_vector_unaware(literal(0)));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
      }
      {
        ARROW_SCOPED_TRACE("selection vector mis-aware");
        auto if_else_sp =
            if_else_special(literal(true), selection_vector_mis_aware(literal(1)),
                            selection_vector_mis_aware(literal(0)));
        AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("selection vector must exist");
        {
          auto if_else_sp = if_else_special(
              literal(true), selection_vector_must_exist(literal(1)), literal(0));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: There is no selection vector",
                                     ExecuteScalarExpression(bound, batch));
        }
        {
          auto if_else_sp = if_else_special(literal(true), literal(1),
                                            selection_vector_must_exist(literal(0)));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
      }
    }
    {
      ARROW_SCOPED_TRACE("if (false) then 1 else 0");
      auto expected = ArrayFromJSON(int32(), "[0]");
      {
        ARROW_SCOPED_TRACE("halt selection vector aware");
        {
          auto if_else_sp = if_else_special(
              literal(false), halt_selection_vector_aware(literal(1)), literal(0));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
        {
          auto if_else_sp = if_else_special(literal(false), literal(1),
                                            halt_selection_vector_aware(literal(0)));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                     ExecuteScalarExpression(bound, batch));
        }
      }
      {
        ARROW_SCOPED_TRACE("halt selection vector aware");
        {
          auto if_else_sp = if_else_special(
              literal(false), halt_selection_vector_unaware(literal(1)), literal(0));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
        {
          auto if_else_sp = if_else_special(literal(false), literal(1),
                                            halt_selection_vector_unaware(literal(0)));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: Halting",
                                     ExecuteScalarExpression(bound, batch));
        }
      }
      {
        ARROW_SCOPED_TRACE("selection vector mis-aware");
        auto if_else_sp =
            if_else_special(literal(false), selection_vector_mis_aware(literal(1)),
                            selection_vector_mis_aware(literal(0)));
        AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("selection vector must exist");
        {
          auto if_else_sp = if_else_special(
              literal(false), selection_vector_must_exist(literal(1)), literal(0));
          AssertExprEqualIgnoreShape(expected, if_else_sp, schema, batch);
        }
        {
          auto if_else_sp = if_else_special(literal(false), literal(1),
                                            selection_vector_must_exist(literal(0)));
          ASSERT_OK_AND_ASSIGN(auto bound, if_else_sp.Bind(*schema));
          ASSERT_RAISES_WITH_MESSAGE(Invalid, "Invalid: There is no selection vector",
                                     ExecuteScalarExpression(bound, batch));
        }
      }
    }
  }
}

TEST_F(IfElseSpecialFormTest, Shortcut) {
  {
    auto schema = arrow::schema({field("", int32())});
    std::vector<ExecBatch> batches = {
        ExecBatch({*ArrayFromJSON(int32(), "[]")}, 0),
        ExecBatch({*ArrayFromJSON(int32(), "[1]")}, 1),
    };
    for (const auto& batch : batches) {
      {
        ARROW_SCOPED_TRACE("if (null) then 1 else 0");
        AssertIfElseEqualWithExpr(literal(MakeNullScalar(boolean())), literal(1),
                                  literal(0), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then 1 else 0");
        AssertIfElseEqualWithExpr(literal(true), literal(1), literal(0), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then 1 else 0");
        AssertIfElseEqualWithExpr(literal(false), literal(1), literal(0), schema, batch);
      }
    }
  }
  {
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
      {
        ARROW_SCOPED_TRACE("if (null) then a else b");
        AssertIfElseEqualWithExpr(literal(MakeNullScalar(boolean())), field_ref("a"),
                                  field_ref("b"), schema, batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then 0 else b");
        AssertIfElseEqualWithExpr(literal(true), literal(0), field_ref("b"), schema,
                                  batch);
      }
      {
        ARROW_SCOPED_TRACE("if (true) then a else b");
        AssertIfElseEqualWithExpr(literal(true), field_ref("a"), field_ref("b"), schema,
                                  batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else 1");
        AssertIfElseEqualWithExpr(literal(false), field_ref("a"), literal(1), schema,
                                  batch);
      }
      {
        ARROW_SCOPED_TRACE("if (false) then a else b");
        AssertIfElseEqualWithExpr(literal(false), field_ref("a"), field_ref("b"), schema,
                                  batch);
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
        ARROW_SCOPED_TRACE("if (a) then b else c");
        AssertIfElseEqualWithExpr(field_ref("a"), field_ref("b"), field_ref("c"), schema,
                                  batch);
      }
      {
        ARROW_SCOPED_TRACE("if (a) then 1 else 0");
        AssertIfElseEqualWithExpr(field_ref("a"), literal(1), literal(0), schema, batch);
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
