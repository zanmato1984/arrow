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

// class IfElseSpecialFormTest : public ::testing::Test {
//  protected:
//   static void SetUpTestSuite() { ASSERT_OK(RegisterAuxilaryFunctions()); }

//  protected:
//   static Status UnreachableExec(KernelContext*, const ExecSpan&, ExecResult*) {
//     return Status::Invalid("Unreachable");
//   }

//   static ArrayKernelSelectiveExec AssertSelection() {
//     return [](KernelContext*, const ExecSpan&, const SelectionVectorSpan&, ExecResult*)
//     {
//       return Status::Invalid("Unreachable");
//     };
//   }

//   static Status IdentityExec(KernelContext*, const ExecSpan& span, ExecResult* out) {
//     DCHECK_EQ(span.num_values(), 1);
//     const auto& arg = span[0];
//     DCHECK(arg.is_array());
//     *out->array_data_mutable() = *arg.array.ToArrayData();
//     return Status::OK();
//   }

//   static Status AssertSelectionVectorNotExistExec(KernelContext* kernel_ctx,
//                                                   const ExecSpan& span, ExecResult*
//                                                   out) {
//     return IdentityExec(kernel_ctx, span, out);
//   }

//   static Status AssertSelectionVectorExistExec(KernelContext* kernel_ctx,
//                                                const ExecSpan& span,
//                                                const SelectionVectorSpan& selection,
//                                                ExecResult* out) {
//     for (int32_t i = 0; i < selection.length(); ++i) {
//       EXPECT_GE(selection[i], 0);
//       EXPECT_LT(selection[i], span.length);
//     }
//     return IdentityExec(kernel_ctx, span, out);
//   }

//   static Status RegisterAuxilaryFunctions() {
//     auto registry = GetFunctionRegistry();

//     {
//       auto register_unreachable_func = [&](const std::string& name) -> Status {
//         auto func =
//             std::make_shared<ScalarFunction>(name, Arity::Unary(),
//             FunctionDoc::Empty());

//         ScalarKernel kernel({InputType::Any()}, internal::FirstType, UnreachableExec);
//         kernel.can_write_into_slices = false;
//         kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
//         kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
//         RETURN_NOT_OK(func->AddKernel(kernel));
//         RETURN_NOT_OK(registry->AddFunction(std::move(func)));
//         return Status::OK();
//       };

//       RETURN_NOT_OK(register_unreachable_func("unreachable"));
//     }

//     {
//       auto register_sv_awareness_func = [&](const std::string& name,
//                                             bool sv_awareness) -> Status {
//         auto func =
//             std::make_shared<ScalarFunction>(name, Arity::Unary(),
//             FunctionDoc::Empty());

//         ScalarKernel kernel({InputType::Any()}, internal::FirstType, IdentityExec);
//         kernel.can_write_into_slices = false;
//         kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
//         kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
//         RETURN_NOT_OK(func->AddKernel(kernel));
//         RETURN_NOT_OK(registry->AddFunction(std::move(func)));
//         return Status::OK();
//       };

//       RETURN_NOT_OK(register_sv_awareness_func("sv_suppress", false));
//     }

//     {
//       auto register_assert_sv_func = [&](const std::string& name,
//                                          bool sv_existence) -> Status {
//         auto func =
//             std::make_shared<ScalarFunction>(name, Arity::Unary(),
//             FunctionDoc::Empty());

//         ArrayKernelExec exec = AssertSelectionVectorNotExistExec;
//         ArrayKernelSelectiveExec selective_exec = nullptr;
//         if (sv_existence) {
//           selective_exec = AssertSelectionVectorExistExec;
//         }
//         ScalarKernel kernel({InputType::Any()}, internal::FirstType, std::move(exec),
//                             std::move(selective_exec),
//                             /*init=*/nullptr);
//         kernel.can_write_into_slices = false;
//         kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
//         kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
//         RETURN_NOT_OK(func->AddKernel(kernel));
//         RETURN_NOT_OK(registry->AddFunction(std::move(func)));
//         return Status::OK();
//       };

//       RETURN_NOT_OK(register_assert_sv_func("assert_sv_exist", /*sv_existence=*/true));
//       RETURN_NOT_OK(register_assert_sv_func("assert_sv_empty",
//       /*sv_existence=*/false));
//     }

//     return Status::OK();
//   }

//   static std::vector<Expression> SuppressSelectionVectorAwareForIfElse(
//       const Expression& cond, const Expression& if_true, const Expression& if_false) {
//     auto suppress_if_else_recursive =
//         [&](const Expression& expr) -> std::vector<Expression> {
//       if (const auto& sp = expr.special(); sp && sp->special_form->name() == "if_else")
//       {
//         const auto& cond = sp->arguments[0];
//         const auto& if_true = sp->arguments[1];
//         const auto& if_false = sp->arguments[2];
//         return SuppressSelectionVectorAwareForIfElse(cond, if_true, if_false);
//       } else {
//         return {expr};
//       }
//     };
//     auto suppressed_conds = suppress_if_else_recursive(cond);
//     auto suppressed_if_trues = suppress_if_else_recursive(if_true);
//     auto suppressed_if_falses = suppress_if_else_recursive(if_false);
//     std::vector<Expression> result;
//     for (const auto& suppressed_cond : suppressed_conds) {
//       for (const auto& suppressed_if_true : suppressed_if_trues) {
//         for (const auto& suppressed_if_false : suppressed_if_falses) {
//           result.emplace_back(
//               if_else_special(suppressed_cond, suppressed_if_true,
//               suppressed_if_false));
//           result.emplace_back(if_else_special(sv_suppress(suppressed_cond),
//                                               suppressed_if_true,
//                                               suppressed_if_false));
//           result.emplace_back(if_else_special(suppressed_cond,
//                                               sv_suppress(suppressed_if_true),
//                                               sv_suppress(suppressed_if_false)));
//         }
//       }
//     }
//     return result;
//   }

//   static void CheckIfElseIgnoreShape(const Expression& cond, const Expression& if_true,
//                                      const Expression& if_false,
//                                      const std::shared_ptr<Schema>& schema,
//                                      const ExecBatch& batch,
//                                      ExecContext* exec_context =
//                                      default_exec_context()) {
//     auto if_else = if_else_regular(cond, if_true, if_false);
//     auto exprs = SuppressSelectionVectorAwareForIfElse(cond, if_true, if_false);
//     AssertExprEqualExprsIgnoreShape(if_else, exprs, schema, batch, exec_context);
//   }
// };

namespace {

Expression if_else(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

Result<Datum> ExecuteExpr(Expression expr, const Schema& schema, const ExecBatch& batch,
                          ExecContext* exec_context = default_exec_context()) {
  ARROW_ASSIGN_OR_RAISE(auto bound, expr.Bind(schema, exec_context));
  return ExecuteScalarExpression(bound, batch, exec_context);
}

// Result<Datum> ExecuteIfElse(Expression cond, Expression if_true, Expression if_false,
//                             const Schema& schema, const ExecBatch& batch,
//                             ExecContext* exec_context = default_exec_context()) {
//   return ExecuteExpr(if_else(std::move(cond), std::move(if_true), std::move(if_false)),
//                      schema, batch, exec_context);
// }

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
      [=](MakeIfElseFunc make_if_else) { return if_else(cond, if_true, if_false); },
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

 protected:
  void DoTestBasic(const std::vector<Expression>& cond_exprs,
                   const std::vector<Expression>& if_true_exprs,
                   const std::vector<Expression>& if_false_exprs,
                   const std::vector<Datum>& boolean_datums,
                   const std::vector<Datum>& int_datums) {
    for (const auto& cond_expr : cond_exprs) {
      for (const auto& if_true_expr : if_true_exprs) {
        for (const auto& if_false_expr : if_false_exprs) {
          ARROW_SCOPED_TRACE(
              "if_else_special: " +
              if_else_special(cond_expr, if_true_expr, if_false_expr).ToString());
          for (const auto& b1_datum : boolean_datums) {
            for (const auto& b2_datum : boolean_datums) {
              for (const auto& i1_datum : int_datums) {
                for (const auto& i2_datum : int_datums) {
                  ExecBatch batch({b1_datum, b2_datum, i1_datum, i2_datum}, length);
                  ARROW_SCOPED_TRACE("batch: " + batch.ToString());
                  CheckIfElseSpecial(cond_expr, if_true_expr, if_false_expr, *schm,
                                     batch);
                }
              }
            }
          }
        }
      }
    }
  }

  void DoTestNestedCond(const std::vector<Expression>& cond_exprs,
                        const std::vector<Expression>& if_true_exprs,
                        const std::vector<Expression>& if_false_exprs,
                        const std::vector<Datum>& boolean_datums,
                        const std::vector<Datum>& int_datums) {
    for (const auto& cond_expr : cond_exprs) {
      for (const auto& if_true_expr : if_true_exprs) {
        for (const auto& if_false_expr : if_false_exprs) {
          ARROW_SCOPED_TRACE(
              "if_else_special: " +
              if_else_special(if_else_special(cond_expr, if_true_expr, if_false_expr),
                              int1, int2)
                  .ToString());
          for (const auto& b1_datum : boolean_datums) {
            for (const auto& b2_datum : boolean_datums) {
              for (const auto& i1_datum : int_datums) {
                for (const auto& i2_datum : int_datums) {
                  ExecBatch batch({b1_datum, b2_datum, i1_datum, i2_datum}, length);
                  ARROW_SCOPED_TRACE("batch: " + batch.ToString());
                  CheckIfElseSpecial(
                      [=](MakeIfElseFunc make_if_else) {
                        return make_if_else(
                            make_if_else(cond_expr, if_true_expr, if_false_expr), int1,
                            int2);
                      },
                      *schm, batch);
                }
              }
            }
          }
        }
      }
    }
  }

  void DoTestNestedCond(const std::vector<Expression>& cond_exprs,
                        const std::vector<Expression>& if_true_exprs,
                        const std::vector<Expression>& if_false_exprs,
                        const ExecBatch& batch) {
    for (const auto& cond_expr : cond_exprs) {
      for (const auto& if_true_expr : if_true_exprs) {
        for (const auto& if_false_expr : if_false_exprs) {
          ARROW_SCOPED_TRACE(
              "if_else_special: " +
              if_else_special(if_else_special(cond_expr, if_true_expr, if_false_expr),
                              int1, int2)
                  .ToString());
          ARROW_SCOPED_TRACE("batch: " + batch.ToString());
          CheckIfElseSpecial(
              [=](MakeIfElseFunc make_if_else) {
                return make_if_else(make_if_else(cond_expr, if_true_expr, if_false_expr),
                                    int1, int2);
              },
              *schm, batch);
        }
      }
    }
  }
};

TEST_F(TestExecuteIfElseSpecial, AllLiterals) {
  DoTestBasic(boolean_literals, int_literals, int_literals, boolean_arrays, int_arrays);
}

TEST_F(TestExecuteIfElseSpecial, AllScalars) {
  DoTestBasic(boolean_fields, int_fields, int_fields, boolean_scalars, int_scalars);
}

TEST_F(TestExecuteIfElseSpecial, FieldWithArrays) {
  DoTestBasic(boolean_fields, int_fields, int_fields, boolean_arrays, int_arrays);
}

TEST_F(TestExecuteIfElseSpecial, ComplexExprsWithArrays) {
  DoTestBasic(boolean_complex_exprs, int_complex_exprs, int_complex_exprs, boolean_arrays,
              int_arrays);
}

TEST_F(TestExecuteIfElseSpecial, NestedCondAllLiterals) {
  DoTestNestedCond(boolean_literals, boolean_literals, boolean_literals, boolean_arrays,
                   int_arrays);
}

TEST_F(TestExecuteIfElseSpecial, NestedCondAllScalars) {
  DoTestNestedCond(boolean_fields, boolean_fields, boolean_fields, boolean_scalars,
                   int_scalars);
}

TEST_F(TestExecuteIfElseSpecial, NestedCondFieldWithArrays) {
  DoTestNestedCond(boolean_fields, boolean_fields, boolean_fields, boolean_arrays,
                   int_arrays);
}

// TEST_F(TestExecuteIfElseSpecial, NarrowDown) {
//   DoTestNestedCond(
//       {boolean1}, {boolean2}, {boolean1},
//       ExecBatch({boolean1_arr, boolean1_chunked, int1_arr, int2_chunked}, length));
// }

// TEST(IfElseSpecial, ExecuteNestedCond) {
//   auto boolean_exprs = {boolean1, boolean2, and_(boolean1, boolean2),
//                         or_(boolean1, boolean2)};
//   for (const auto& cond_inner_expr : boolean_exprs) {
//     for (const auto& if_true_inner_expr : boolean_exprs) {
//       for (const auto& if_false_inner_expr : boolean_exprs) {
//         ARROW_SCOPED_TRACE(
//             "expression: if (" +
//             if_else_special(cond_inner_expr, if_true_inner_expr, if_false_inner_expr)
//                 .ToString() +
//             ") then int1 else int2");
//         for (const auto& boolean1_datum :
//              {Datum(boolean1_arr), Datum(boolean1_chunked)}) {
//           ARROW_SCOPED_TRACE("boolean1: " + boolean1_datum.ToString());
//           for (const auto& boolean2_datum :
//                {Datum(boolean2_arr), Datum(boolean2_chunked)}) {
//             ARROW_SCOPED_TRACE("boolean2: " + boolean2_datum.ToString());
//             for (const auto& int1_datum : {Datum(int1_arr), Datum(int1_chunked)}) {
//               ARROW_SCOPED_TRACE("int1: " + int1_datum.ToString());
//               for (const auto& int2_datum : {Datum(int2_arr), Datum(int2_chunked)}) {
//                 ARROW_SCOPED_TRACE("int2: " + int2_datum.ToString());
//                 CheckIfElseSpecial(
//                     [=](MakeIfElseFunc make_if_else) {
//                       return make_if_else(
//                           make_if_else(cond_inner_expr, if_true_inner_expr,
//                                        if_false_inner_expr),
//                           int1, int2);
//                     },
//                     *schm,
//                     ExecBatch({Datum(boolean1_arr), Datum(boolean2_arr),
//                     Datum(int1_arr),
//                                Datum(int2_arr)},
//                               length));
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// TEST(IfElseSpecial, ExecuteNestedBody) {
//   const int64_t length = 7;

//   auto schm = schema({field("boolean1", boolean()), field("boolean2", boolean()),
//                       field("int1", int32()), field("int2", int32())});
//   auto boolean1 = field_ref("boolean1");
//   auto boolean2 = field_ref("boolean2");
//   auto int1 = field_ref("int1");
//   auto int2 = field_ref("int2");

//   auto boolean1_arr =
//       ArrayFromJSON(boolean(), "[null, true, false, true, false, null, true]");
//   auto boolean1_chunked = ChunkedArrayFromJSON(
//       boolean(), {"[null, true, false]", "[]", "[true, false, null, true]"});

//   auto boolean2_arr =
//       ArrayFromJSON(boolean(), "[true, false, true, true, null, false, true]");
//   auto boolean2_chunked = ChunkedArrayFromJSON(
//       boolean(), {"[true, false]", "[true]", "[true]", "[null, false, true]"});

//   auto int1_arr = ArrayFromJSON(int32(), "[0, 1, 2, 3, 4, 5, 6]");
//   auto int1_chunked = ChunkedArrayFromJSON(int32(), {"[0, 1]", "[2, 3, 4]", "[5,
//   6]"});

//   auto int2_arr = ArrayFromJSON(int32(), "[0, 10, 20, 30, 40, 50, 60]");
//   auto int2_chunked =
//       ChunkedArrayFromJSON(int32(), {"[0]", "[10, 20]", "[30, 40, 50]", "[]",
//       "[60]"});

//   auto boolean_exprs = {boolean1, boolean2, and_(boolean1, boolean2),
//                         or_(boolean1, boolean2)};
//   auto int_exprs = {int1, int2, add(int1, int2)};
//   for (const auto& cond_outer_expr : boolean_exprs) {
//     for (const auto& cond_inner_expr : boolean_exprs) {
//       for (const auto& if_true_inner_expr : int_exprs) {
//         for (const auto& if_false_inner_expr : int_exprs) {
//           for (const auto& if_false_outer_expr : int_exprs) {
//             ARROW_SCOPED_TRACE(
//                 "expression: if (" + cond_outer_expr.ToString() + ") then (" +
//                 if_else_special(cond_inner_expr, if_true_inner_expr,
//                 if_false_inner_expr)
//                     .ToString() +
//                 ") else (" + if_false_outer_expr.ToString() + ")");
//             for (const auto& boolean1_datum :
//                  {Datum(boolean1_arr), Datum(boolean1_chunked)}) {
//               ARROW_SCOPED_TRACE("boolean1: " + boolean1_datum.ToString());
//               for (const auto& boolean2_datum :
//                    {Datum(boolean2_arr), Datum(boolean2_chunked)}) {
//                 ARROW_SCOPED_TRACE("boolean2: " + boolean2_datum.ToString());
//                 for (const auto& int1_datum : {Datum(int1_arr), Datum(int1_chunked)})
//                 {
//                   ARROW_SCOPED_TRACE("int1: " + int1_datum.ToString());
//                   for (const auto& int2_datum : {Datum(int2_arr),
//                   Datum(int2_chunked)})
//                   {
//                     ARROW_SCOPED_TRACE("int2: " + int2_datum.ToString());
//                     CheckIfElseSpecial(
//                         [=](MakeIfElseFunc make_if_else) {
//                           return make_if_else(
//                               cond_outer_expr,
//                               make_if_else(cond_inner_expr, if_true_inner_expr,
//                                            if_false_inner_expr),
//                               if_false_outer_expr);
//                         },
//                         *schm,
//                         ExecBatch({Datum(boolean1_arr), Datum(boolean2_arr),
//                                    Datum(int1_arr), Datum(int2_arr)},
//                                   length));
//                   }
//                 }
//               }
//             }
//           }

//           for (const auto& if_true_outer_expr : int_exprs) {
//             ARROW_SCOPED_TRACE(
//                 "expression: if (" + cond_outer_expr.ToString() + ") then (" +
//                 if_true_outer_expr.ToString() + ") else (" +
//                 if_else_special(cond_inner_expr, if_true_inner_expr,
//                 if_false_inner_expr)
//                     .ToString() +
//                 ")");
//             for (const auto& boolean1_datum :
//                  {Datum(boolean1_arr), Datum(boolean1_chunked)}) {
//               ARROW_SCOPED_TRACE("boolean1: " + boolean1_datum.ToString());
//               for (const auto& boolean2_datum :
//                    {Datum(boolean2_arr), Datum(boolean2_chunked)}) {
//                 ARROW_SCOPED_TRACE("boolean2: " + boolean2_datum.ToString());
//                 for (const auto& int1_datum : {Datum(int1_arr), Datum(int1_chunked)})
//                 {
//                   ARROW_SCOPED_TRACE("int1: " + int1_datum.ToString());
//                   for (const auto& int2_datum : {Datum(int2_arr),
//                   Datum(int2_chunked)})
//                   {
//                     ARROW_SCOPED_TRACE("int2: " + int2_datum.ToString());
//                     CheckIfElseSpecial(
//                         [=](MakeIfElseFunc make_if_else) {
//                           return make_if_else(
//                               cond_outer_expr, if_true_outer_expr,
//                               make_if_else(cond_inner_expr, if_true_inner_expr,
//                                            if_false_inner_expr));
//                         },
//                         *schm,
//                         ExecBatch({Datum(boolean1_arr), Datum(boolean2_arr),
//                                    Datum(int1_arr), Datum(int2_arr)},
//                                   length));
//                   }
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// TEST(IfElseSpecial, ExecuteNested) {
//   // if (if (boolean1) then (boolean2) else (int1 == int2))) then if ()
//   // if (if (boolean1) then (boolean2) else (int1 == int2)) then
//   //   if (boolean1) then (int1) else (int1 + int2)
//   // else
//   //   if (boolean2) then (int2) else (int1 + int2)
//   const int64_t length = 7;

//   auto schm = schema({field("boolean1", boolean()), field("boolean2", boolean()),
//                       field("int1", int32()), field("int2", int32())});
//   auto boolean1 = field_ref("boolean1");
//   auto boolean2 = field_ref("boolean2");
//   auto int1 = field_ref("int1");
//   auto int2 = field_ref("int2");

//   auto boolean1_arr =
//       ArrayFromJSON(boolean(), "[null, true, false, true, false, null, true]");
//   auto boolean1_chunked = ChunkedArrayFromJSON(
//       boolean(), {"[null, true, false]", "[]", "[true, false, null, true]"});

//   auto boolean2_arr =
//       ArrayFromJSON(boolean(), "[true, false, true, true, null, false, true]");
//   auto boolean2_chunked = ChunkedArrayFromJSON(
//       boolean(), {"[true, false]", "[true]", "[true]", "[null, false, true]"});

//   auto int1_arr = ArrayFromJSON(int32(), "[0, 1, 2, 3, 4, 5, 6]");
//   auto int1_chunked = ChunkedArrayFromJSON(int32(), {"[0, 1]", "[2, 3, 4]", "[5,
//   6]"});

//   auto int2_arr = ArrayFromJSON(int32(), "[0, 10, 20, 30, 40, 50, 60]");
//   auto int2_chunked =
//       ChunkedArrayFromJSON(int32(), {"[0]", "[10, 20]", "[30, 40, 50]", "[]",
//       "[60]"});

//   for (const auto& cond_inner_expr :
//        {literal(MakeNullScalar(boolean())), literal(true), literal(false), boolean1,
//         boolean2, equal(int1, int2)}) {
//     for (const auto& if_true_inner_expr :
//          {literal(MakeNullScalar(int32())), literal(42),
//           literal(std::numeric_limits<int64_t>::max()), int1, int2, add(int1, int2)})
//           {
//       for (const auto& if_false_inner_expr :
//            {literal(MakeNullScalar(int32())), literal(24),
//             literal(std::numeric_limits<int64_t>::min()), int2, add(b, c)}) {
//         ARROW_SCOPED_TRACE(
//             "expression: " +
//             if_else_special(cond_expr, if_true_expr, if_false_expr).ToString());
//         for (const auto& cond_datum : {Datum(a_arr), Datum(a_chunked)}) {
//           ARROW_SCOPED_TRACE("cond: " + cond_datum.ToString());
//           for (const auto& if_true_datum : {Datum(b_arr), Datum(b_chunked)}) {
//             ARROW_SCOPED_TRACE("if_true: " + if_true_datum.ToString());
//             for (const auto& if_false_datum : {Datum(c_arr), Datum(c_chunked)}) {
//               ARROW_SCOPED_TRACE("if_false: " + if_false_datum.ToString());
//               CheckIfElseSpecial(
//                   cond_expr, if_true_expr, if_false_expr, *schm,
//                   ExecBatch({cond_datum, if_true_datum, if_false_datum}, length));
//             }
//           }
//         }
//       }
//     }
//   }
// }

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

// namespace {

// auto kCanonicalSchema = arrow::schema({field("a", boolean()), field("b", int32())});

// auto kCanonicalA = field_ref("a");
// auto kCanonicalB = field_ref("b");

// const auto kCanonicalBooleanCols = {kBooleanNull, literal(true), literal(false),
//                                     kCanonicalA};
// const auto kCanonicalIntCols = {kIntNull, literal(42), kCanonicalB};

// const std::vector<ExecBatch> kCanonicalBatches = {
//     ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
//         [null, 0],
//         [null, null],
//         [null, 1]
//       ])")),
//     ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
//         [true, 0],
//         [true, null],
//         [true, 1]
//       ])")),
//     ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
//         [false, 0],
//         [false, null],
//         [false, 1]
//       ])")),
//     ExecBatch(*RecordBatchFromJSON(kCanonicalSchema, R"([
//         [false, 0],
//         [true, null],
//         [null, 1]
//       ])")),
// };

// }  // namespace

// TEST_F(IfElseSpecialFormTest, Simple) {
//   const auto& schema = kCanonicalSchema;
//   const auto& boolean_datums = kCanonicalBooleanCols;
//   const auto& int_datums = kCanonicalIntCols;
//   const auto& batches = kCanonicalBatches;
//   for (const auto& cond : boolean_datums) {
//     for (const auto& if_true : int_datums) {
//       for (const auto& if_false : int_datums) {
//         for (const auto& batch : batches) {
//           CheckIfElseIgnoreShape(cond, if_true, if_false, schema, batch);
//         }
//       }
//     }
//   }
// }

// TEST_F(IfElseSpecialFormTest, NestedSimple) {
//   const auto& schema = kCanonicalSchema;
//   const auto& a = kCanonicalA;
//   const auto& b = kCanonicalB;
//   ExecBatch batch(*RecordBatchFromJSON(kCanonicalSchema, R"([
//       [false, 0],
//       [true, null],
//       [null, 1]
//     ])"));
//   for (const auto& cond : {
//            if_else_special(a, kBooleanNull, a),
//            if_else_special(a, a, literal(true)),
//        }) {
//     for (const auto& if_true : {
//              if_else_special(a, kIntNull, b),
//              if_else_special(a, b, literal(42)),
//          }) {
//       for (const auto& if_false : {
//                if_else_special(a, kIntNull, b),
//                if_else_special(a, b, literal(42)),
//            }) {
//         CheckIfElseIgnoreShape(cond, if_true, if_false, schema, batch);
//       }
//     }
//   }
// }

// // TODO: Deprecate this test due to slowness.
// TEST_F(IfElseSpecialFormTest, NestedConditionComplex) {
//   const auto& batches = kCanonicalBatches;
//   const auto& schema = kCanonicalSchema;
//   const auto& boolean_datums = kCanonicalBooleanCols;
//   const auto& int_datums = kCanonicalIntCols;
//   for (const auto& nested_cond : boolean_datums) {
//     for (const auto& nested_if_true : boolean_datums) {
//       for (const auto& nested_if_false : boolean_datums) {
//         auto nested_if_else_sp =
//             if_else_special(nested_cond, nested_if_true, nested_if_false);
//         for (const auto& if_true : int_datums) {
//           for (const auto& if_false : int_datums) {
//             for (const auto& batch : batches) {
//               CheckIfElseIgnoreShape(nested_if_else_sp, if_true, if_false, schema,
//               batch);
//             }
//           }
//         }
//       }
//     }
//   }
// }

// // TODO: Deprecate this test due to slowness.
// TEST_F(IfElseSpecialFormTest, NestedBodyComplex) {
//   const auto& batches = kCanonicalBatches;
//   const auto& schema = kCanonicalSchema;
//   const auto& boolean_datums = kCanonicalBooleanCols;
//   const auto& int_datums = kCanonicalIntCols;
//   for (const auto& cond : boolean_datums) {
//     for (const auto& nested_cond : boolean_datums) {
//       for (const auto& nested_if_true : int_datums) {
//         for (const auto& nested_if_false : int_datums) {
//           auto nested_if_else_sp =
//               if_else_special(nested_cond, nested_if_true, nested_if_false);
//           for (const auto& batch : batches) {
//             CheckIfElseIgnoreShape(cond, nested_if_else_sp, nested_if_else_sp,
//             schema,
//                                    batch);
//           }
//         }
//       }
//     }
//   }
// }

// // TODO: Deprecate this test due to slowness.
// TEST_F(IfElseSpecialFormTest, NestedComplex) {
//   const auto& batches = kCanonicalBatches;
//   const auto& schema = kCanonicalSchema;
//   const auto& boolean_datums = kCanonicalBooleanCols;
//   const auto& int_datums = kCanonicalIntCols;
//   for (const auto& cond_nested_cond : boolean_datums) {
//     for (const auto& cond_nested_if_true : boolean_datums) {
//       for (const auto& cond_nested_if_false : boolean_datums) {
//         auto cond =
//             if_else_special(cond_nested_cond, cond_nested_if_true,
//             cond_nested_if_false);
//         for (const auto& nested_cond : boolean_datums) {
//           for (const auto& nested_if_true : int_datums) {
//             for (const auto& nested_if_false : int_datums) {
//               auto nested_if_else_sp =
//                   if_else_special(nested_cond, nested_if_true, nested_if_false);
//               for (const auto& batch : batches) {
//                 CheckIfElseIgnoreShape(cond, nested_if_else_sp, nested_if_else_sp,
//                 schema,
//                                        batch);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// // TODO: ChunkedArray.

}  // namespace arrow::compute
