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

Result<Datum> ExecuteIfElse(Expression cond, Expression if_true, Expression if_false,
                            const Schema& schema, const ExecBatch& batch,
                            ExecContext* exec_context = default_exec_context()) {
  return ExecuteExpr(if_else(std::move(cond), std::move(if_true), std::move(if_false)),
                     schema, batch, exec_context);
}

Result<Datum> ExecuteIfElseSpecial(Expression cond, Expression if_true,
                                   Expression if_false, const Schema& schema,
                                   const ExecBatch& batch,
                                   ExecContext* exec_context = default_exec_context()) {
  return ExecuteExpr(
      if_else_special(std::move(cond), std::move(if_true), std::move(if_false)), schema,
      batch, exec_context);
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

void CheckIfElseSpecial(Expression cond, Expression if_true, Expression if_false,
                        const Schema& schema, const ExecBatch& batch,
                        ExecContext* exec_context = default_exec_context()) {
  ASSERT_OK_AND_ASSIGN(
      auto expected, ExecuteIfElse(cond, if_true, if_false, schema, batch, exec_context));
  ASSERT_OK_AND_ASSIGN(auto result, ExecuteIfElseSpecial(cond, if_true, if_false, schema,
                                                         batch, exec_context));
  AssertDatumsEqualIgnoreShape(expected, result);
}

}  // namespace

// GH-XXXXX:
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

TEST(IfElseSpecial, ExecuteScalar) {
  for (const auto& cond_expr :
       {literal(MakeNullScalar(boolean())), literal(true), literal(false)}) {
    std::vector<std::vector<Expression>> literals_list{
        {literal(MakeNullScalar(int32())), literal(0), literal(1),
         literal(std::numeric_limits<int64_t>::min()),
         literal(std::numeric_limits<int64_t>::max())},
        {literal(MakeNullScalar(utf8())), literal(""), literal("foo"), literal("bar")}};
    for (const auto& literals : literals_list) {
      for (const auto& if_true_expr : literals) {
        for (const auto& if_false_expr : literals) {
          for (int64_t length : {1, 2}) {
            CheckIfElseSpecial(cond_expr, if_true_expr, if_false_expr, *schema({}),
                               ExecBatch({}, length));
          }
        }
      }
    }
  }
}

TEST(IfElseSpecial, ExecuteBasic) {
  const int64_t length = 7;

  auto schm = schema(
      {field("cond", boolean()), field("if_true", int32()), field("if_false", int32())});

  auto cond_arr =
      ArrayFromJSON(boolean(), "[null, true, false, true, false, null, true]");
  auto cond_chunked = ChunkedArrayFromJSON(
      boolean(), {"[null, true, false]", "[]", "[true, false, null, true]"});

  auto if_true_arr = ArrayFromJSON(int32(), "[1, 2, 3, 4, 5, 6, 7]");
  auto if_true_chunked = ChunkedArrayFromJSON(int32(), {"[1, 2]", "[3, 4, 5]", "[6, 7]"});

  auto if_false_arr = ArrayFromJSON(int32(), "[10, 20, 30, 40, 50, 60, 70]");
  auto if_false_chunked =
      ChunkedArrayFromJSON(int32(), {"[10, 20, 30]", "[]", "[40, 50, 60, 70]"});

  for (const auto& cond_expr : {literal(MakeNullScalar(boolean())), literal(true),
                                literal(false), field_ref("cond")}) {
    for (const auto& if_true_expr :
         {literal(MakeNullScalar(int32())), literal(42),
          literal(std::numeric_limits<int64_t>::max()), field_ref("if_true")}) {
      for (const auto& if_false_expr :
           {literal(MakeNullScalar(int32())), literal(24),
            literal(std::numeric_limits<int64_t>::min()), field_ref("if_false")}) {
        ARROW_SCOPED_TRACE(
            "expression: " +
            if_else_special(cond_expr, if_true_expr, if_false_expr).ToString());

        for (const auto& cond_datum : {Datum(cond_arr), Datum(cond_chunked)}) {
          ARROW_SCOPED_TRACE("cond: " + cond_datum.ToString());
          for (const auto& if_true_datum : {Datum(if_true_arr), Datum(if_true_chunked)}) {
            ARROW_SCOPED_TRACE("if_true: " + if_true_datum.ToString());
            for (const auto& if_false_datum :
                 {Datum(if_false_arr), Datum(if_false_chunked)}) {
              ARROW_SCOPED_TRACE("if_false: " + if_false_datum.ToString());
              CheckIfElseSpecial(
                  cond_expr, if_true_expr, if_false_expr, *schm,
                  ExecBatch({cond_datum, if_true_datum, if_false_datum}, length));
            }
          }
        }
      }
    }
  }
}

TEST(IfElseSpecial, ExecuteBasicDebug) {
  const int64_t length = 7;

  auto schm = schema(
      {field("cond", boolean()), field("if_true", int32()), field("if_false", int32())});

  auto cond_arr =
      ArrayFromJSON(boolean(), "[null, true, false, true, false, null, true]");
  // auto cond_chunked = ChunkedArrayFromJSON(
  //     boolean(), {"[null, true, false]", "[]", "[true, false, null, true]"});

  // auto if_true_arr = ArrayFromJSON(int32(), "[1, 2, 3, 4, 5, 6, 7]");
  auto if_true_chunked = ChunkedArrayFromJSON(int32(), {"[1, 2]", "[3, 4, 5]", "[6, 7]"});

  auto if_false_arr = ArrayFromJSON(int32(), "[10, 20, 30, 40, 50, 60, 70]");
  // auto if_false_chunked =
  //     ChunkedArrayFromJSON(int32(), {"[10, 20, 30]", "[]", "[40, 50, 60, 70]"});

  for (const auto& cond_expr : {field_ref("cond")}) {
    for (const auto& if_true_expr : {field_ref("if_true")}) {
      for (const auto& if_false_expr : {literal(24)}) {
        ARROW_SCOPED_TRACE(
            "expression: " +
            if_else_special(cond_expr, if_true_expr, if_false_expr).ToString());

        for (const auto& cond_datum : {Datum(cond_arr)}) {
          ARROW_SCOPED_TRACE("cond: " + cond_datum.ToString());
          for (const auto& if_true_datum : {Datum(if_true_chunked)}) {
            ARROW_SCOPED_TRACE("if_true: " + if_true_datum.ToString());
            for (const auto& if_false_datum : {Datum(if_false_arr)}) {
              ARROW_SCOPED_TRACE("if_false: " + if_false_datum.ToString());
              CheckIfElseSpecial(
                  cond_expr, if_true_expr, if_false_expr, *schm,
                  ExecBatch({cond_datum, if_true_datum, if_false_datum}, length));
            }
          }
        }
      }
    }
  }
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
