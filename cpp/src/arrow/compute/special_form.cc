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

#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/expression_internal.h"
#include "arrow/compute/registry.h"

namespace arrow::compute {

Result<TypeHolder> IfElseSpecialForm::Resolve(std::vector<Expression>* arguments,
                                              ExecContext* exec_context) const {
  ARROW_ASSIGN_OR_RAISE(auto function,
                        exec_context->func_registry()->GetFunction("if_else"));
  std::vector<TypeHolder> types = GetTypes(*arguments);

  // TODO: DispatchBest and implicit cast.
  ARROW_ASSIGN_OR_RAISE(auto maybe_exact_match, function->DispatchExact(types));
  KernelContext kernel_context(exec_context, maybe_exact_match);
  if (maybe_exact_match->init) {
    const FunctionOptions* options = function->default_options();
    ARROW_ASSIGN_OR_RAISE(
        auto kernel_state,
        maybe_exact_match->init(&kernel_context, {maybe_exact_match, types, options}));
    kernel_context.SetState(kernel_state.get());
  }
  return maybe_exact_match->signature->out_type().Resolve(&kernel_context, types);
}

namespace {

// struct Condition {
//   Expression expr;
// };

// struct Action {
//   Expression expr;
// };

// struct BranchResult {
//   int instruction;
// };

// struct Branch {
//   Condition condition;
//   Action action;
//   bool masked = true;
//   Branch* next = nullptr;

//   Result<BranchResult> Execute(BranchResult last, const ExecBatch& input,
//                                ExecContext* exec_context) const {
//     if (masked) {
//       auto cond = condition.Execute(last.Mask(), input, exec_context);
//       if (cond.Complete()) {
//         if (cond.AllNull()) {
//           auto this_result = MakeNullScalar(action.expr.type()->GetSharedPtr());
//           return MergeResult(last, this_result);
//         } else {
//           auto this_result = action.Execute(input, cond.Mask(), exec_context);
//           return MergeResult(last, this_result);
//         }
//       }
//       if (cond.AllNull()) {
//         auto this_result = MakeNullScalar(action.expr.type()->GetSharedPtr());
//         auto merged = MergeResult(last, this_result);
//         return next->Execute(merged, input, exec_context);
//       } else {
//         auto this_result = action.Execute(input, cond.Mask(), exec_context);
//         auto merged = MergeResult(last, this_result);
//         return next->Execute(merged, input, exec_context);
//       }
//     }

//     auto cond = condition.Execute(last.SelectionVector(), input, exec_context);
//     if (cond.Complete()) {
//       if (cond.AllNull()) {
//         auto this_result = MakeNullScalar(action.expr.type()->GetSharedPtr());
//         return MergeResult(last, this_result);
//       } else {
//         auto this_result = action.Execute(input, cond.SelectionVector(), exec_context);
//         return MergeResult(last, this_result);
//       }
//     }
//     if (cond.AllNull()) {
//       auto this_result = MakeNullScalar(action.expr.type()->GetSharedPtr());
//       return MergeResult(last, this_result);
//     } else {
//       auto this_result = action.Execute(input, cond.SelectionVector(), exec_context);
//       return MergeResult(last, this_result);
//     }
//   }
// };

struct Mask {
  virtual std::shared_ptr<Mask> Not() const = 0;
  virtual std::shared_ptr<Mask> And(const std::shared_ptr<Mask>& other) const = 0;
  virtual Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                              ExecContext* exec_context) const = 0;
};

struct AllTrueMask : public Mask {
  std::shared_ptr<Mask> Not() const override { return std::make_unique<AllFalseMask>(); }
  std::shared_ptr<Mask> And(const std::shared_ptr<Mask>& other) const override {
    return other;
  }
};

struct AllFalseMask : public Mask {
  std::shared_ptr<Mask> Not() const override { return std::make_unique<AllTrueMask>(); }
  std::shared_ptr<Mask> And(const std::shared_ptr<Mask>& other) const override {
    return std::make_shared<AllFalseMask>();
  }
};

struct Branch {};

struct SelVecAwareExecutor {
  Result<Datum> Execute(const std::vector<Branch>& branches, const ExecBatch& input,
                        ExecContext* exec_context) const {
    std::vector<Datum> results;
    std::vector<std::shared_ptr<SelectionVector>> sel_vecs;
    auto non_taken = Mask(input.selection_vector);
    for (const auto& branch : branches) {
      if (non_taken->length() == 0) {
        break;
      }
      auto cond = branch.Condition(non_taken, input, exec_context);
      auto taken = SelVecAnd(non_taken, cond, exec_context);
      if (taken->length() == 0) {
        continue;
      }
      ARROW_ASSIGN_OR_RAISE(auto result, branch.Action(taken, input, exec_context));
      results.push_back(result);
      sel_vecs.push_back(taken);
      non_taken = SelVecAnd(non_taken, SelVecNot(cond, exec_context), exec_context);
    }
    return Choose(results, sel_vecs, input.length, exec_context);
  }
};

// TODO: Take scalar may not work.
Result<ExecBatch> TakeBySelectionVector(const ExecBatch& input,
                                        const Datum& selection_vector,
                                        ExecContext* exec_context) {
  std::vector<Datum> values(input.num_values());
  for (int i = 0; i < input.num_values(); ++i) {
    ARROW_ASSIGN_OR_RAISE(
        values[i], Take(input[i], selection_vector, TakeOptions{/*boundcheck=*/false},
                        exec_context));
  }
  return ExecBatch::Make(std::move(values), selection_vector.length());
}

std::shared_ptr<ChunkedArray> ChunkedArrayFromDatums(const std::vector<Datum>& datums) {
  std::vector<std::shared_ptr<Array>> chunks;
  for (const auto& datum : datums) {
    DCHECK(datum.is_arraylike());
    if (datum.is_array() && datum.length() > 0) {
      chunks.push_back(datum.make_array());
    } else {
      DCHECK(datum.is_chunked_array());
      for (const auto& chunk : datum.chunked_array()->chunks()) {
        if (chunk->length() > 0) {
          chunks.push_back(chunk);
        }
      }
    }
  }
  return std::make_shared<ChunkedArray>(std::move(chunks));
}

bool IsSelectionVectorAwarePathAvailable(const std::vector<Expression>& arguments,
                                         const ExecBatch& input,
                                         ExecContext* exec_context) {
  for (const auto& expr : arguments) {
    if (!expr.selection_vector_aware()) {
      return false;
    }
  }
  for (const auto& value : input.values) {
    if (value.is_scalar()) {
      continue;
    }
    if (value.is_array() && input.length <= exec_context->exec_chunksize()) {
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

Result<Datum> IfElseSpecialForm::Execute(const std::vector<Expression>& arguments,
                                         const ExecBatch& input,
                                         ExecContext* exec_context) const {
  DCHECK_EQ(input.selection_vector, nullptr);
  DCHECK_EQ(arguments.size(), 3);

  const auto& cond_expr = arguments[0];
  DCHECK_EQ(cond_expr.type()->id(), Type::BOOL);
  const auto& if_true_expr = arguments[1];
  const auto& if_false_expr = arguments[2];
  DCHECK_EQ(if_true_expr.type()->id(), if_false_expr.type()->id());

  ARROW_ASSIGN_OR_RAISE(auto cond,
                        ExecuteScalarExpression(cond_expr, input, exec_context));
  DCHECK_EQ(cond.type()->id(), Type::BOOL);

  if (cond.is_scalar()) {
    auto cond_scalar = cond.scalar_as<BooleanScalar>();
    if (!cond_scalar.is_valid) {
      // Always eagerly return minimal "shape" whenever possible because special form has
      // all the power to shortcut the computation. This means we don't really respect the
      // shape of the input (scalar/array/chunked array) as what normal expression
      // evaluation does.
      return MakeNullScalar(if_true_expr.type()->GetSharedPtr());
    } else if (cond_scalar.value) {
      return ExecuteScalarExpression(if_true_expr, input, exec_context);
    } else {
      return ExecuteScalarExpression(if_false_expr, input, exec_context);
    }
  }

  Datum if_true = MakeNullScalar(if_true_expr.type()->GetSharedPtr());
  Datum if_false = MakeNullScalar(if_false_expr.type()->GetSharedPtr());

  if (IsSelectionVectorAwarePathAvailable({if_true_expr, if_false_expr}, input,
                                          exec_context)) {
    DCHECK(cond.is_array());
    auto boolean_cond = cond.array_as<BooleanArray>();
    if (boolean_cond->null_count() == boolean_cond->length()) {
      return MakeArrayOfNull(if_true_expr.type()->GetSharedPtr(), input.length,
                             exec_context->memory_pool());
    }

    if (boolean_cond->true_count() == boolean_cond->length()) {
      return ExecuteScalarExpression(if_true_expr, input, exec_context);
    }

    if (boolean_cond->false_count() == boolean_cond->length()) {
      return ExecuteScalarExpression(if_false_expr, input, exec_context);
    }

    ARROW_ASSIGN_OR_RAISE(auto sel_true, SelectionVector::FromMask(
                                             *boolean_cond, exec_context->memory_pool()));
    if (sel_true->length() > 0) {
      ExecBatch input_true = input;
      input_true.selection_vector = sel_true;
      ARROW_ASSIGN_OR_RAISE(
          if_true, ExecuteScalarExpression(if_true_expr, input_true, exec_context));
    }

    ARROW_ASSIGN_OR_RAISE(auto cond_inverted,
                          CallFunction("invert", {cond}, exec_context));
    DCHECK(cond_inverted.is_array());
    auto boolean_cond_inverted = cond_inverted.array_as<BooleanArray>();
    ARROW_ASSIGN_OR_RAISE(
        auto sel_false,
        SelectionVector::FromMask(*boolean_cond_inverted, exec_context->memory_pool()));
    if (sel_false->length() > 0) {
      ExecBatch input_false = input;
      input_false.selection_vector = sel_false;
      ARROW_ASSIGN_OR_RAISE(
          if_false, ExecuteScalarExpression(if_false_expr, input_false, exec_context));
    }

    return CallFunction("if_else", {cond, if_true, if_false}, exec_context);
  }

  ARROW_ASSIGN_OR_RAISE(auto sel_true,
                        CallFunction("indices_nonzero", {cond}, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto cond_inverted, CallFunction("invert", {cond}, exec_context));
  ARROW_ASSIGN_OR_RAISE(auto sel_false,
                        CallFunction("indices_nonzero", {cond_inverted}, exec_context));

  if (sel_true.length() == 0 && sel_false.length() == 0) {
    return MakeArrayOfNull(if_true_expr.type()->GetSharedPtr(), input.length,
                           exec_context->memory_pool());
  }

  if (sel_true.length() == input.length) {
    return ExecuteScalarExpression(if_true_expr, input, exec_context);
  }

  if (sel_false.length() == input.length) {
    return ExecuteScalarExpression(if_false_expr, input, exec_context);
  }

  if (sel_true.length() > 0) {
    ARROW_ASSIGN_OR_RAISE(auto input_true,
                          TakeBySelectionVector(input, sel_true, exec_context));
    ARROW_ASSIGN_OR_RAISE(
        if_true, ExecuteScalarExpression(if_true_expr, input_true, exec_context));
  }

  if (sel_false.length() > 0) {
    ARROW_ASSIGN_OR_RAISE(auto input_false,
                          TakeBySelectionVector(input, sel_false, exec_context));
    ARROW_ASSIGN_OR_RAISE(
        if_false, ExecuteScalarExpression(if_false_expr, input_false, exec_context));
  }

  auto if_true_false = ChunkedArrayFromDatums({if_true, if_false});
  auto sel = ChunkedArrayFromDatums({sel_true, sel_false});
  ARROW_ASSIGN_OR_RAISE(
      auto result_datum,
      Permute(if_true_false, sel, ScatterOptions{/*max_index=*/input.length - 1}));
  DCHECK(result_datum.is_arraylike());
  if (result_datum.is_chunked_array() &&
      result_datum.chunked_array()->num_chunks() == 1) {
    return result_datum.chunked_array()->chunk(0);
  } else {
    return result_datum;
  }
}

}  // namespace arrow::compute
