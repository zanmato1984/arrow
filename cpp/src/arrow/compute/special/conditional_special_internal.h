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

#pragma once

#include <memory>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/special/special_form_internal.h"
#include "arrow/compute/visibility.h"
#include "arrow/util/unreachable.h"

namespace arrow::compute::internal {

/// Structures to model masks for branching expressions, mostly for efficient
/// short-circuiting of branching chains. The whole abstraction is as follows:
/// - A branch represents a compound expression of condition and body;
/// - A branch mask represents the set of rows (in an ExecBatch) to be evaluated for the
///   branch condition;
/// - When a branch mask is applied to a condition expression, it produces a body mask;
/// - A body mask represents the set of rows (in an ExecBatch) to be evaluated for the
///   branch body.
/// - A body mask also preserves information from its originating branch mask and derives
///   the next branch mask, representing the set of rows remaining to be evaluated for the
///   next branch in the chain.
///
/// For example, consider the following conditional special form:
///   if_else_sp(/*cond=*/eq(a, 'x'), /*if_true=*/foo(b), /*if_false=*/bar(c))
/// is being evaluated on an ExecBatch:
///   [a: ['x', 'y', 'x', 'z'], b: ['b0', 'b1', 'b2', 'b3'], c: ['c0', 'c1', 'c2', 'c3']]
/// We'll have an initial branch mask that passes all rows:
///   BranchMask0: [0, 1, 2, 3] // all rows in the batch
/// (In practice we can have a specialized branch mask implementation that doesn't
/// necessarily store all the row indices when all rows are to be evaluated.)
/// Then BranchMask0 is applied to the condition eq(a, 'x'), producing the condition
/// result:
///   [true, false, true, false]
/// Which is then used by BranchMask0 to make a body mask:
///   BodyMask0: [0, 2] // rows with true condition out of [0, 1, 2, 3]
/// It is then applied to the first branch body foo(b), producing the result for this
/// branch:
///   [foo('b0'), foo('b2')] // at rows [0, 2]
/// After that, BodyMask0 produces the next branch mask:
///   BranchMask1: [1, 3] // [0, 1, 2, 3] - [0, 2]
/// Which is then applied to the next branch condition, which is an implicit true literal
/// in this case, producing the condition result:
///   [true, true] // at rows [1, 3]
/// Which is then used by BranchMask1 to make a body mask:
///   BodyMask1: [1, 3] // rows with true condition out of [1, 3]
/// It is then applied to the second branch body bar(c), producing the result for this
/// branch:
///   [bar('c1'), bar('c3')] // at rows [1, 3]
/// Finally, the results from all branches are combined to produce the final result:
///   [foo('b0'), bar('c1'), foo('b2'), bar('c3')]

struct ARROW_COMPUTE_EXPORT BodyMask;

/// @brief A mask representing the set of rows to be evaluated for a branch condition.
/// Being empty indicates that the entire branch, in addition to all the subsequent
/// branches, are concluded. Otherwise, a selection vector can be obtained to evaluate the
/// branch condition, whose result will be used to further produce a body mask.
struct ARROW_COMPUTE_EXPORT BranchMask : public std::enable_shared_from_this<BranchMask> {
  virtual ~BranchMask() = default;

  /// @brief Check if the branch mask is empty, in which case no rows are to be evaluated.
  virtual bool empty() const = 0;

  /// @brief Get the selection vector representing the rows to be evaluated. Null
  /// indicates that all rows are to be evaluated.
  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  /// @brief Create a body mask for this branch from the given datum, which is the result
  /// of evaluating the condition under this branch mask. All possible trivial cases, such
  /// as constant true/false/null and all-true/false/null arrays, are handled here. If
  /// none of the trivial cases apply, the call is forwarded to the concrete
  /// implementations of MakeBodyMaskFromBitmap().
  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(const Datum& datum,
                                                       ExecContext* exec_context) const;

  /// @brief Create a branch mask from the given selection vector. Based on the content of
  /// the selection vector, it may return concrete branch mask implementations that can
  /// take advantage of short-circuiting.
  static Result<std::shared_ptr<const BranchMask>> FromSelectionVector(
      std::shared_ptr<SelectionVector> selection, int64_t length);

 protected:
  /// @brief Create a body mask from the given bitmap, which is the result of evaluating
  /// the condition under this branch mask.
  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const = 0;

  /// @brief Create a body mask from the given chunked bitmap, which is the result of
  /// evaluating the condition under this branch mask.
  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const = 0;
};

/// @brief A branch mask that evaluates the condition for all rows. The selection vector
/// obtained is null, indicating that all rows are to be evaluated. And the body mask
/// produced is solely based on the condition result.
struct ARROW_COMPUTE_EXPORT AllPassBranchMask : public BranchMask {
  explicit AllPassBranchMask(int64_t length) : length_(length) {}

  bool empty() const override { return length_ == 0; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return NULLPTR;
  }

 protected:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap,
      ExecContext* exec_context) const override;

 private:
  int64_t length_;
};

/// @brief A branch mask that evaluates the condition for no rows and concludes
/// all the subsequent branches. One should never try to obtain a selection vector or
/// produce a body mask from this branch mask.
struct ARROW_COMPUTE_EXPORT AllFailBranchMask : public BranchMask {
  AllFailBranchMask() = default;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    Unreachable("AllFailBranchMask::GetSelectionVector should not be called");
  }

 protected:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override {
    Unreachable("AllFailBranchMask::MakeBodyMaskFromBitmap should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap,
      ExecContext* exec_context) const override {
    Unreachable("AllFailBranchMask::MakeBodyMaskFromBitmap should not be called");
  }
};

/// @brief A branch mask that evaluates the condition for rows indicated by the given
/// selection vector, which is also the one obtained from it. The body mask produced, if
/// no short-circuiting available, is one that with a selection vector that is
/// conceptually AND-ing the branch mask's selection vector and the condition result.
struct ARROW_COMPUTE_EXPORT ConditionalBranchMask : public BranchMask {
  ConditionalBranchMask(std::shared_ptr<SelectionVector> selection_vector, int64_t length)
      : selection_vector_(std::move(selection_vector)), length_(length) {
#ifndef NDEBUG
    DCHECK_OK(selection_vector_->Validate(length_));
#endif
  }

  bool empty() const override { return selection_vector_->length() == 0; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return selection_vector_;
  }

 protected:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap,
      ExecContext* exec_context) const override;

 protected:
  std::shared_ptr<SelectionVector> selection_vector_ = nullptr;
  int64_t length_ = 0;
};

/// @brief A mask representing the set of rows to be evaluated for a branch body. Being
/// empty indicates that no rows are to be evaluated for the body. Otherwise, a selection
/// vector can be obtained to evaluate the branch body, whose result will be used as part
/// of the final result. In addition, a branch mask is produced for the next branch.
struct ARROW_COMPUTE_EXPORT BodyMask : public std::enable_shared_from_this<BodyMask> {
  virtual ~BodyMask() = default;

  /// @brief Check if the body mask is empty, in which case no rows are to be evaluated.
  virtual bool empty() const = 0;

  /// @brief Get the selection vector representing the rows to be evaluated. Null
  /// indicates that all rows are to be evaluated.
  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  /// @brief Create a branch mask for the next branch. May return concrete branch mask
  /// implementations for short-circuiting.
  virtual Result<std::shared_ptr<const BranchMask>> NextBranchMask() const = 0;
};

/// @brief A body mask that emits nulls for all rows and produces an all-fail branch mask
/// for the next branch. For example, the body mask for branch:
///  [... else] if (null) ...
// XXX Only works for null policy of intersection (any operands null -> null). Other
// variants may needed for different null policies.
struct ARROW_COMPUTE_EXPORT AllNullBodyMask : public BodyMask {
  AllNullBodyMask() = default;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    Unreachable("AllNullBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

/// @brief A body mask that delegates certain operations to an underlying branch mask.
/// Subclasses can override behaviors as needed.
struct ARROW_COMPUTE_EXPORT DelegateBodyMask : public BodyMask {
  explicit DelegateBodyMask(std::shared_ptr<const BranchMask> branch_mask)
      : branch_mask_(std::move(branch_mask)) {}

 protected:
  std::shared_ptr<const BranchMask> branch_mask_;
};

/// @brief A body mask that evaluates the body for all rows indicated by the underlying
/// branch mask, and produces an all-fail branch mask for the next branch. For example,
/// the body mask for branch:
///   [... else] if (true) ...
struct ARROW_COMPUTE_EXPORT AllPassBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return branch_mask_->empty(); }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return branch_mask_->GetSelectionVector();
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

/// @brief A body mask that evaluates the body for no rows, and pass through the
/// underlying branch mask for the next branch. For example, the body mask for branch:
///   [... else] if (false) ...
struct ARROW_COMPUTE_EXPORT AllFailBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    Unreachable("AllFailBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return branch_mask_;
  }
};

/// @brief A body mask that evaluates the body for rows indicated by the given selection
/// vector, and produces a branch mask for the next branch from the remainder rows.
struct ARROW_COMPUTE_EXPORT ConditionalBodyMask : public BodyMask {
  ConditionalBodyMask(std::shared_ptr<SelectionVector> body,
                      std::shared_ptr<SelectionVector> remainder, int64_t length)
      : body_(std::move(body)), remainder_(std::move(remainder)), length_(length) {
#ifndef NDEBUG
    DCHECK_OK(body_->Validate(length_));
    DCHECK_OK(remainder_->Validate(length_));
#endif
  }

  bool empty() const override { return body_->length() == 0; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return body_;
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return BranchMask::FromSelectionVector(remainder_, length_);
  }

 private:
  std::shared_ptr<SelectionVector> body_;
  std::shared_ptr<SelectionVector> remainder_;
  int64_t length_;
};

struct ARROW_COMPUTE_EXPORT Branch {
  Expression cond;
  Expression body;
};

/// @brief A simple structure that assembles the process of executing a sequence of
/// branches, including:
/// - Iterating all branches by:
///   - Evaluating each branch condition under the branch mask;
///   - Producing the body mask from the condition result;
///   - Evaluating the branch body under the body mask;
///   - Producing the next branch mask from the body mask.
/// - Collecting all branch body results and selection vectors, and multiplexing them into
///   the final result.
struct ARROW_COMPUTE_EXPORT ConditionalExec {
  ConditionalExec(const std::vector<Branch>& branches, const TypeHolder& result_type)
      : branches(branches), result_type(result_type) {}

  Result<Datum> Execute(const ExecBatch& input, ExecContext* exec_context) const&&;

 private:
  /// @brief A simple helper structure to collect branch body results and their
  /// corresponding selection vectors.
  struct BranchResults {
    void Reserve(int64_t size) {
      body_results_.reserve(size);
      selection_vectors_.reserve(size);
    }

    void Emplace(Datum body_result, std::shared_ptr<SelectionVector> selection_vector) {
      body_results_.emplace_back(std::move(body_result));
      selection_vectors_.emplace_back(std::move(selection_vector));
    }

    bool empty() const { return body_results_.empty(); }

    size_t size() const { return body_results_.size(); }

    const std::vector<Datum>& body_results() const { return body_results_; }

    const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors() const {
      return selection_vectors_;
    }

   private:
    std::vector<Datum> body_results_;
    std::vector<std::shared_ptr<SelectionVector>> selection_vectors_;
  };

  /// @brief Get the initial branch mask based on the existence of the selection vector in
  /// the given ExecBatch.
  ///
  /// If a selection vector exists in the input batch, it implies that we are under a
  /// masked execution, e.g., within another outer conditional special form. In this case,
  /// the initial selection vector should be respected by all the branches, and thus
  /// treated as the initial branch mask and propagated to the rest.
  Result<std::shared_ptr<const BranchMask>> InitBranchMask(
      const ExecBatch& input, ExecContext* exec_context) const {
    if (input.selection_vector) {
      return BranchMask::FromSelectionVector(input.selection_vector, input.length);
    }
    return std::make_shared<AllPassBranchMask>(input.length);
  }

  Result<std::shared_ptr<const BodyMask>> EvaluateCond(
      const std::shared_ptr<const BranchMask>& branch_mask, const Expression& cond,
      const ExecBatch& input, ExecContext* exec_context) const {
    auto input_with_selection = input;
    ARROW_ASSIGN_OR_RAISE(input_with_selection.selection_vector,
                          branch_mask->GetSelectionVector());
    ARROW_ASSIGN_OR_RAISE(
        auto datum, ExecuteScalarExpression(cond, input_with_selection, exec_context));
    return branch_mask->MakeBodyMask(datum, exec_context);
  }

  Result<Datum> EvaluateBody(const std::shared_ptr<const BodyMask>& body_mask,
                             const Expression& body, const ExecBatch& input,
                             ExecContext* exec_context) const {
    auto input_with_selection = input;
    ARROW_ASSIGN_OR_RAISE(input_with_selection.selection_vector,
                          body_mask->GetSelectionVector());
    return ExecuteScalarExpression(body, input_with_selection, exec_context);
  }

  /// @brief Multiplex all branch body results into the final result based on their
  /// corresponding selection vectors.
  Result<Datum> MultiplexResults(const ExecBatch& input, const BranchResults& results,
                                 ExecContext* exec_context) const;

  /// @brief A helper function to choose indices from multiple selection vectors.
  Result<Datum> ChooseIndices(
      const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors,
      int64_t length, ExecContext* exec_context) const;

 private:
  const std::vector<Branch>& branches;
  const TypeHolder& result_type;
};

class ARROW_COMPUTE_EXPORT ConditionalSpecialExecutor : public SpecialExecutor {
 public:
  ConditionalSpecialExecutor(std::vector<Branch> branches, TypeHolder out_type)
      : SpecialExecutor(std::move(out_type), /*options=*/nullptr),
        branches(std::move(branches)) {}

  Result<Datum> Execute(const ExecBatch& input,
                        ExecContext* exec_context) const override {
    return ConditionalExec(branches, out_type_).Execute(input, exec_context);
  }

 private:
  std::vector<Branch> branches;
};

template <typename Impl>
class ConditionalSpecialForm
    : public FunctionBackedSpecialForm<ConditionalSpecialForm<Impl>> {
 public:
  using FunctionBackedSpecialForm<
      ConditionalSpecialForm<Impl>>::FunctionBackedSpecialForm;

  ARROW_DISALLOW_COPY_AND_ASSIGN(ConditionalSpecialForm);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ConditionalSpecialForm);

 protected:
  Result<std::unique_ptr<SpecialExecutor>> BindWithBoundCall(
      Expression::Call call, ExecContext* exec_context) const {
    // Shouldn't have options. This is guaranteed by the call binding.
    DCHECK_EQ(call.options, nullptr);

    auto branches =
        static_cast<const Impl*>(this)->GetBranches(std::move(call.arguments));
    return std::make_unique<ConditionalSpecialExecutor>(std::move(branches),
                                                        std::move(call.type));
  }

  friend class FunctionBackedSpecialForm<ConditionalSpecialForm<Impl>>;
};

}  // namespace arrow::compute::internal
