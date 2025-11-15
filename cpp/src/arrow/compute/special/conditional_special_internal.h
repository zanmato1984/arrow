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

namespace arrow::compute::internal {

/// Structures to model masks for branching expressions, mostly for efficient
/// short-circuiting of branching chains. The whole abstraction is as follows:
/// - A branch represents a compound expression of condition and body;
/// - A branch mask represents the set of rows (in an ExecBatch) to be evaluated for the
///   branch condition;
/// - When a branch mask is applied to a condition expression, it produces a body mask;
/// - A body mask represents the set of rows (in an ExecBatch) to be evaluated for the
///   branch body.
/// - A body mask also produces the next branch mask, representing the set of rows
///   remaining to be evaluated for the next branch in the chain.

struct ARROW_COMPUTE_EXPORT BodyMask;

struct ARROW_COMPUTE_EXPORT BranchMask : public std::enable_shared_from_this<BranchMask> {
  virtual ~BranchMask() = default;

  virtual bool empty() const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(const Datum& datum,
                                                       ExecContext* exec_context) const;

  static Result<std::shared_ptr<const BranchMask>> FromSelectionVector(
      std::shared_ptr<SelectionVector> selection, int64_t length);

 protected:
  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const = 0;
};

struct ARROW_COMPUTE_EXPORT AllPassBranchMask : public BranchMask {
  explicit AllPassBranchMask(int64_t length) : length_(length) {}

  bool empty() const override { return length_ == 0; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override;

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

struct ARROW_COMPUTE_EXPORT AllFailBranchMask : public BranchMask {
  AllFailBranchMask() = default;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::GetSelectionVector should not be called");
  }

 protected:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid(
        "AllFailBranchMask::MakeBodyMaskFromBitmap should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap,
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid(
        "AllFailBranchMask::MakeBodyMaskFromBitmap should not be called");
  }
};

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

struct ARROW_COMPUTE_EXPORT BodyMask : public std::enable_shared_from_this<BodyMask> {
  virtual ~BodyMask() = default;

  virtual bool empty() const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  virtual Result<std::shared_ptr<const BranchMask>> NextBranchMask() const = 0;
};

struct ARROW_COMPUTE_EXPORT AllNullBodyMask : public BodyMask {
  AllNullBodyMask() = default;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

struct ARROW_COMPUTE_EXPORT DelegateBodyMask : public BodyMask {
  explicit DelegateBodyMask(std::shared_ptr<const BranchMask> branch_mask)
      : branch_mask_(std::move(branch_mask)) {}

 protected:
  std::shared_ptr<const BranchMask> branch_mask_;
};

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

struct ARROW_COMPUTE_EXPORT AllFailBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return true; }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return branch_mask_;
  }
};

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

struct ARROW_COMPUTE_EXPORT ConditionalExec {
  ConditionalExec(const std::vector<Branch>& branches, const TypeHolder& result_type)
      : branches(branches), result_type(result_type) {}

  Result<Datum> Execute(const ExecBatch& input, ExecContext* exec_context) const&&;

 private:
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

  Result<Datum> MultiplexResults(const ExecBatch& input, const BranchResults& results,
                                 ExecContext* exec_context) const;

  Result<Datum> ChooseIndices(
      const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors,
      int64_t length, ExecContext* exec_context) const;

 protected:
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
