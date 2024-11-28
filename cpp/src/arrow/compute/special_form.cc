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

#include "arrow/array/builder_primitive.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/expression_internal.h"
#include "arrow/compute/registry.h"

namespace arrow::compute {

namespace {

// TODO: Clean free functions.

struct BodyMask;

struct BranchMask : public std::enable_shared_from_this<BranchMask> {
  virtual ~BranchMask() = default;

  Result<std::shared_ptr<const BodyMask>> ApplyCondSparse(
      const Expression& expr, const ExecBatch& input, ExecContext* exec_context) const {
    ARROW_ASSIGN_OR_RAISE(auto datum, ApplySparse(expr, input, exec_context));
    return MakeBodyMaskFromSparseDatum(datum, exec_context);
  }

  Result<std::shared_ptr<const BodyMask>> ApplyCondDense(
      const Expression& expr, const ExecBatch& input, ExecContext* exec_context) const {
    ARROW_ASSIGN_OR_RAISE(auto datum, ApplyDense(expr, input, exec_context));
    return MakeBodyMaskFromDenseDatum(datum, exec_context);
  }

  virtual bool empty() const = 0;

 protected:
  virtual Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                                    ExecContext* exec_context) const = 0;

  virtual Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                                   ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromScalar(
      const BooleanScalar& scalar, ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromSparseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDenseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const = 0;

 private:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromSparseDatum(
      const Datum& datum, ExecContext* exec_context) const {
    DCHECK(datum.type()->id() == Type::BOOL);
    if (datum.is_scalar()) {
      return MakeBodyMaskFromScalar(datum.scalar_as<BooleanScalar>(), exec_context);
    }
    DCHECK(datum.is_array());
    return MakeBodyMaskFromSparseBitmap(datum.array_as<BooleanArray>(), exec_context);
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDenseDatum(
      const Datum& datum, ExecContext* exec_context) const {
    DCHECK(datum.type()->id() == Type::BOOL);
    if (datum.is_scalar()) {
      return MakeBodyMaskFromScalar(datum.scalar_as<BooleanScalar>(), exec_context);
    }
    DCHECK(datum.is_array());
    return MakeBodyMaskFromDenseBitmap(datum.array_as<BooleanArray>(), exec_context);
  }

  friend struct NestedBodyMask;
};

struct BodyMask : public std::enable_shared_from_this<BodyMask> {
  virtual ~BodyMask() = default;

  virtual bool empty() const = 0;

  virtual Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                                    ExecContext* exec_context) const = 0;

  virtual Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                                   ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  virtual Result<std::shared_ptr<const BranchMask>> NextBranchMask() const = 0;
};

struct AllPassBranchMask : public BranchMask {
  explicit AllPassBranchMask(int64_t length) : length_(length) {}

  bool empty() const override { return false; }

 protected:
  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK_EQ(input.length, length_);
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = nullptr;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    DCHECK_EQ(input.length, length_);
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = nullptr;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return nullptr;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromScalar(
      const BooleanScalar& scalar, ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromSparseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDenseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

 private:
  int64_t length_;
};

struct AllFailBranchMask : public BranchMask {
  AllFailBranchMask() = default;

  bool empty() const override { return true; }

 protected:
  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::ApplySparse should not be called");
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::ApplyDense should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromScalar(
      const BooleanScalar& scalar, ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::MakeBodyMask should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromSparseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid(
        "AllFailBranchMask::MakeBodyMaskFromSparseBitmap should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDenseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid(
        "AllFailBranchMask::MakeBodyMaskFromDenseBitmap should not be called");
  }
};

struct NestedBodyMask : public BodyMask {
  explicit NestedBodyMask(std::shared_ptr<const BranchMask> branch_mask)
      : branch_mask_(std::move(branch_mask)) {}

 protected:
  Result<Datum> DelegateApplySparse(const Expression& expr, const ExecBatch& input,
                                    ExecContext* exec_context) const {
    return branch_mask_->ApplySparse(expr, input, exec_context);
  }

  Result<Datum> DelegateApplyDense(const Expression& expr, const ExecBatch& input,
                                   ExecContext* exec_context) const {
    return branch_mask_->ApplyDense(expr, input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> DelegateGetSelectionVector() const {
    return branch_mask_->GetSelectionVector();
  }

 protected:
  std::shared_ptr<const BranchMask> branch_mask_;
};

struct AllNullBodyMask : public NestedBodyMask {
  using NestedBodyMask::NestedBodyMask;

  bool empty() const override { return true; }

  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::ApplySparse should not be called");
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::ApplyDense should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

struct AllPassBodyMask : public NestedBodyMask {
  using NestedBodyMask::NestedBodyMask;

  bool empty() const override { return false; }

  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    return DelegateApplySparse(expr, input, exec_context);
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    return DelegateApplyDense(expr, input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return DelegateGetSelectionVector();
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

struct AllFailBodyMask : public NestedBodyMask {
  using NestedBodyMask::NestedBodyMask;

  bool empty() const override { return true; }

  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::ApplySparse should not be called");
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::ApplyDense should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return branch_mask_;
  }
};

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

struct ConditionalBranchMask : public BranchMask {
  ConditionalBranchMask(std::shared_ptr<SelectionVector> selection_vector, int64_t length)
      : selection_vector_(std::move(selection_vector)), length_(length) {}

  bool empty() const override { return selection_vector_->length() == 0; }

 protected:
  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    auto sparse_input = input;
    sparse_input.selection_vector = selection_vector_;
    return ExecuteScalarExpression(expr, sparse_input, exec_context);
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(
        auto dense_input,
        TakeBySelectionVector(input, *selection_vector_->data(), exec_context));
    return ExecuteScalarExpression(expr, dense_input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return selection_vector_;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromScalar(
      const BooleanScalar& scalar, ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromSparseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDenseBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

 protected:
  std::shared_ptr<SelectionVector> selection_vector_ = nullptr;
  int64_t length_ = 0;
};

struct ConditionalBodyMask : public BodyMask {
  ConditionalBodyMask(std::shared_ptr<SelectionVector> body,
                      std::shared_ptr<SelectionVector> rest, int64_t length)
      : body_(std::move(body)), rest_(std::move(rest)), length_(length) {}

  bool empty() const override { return body_->length() == 0; }

  Result<Datum> ApplySparse(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    auto sparse_input = input;
    sparse_input.selection_vector = body_;
    return ExecuteScalarExpression(expr, sparse_input, exec_context);
  }

  Result<Datum> ApplyDense(const Expression& expr, const ExecBatch& input,
                           ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto dense_input,
                          TakeBySelectionVector(input, *body_->data(), exec_context));
    return ExecuteScalarExpression(expr, dense_input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return body_;
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    if (!rest_) {
      return std::make_shared<AllFailBranchMask>();
    }
    return std::make_shared<ConditionalBranchMask>(rest_, length_);
  }

 private:
  std::shared_ptr<SelectionVector> body_;
  std::shared_ptr<SelectionVector> rest_;
  int64_t length_;
};

Result<std::shared_ptr<BodyMask>> BodyMaskFromScalar(
    const BooleanScalar& scalar, std::shared_ptr<const BranchMask> branch_mask,
    ExecContext* exec_context) {
  if (!scalar.is_valid) {
    return std::make_shared<AllNullBodyMask>(std::move(branch_mask));
  } else if (scalar.value) {
    return std::make_shared<AllPassBodyMask>(std::move(branch_mask));
  } else {
    return std::make_shared<AllFailBodyMask>(std::move(branch_mask));
  }
}

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromScalar(
    const BooleanScalar& scalar, ExecContext* exec_context) const {
  return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
}

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromSparseBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(rest_builder.Reserve(length_));

  ArraySpan span(*bitmap->data());
  int32_t i = 0;
  VisitArraySpanInline<BooleanType>(
      span,
      [&](bool mask) {
        if (mask) {
          body_builder.UnsafeAppend(i);
        } else {
          rest_builder.UnsafeAppend(i);
        }
        ++i;
      },
      [&]() { ++i; });

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto rest_arr, rest_builder.Finish());
  auto body = std::make_shared<SelectionVector>(body_arr->data());
  auto rest = std::make_shared<SelectionVector>(rest_arr->data());
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(rest), length_);
}

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromDenseBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  return MakeBodyMaskFromSparseBitmap(bitmap, exec_context);
}

Result<std::shared_ptr<const BodyMask>> ConditionalBranchMask::MakeBodyMaskFromScalar(
    const BooleanScalar& scalar, ExecContext* exec_context) const {
  return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
}

Result<std::shared_ptr<const BodyMask>>
ConditionalBranchMask::MakeBodyMaskFromSparseBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(rest_builder.Reserve(length_));

  for (int i = 0; i < selection_vector_->length(); ++i) {
    auto index = selection_vector_->indices()[i];
    if (!bitmap->IsNull(index)) {
      if (bitmap->Value(index)) {
        body_builder.UnsafeAppend(index);
      } else {
        rest_builder.UnsafeAppend(index);
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto rest_arr, rest_builder.Finish());
  auto body = std::make_shared<SelectionVector>(body_arr->data());
  auto rest = std::make_shared<SelectionVector>(rest_arr->data());
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(rest), length_);
}

Result<std::shared_ptr<const BodyMask>>
ConditionalBranchMask::MakeBodyMaskFromDenseBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), selection_vector_->length());

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(bitmap->true_count()));
  RETURN_NOT_OK(rest_builder.Reserve(bitmap->false_count()));

  for (int i = 0; i < selection_vector_->length(); ++i) {
    if (!bitmap->IsNull(i)) {
      if (bitmap->Value(i)) {
        body_builder.UnsafeAppend(selection_vector_->indices()[i]);
      } else {
        rest_builder.UnsafeAppend(selection_vector_->indices()[i]);
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto rest_arr, rest_builder.Finish());
  auto body = std::make_shared<SelectionVector>(body_arr->data());
  auto rest = std::make_shared<SelectionVector>(rest_arr->data());
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(rest), length_);
}

struct Branch {
  Expression cond;
  Expression body;
};

template <typename Impl>
struct ConditionalExecutor {
  ConditionalExecutor(const std::vector<Branch>& branches, const TypeHolder& result_type)
      : branches(branches), result_type(result_type) {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(ConditionalExecutor);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ConditionalExecutor);

  Result<Datum> Execute(const ExecBatch& input, ExecContext* exec_context) const&& {
    DCHECK(!branches.empty());

    BranchResults results;
    results.Reserve(branches.size());
    ARROW_ASSIGN_OR_RAISE(auto branch_mask, InitBranchMask(input, exec_context));
    for (const auto& branch : branches) {
      if (branch_mask->empty()) {
        break;
      }
      ARROW_ASSIGN_OR_RAISE(auto body_mask,
                            ApplyCond(branch_mask, branch.cond, input, exec_context));
      if (body_mask->empty()) {
        ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask());
        continue;
      }
      ARROW_ASSIGN_OR_RAISE(auto body_result,
                            static_cast<const Impl*>(this)->ApplyBody(
                                body_mask, branch.body, input, exec_context));
      DCHECK(body_result.type()->Equals(*result_type));
      ARROW_ASSIGN_OR_RAISE(auto selection_vector, body_mask->GetSelectionVector());
      results.Emplace(std::move(body_result), std::move(selection_vector));
      ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask());
    }
    return static_cast<const Impl*>(this)->MultiplexBranchResults(input, results,
                                                                  exec_context);
  }

 protected:
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

 private:
  Result<std::shared_ptr<const BranchMask>> InitBranchMask(
      const ExecBatch& input, ExecContext* exec_context) const {
    if (input.selection_vector) {
      return std::make_shared<ConditionalBranchMask>(input.selection_vector,
                                                     input.length);
    }
    return std::make_shared<AllPassBranchMask>(input.length);
  }

  Result<std::shared_ptr<const BodyMask>> ApplyCond(
      const std::shared_ptr<const BranchMask>& branch_mask, const Expression& cond,
      const ExecBatch& input, ExecContext* exec_context) const {
    if (cond.selection_vector_aware()) {
      return branch_mask->ApplyCondSparse(cond, input, exec_context);
    }
    return branch_mask->ApplyCondDense(cond, input, exec_context);
  }

 protected:
  const std::vector<Branch>& branches;
  const TypeHolder& result_type;
};

struct SparseConditionalExecutor : public ConditionalExecutor<SparseConditionalExecutor> {
  using ConditionalExecutor<SparseConditionalExecutor>::ConditionalExecutor;

  Result<Datum> ApplyBody(const std::shared_ptr<const BodyMask>& body_mask,
                          const Expression& body, const ExecBatch& input,
                          ExecContext* exec_context) const {
    return body_mask->ApplySparse(body, input, exec_context);
  }

  Result<Datum> MultiplexBranchResults(const ExecBatch& input,
                                       const BranchResults& results,
                                       ExecContext* exec_context) const {
    if (results.empty()) {
      return MakeArrayOfNull(result_type.GetSharedPtr(), input.length,
                             exec_context->memory_pool());
    }

    if (results.size() == 1) {
      if (const auto& result = results.body_results()[0];
          results.selection_vectors()[0] == nullptr ||
          results.selection_vectors()[0]->length() ==
              (input.selection_vector ? input.selection_vector->length()
                                      : input.length)) {
        return result;
      }
    }

    std::vector<Datum> choose_args;
    choose_args.reserve(results.size() + 1);
    ARROW_ASSIGN_OR_RAISE(auto indices, ChooseIndices(results.selection_vectors(),
                                                      input.length, exec_context));
    choose_args.emplace_back(std::move(indices));
    choose_args.insert(choose_args.end(), results.body_results().begin(),
                       results.body_results().end());
    return CallFunction("choose", choose_args, exec_context);
  }

 private:
  Result<Datum> ChooseIndices(
      const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors,
      int64_t length, ExecContext* exec_context) const {
    const int64_t validity_bytes = bit_util::BytesForBits(length);
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<ResizableBuffer> validity_buf,
        AllocateResizableBuffer(validity_bytes, exec_context->memory_pool()));
    auto validity_data = validity_buf->mutable_data_as<uint8_t>();
    std::memset(validity_data, 0, validity_bytes);

    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<ResizableBuffer> indices_buf,
        AllocateResizableBuffer(length * sizeof(int32_t), exec_context->memory_pool()));
    auto indices_data = indices_buf->mutable_data_as<int32_t>();
    for (int32_t index = 0; index < static_cast<int32_t>(selection_vectors.size());
         ++index) {
      DCHECK_NE(selection_vectors[index], nullptr);
      DCHECK_GT(selection_vectors[index]->length(), 0);
      auto row_ids = selection_vectors[index]->indices();
      for (int32_t i = 0; i < selection_vectors[index]->length(); ++i) {
        const int32_t row_id = row_ids[i];
        DCHECK_EQ(bit_util::GetBit(validity_data, row_id), false);
        bit_util::SetBitTo(validity_data, row_id, true);
        indices_data[row_id] = index;
      }
    }

    return ArrayData::Make(int32(), length,
                           {std::move(validity_buf), std::move(indices_buf)});
  }
};

struct DenseConditionalExecutor : public ConditionalExecutor<DenseConditionalExecutor> {
  using ConditionalExecutor<DenseConditionalExecutor>::ConditionalExecutor;

  Result<Datum> ApplyBody(const std::shared_ptr<const BodyMask>& body_mask,
                          const Expression& body, const ExecBatch& input,
                          ExecContext* exec_context) const {
    return body_mask->ApplyDense(body, input, exec_context);
  }

  Result<Datum> MultiplexBranchResults(const ExecBatch& input,
                                       const BranchResults& results,
                                       ExecContext* exec_context) const {
    if (results.empty()) {
      return MakeArrayOfNull(result_type.GetSharedPtr(), input.length,
                             exec_context->memory_pool());
    }

    if (results.size() == 1) {
      if (!results.selection_vectors()[0] ||
          results.selection_vectors()[0]->length() == input.length) {
        return results.body_results()[0];
      }
    }

    ARROW_ASSIGN_OR_RAISE(
        auto body_results,
        ToChunkedArray(results,
                       [&](const Datum& value,
                           const std::shared_ptr<SelectionVector>& selection_vector,
                           ArrayVector& chunks) -> Status {
                         DCHECK_NE(selection_vector, nullptr);
                         DCHECK_GT(selection_vector->length(), 0);
                         DCHECK(value.is_scalar() || value.is_arraylike());
                         if (value.is_scalar()) {
                           ARROW_ASSIGN_OR_RAISE(
                               auto arr, MakeArrayFromScalar(
                                             *value.scalar(), selection_vector->length(),
                                             exec_context->memory_pool()));
                           chunks.push_back(std::move(arr));
                         } else if (value.is_array() && value.length() > 0) {
                           chunks.push_back(value.make_array());
                         } else {
                           DCHECK(value.is_chunked_array());
                           for (const auto& chunk : value.chunked_array()->chunks()) {
                             if (chunk->length() > 0) {
                               chunks.push_back(chunk);
                             }
                           }
                         }
                         return Status::OK();
                       }));
    ARROW_ASSIGN_OR_RAISE(
        auto indices,
        ToChunkedArray(
            results,
            [](const Datum&, const std::shared_ptr<SelectionVector>& selection_vector,
               ArrayVector& chunks) -> Status {
              DCHECK_NE(selection_vector, nullptr);
              DCHECK_GT(selection_vector->length(), 0);
              chunks.push_back(MakeArray(selection_vector->data()));
              return Status::OK();
            }));
    ARROW_ASSIGN_OR_RAISE(
        auto result,
        Scatter(body_results, indices, ScatterOptions{/*max_index=*/input.length - 1}));
    DCHECK(result.is_arraylike());
    if (result.is_chunked_array() && result.chunked_array()->num_chunks() == 1) {
      return result.chunked_array()->chunk(0);
    } else {
      return result;
    }
  }

 private:
  template <typename ChunkFunc>
  Result<std::shared_ptr<ChunkedArray>> ToChunkedArray(const BranchResults& results,
                                                       ChunkFunc&& chunk_func) const {
    ArrayVector chunks;
    chunks.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      RETURN_NOT_OK(
          chunk_func(results.body_results()[i], results.selection_vectors()[i], chunks));
    }
    return std::make_shared<ChunkedArray>(std::move(chunks));
  }
};

bool IsSelectionVectorAwarePathAvailable(const ExecBatch& input,
                                         ExecContext* exec_context) {
  return std::all_of(input.values.begin(), input.values.end(),
                     [exceeds_chunksize = input.length > exec_context->exec_chunksize()](
                         const Datum& value) {
                       return value.is_scalar() ||
                              (value.is_array() && !exceeds_chunksize);
                     });
}

class IfElseSpecialExec : public SpecialExec {
 public:
  explicit IfElseSpecialExec(std::shared_ptr<Function> function, const Kernel* kernel)
      : function(std::move(function)), kernel(kernel), branches(2) {}

  Result<TypeHolder> Bind(const std::vector<Expression>& arguments,
                          ExecContext* exec_context) override {
    DCHECK(std::all_of(arguments.begin(), arguments.end(),
                       [](const Expression& argument) { return argument.IsBound(); }));

    DCHECK_EQ(arguments.size(), 3);
    const auto& cond = arguments[0];
    const auto& if_true = arguments[1];
    const auto& if_false = arguments[2];

    {
      auto types = GetTypes(arguments);
      KernelContext kernel_context(exec_context, kernel);
      std::unique_ptr<KernelState> kernel_state;
      if (kernel->init) {
        const FunctionOptions* options = function->default_options();
        ARROW_ASSIGN_OR_RAISE(kernel_state,
                              kernel->init(&kernel_context, {kernel, types, options}));
        kernel_context.SetState(kernel_state.get());
      }
      ARROW_ASSIGN_OR_RAISE(
          type, kernel->signature->out_type().Resolve(&kernel_context, types));
    }

    DCHECK_EQ(cond.type()->id(), Type::BOOL);
    DCHECK_EQ(type, *if_true.type());
    DCHECK_EQ(type, *if_false.type());

    all_bodies_selection_vector_aware =
        if_true.selection_vector_aware() && if_false.selection_vector_aware();

    branches[0] = {cond, if_true};
    branches[1] = {literal(true), if_false};

    return type;
  }

  Result<Datum> Execute(const ExecBatch& input,
                        ExecContext* exec_context) const override {
    if (all_bodies_selection_vector_aware &&
        IsSelectionVectorAwarePathAvailable(input, exec_context)) {
      return SparseConditionalExecutor(branches, type).Execute(input, exec_context);
    } else {
      return DenseConditionalExecutor(branches, type).Execute(input, exec_context);
    }
  }

 private:
  // For Bind.
  std::shared_ptr<Function> function;
  const Kernel* kernel;

  // Post-bind, for Execute.
  TypeHolder type;
  bool all_bodies_selection_vector_aware;
  std::vector<Branch> branches;
};

class IfElseSpecialForm : public SpecialForm {
 public:
  IfElseSpecialForm()
      : SpecialForm(/*name=*/"if_else", /*selection_vector_aware=*/true) {}

  Result<std::unique_ptr<SpecialExec>> DispatchExact(
      const std::vector<TypeHolder>& types, ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto function,
                          exec_context->func_registry()->GetFunction(name));
    ARROW_ASSIGN_OR_RAISE(auto kernel, function->DispatchExact(types));
    return std::make_unique<IfElseSpecialExec>(std::move(function), kernel);
  }

  Result<std::unique_ptr<SpecialExec>> DispatchBest(
      std::vector<TypeHolder>* types, ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto function,
                          exec_context->func_registry()->GetFunction(name));
    ARROW_ASSIGN_OR_RAISE(auto kernel, function->DispatchBest(types));
    return std::make_unique<IfElseSpecialExec>(std::move(function), kernel);
  }
};

}  // namespace

std::shared_ptr<SpecialForm> GetIfElseSpecialForm() {
  static auto instance = std::make_shared<IfElseSpecialForm>();
  return instance;
}

}  // namespace arrow::compute
