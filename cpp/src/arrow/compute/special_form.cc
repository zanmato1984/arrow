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

Result<TypeHolder> IfElseSpecialForm::Resolve(std::vector<Expression>* arguments,
                                              ExecContext* exec_context) const {
  ARROW_ASSIGN_OR_RAISE(auto function,
                        exec_context->func_registry()->GetFunction("if_else"));
  std::vector<TypeHolder> types = GetTypes(*arguments);

  // TODO: Resolve choose/scatter function.

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

// TODO: Consider evaluating cond (sparsely VS. densely) independently of the body.

struct BodyMask;

struct BranchMask : public std::enable_shared_from_this<BranchMask> {
  virtual ~BranchMask() = default;

  virtual Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                              ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const = 0;

  virtual bool empty() const = 0;

 protected:
  virtual Status Init(ExecContext* exec_context) { return Status::OK(); }
};

template <typename T, typename... Args>
std::shared_ptr<T> MakeShared(Args&&... args) {
  struct EnableMakeShared : public T {
    explicit EnableMakeShared(Args&&... args) : T(std::forward<Args>(args)...) {}
  };
  return std::make_shared<EnableMakeShared>(std::forward<Args>(args)...);
}

struct SparseAllPassBranchMask : public BranchMask {
  static Result<std::shared_ptr<SparseAllPassBranchMask>> Make(
      int64_t length, ExecContext* exec_context) {
    auto branch_mask = MakeShared<SparseAllPassBranchMask>(length);
    RETURN_NOT_OK(branch_mask->Init(exec_context));
    return branch_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    DCHECK_EQ(input.length, length_);
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = nullptr;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return nullptr;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const override;

  bool empty() const override { return false; }

 protected:
  explicit SparseAllPassBranchMask(int64_t length) : length_(length) {}

 private:
  int64_t length_;
};

struct AllFailBranchMask : public BranchMask {
  static Result<std::shared_ptr<AllFailBranchMask>> Make(ExecContext* exec_context) {
    static auto branch_mask =
        [&exec_context]() -> Result<std::shared_ptr<AllFailBranchMask>> {
      auto branch_mask = MakeShared<AllFailBranchMask>();
      RETURN_NOT_OK(branch_mask->Init(exec_context));
      return branch_mask;
    }();
    return branch_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::Apply should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::MakeBodyMask should not be called");
  }

  bool empty() const override { return true; }

 protected:
  AllFailBranchMask() = default;
};

struct BodyMask : public std::enable_shared_from_this<BodyMask> {
  virtual ~BodyMask() = default;

  virtual Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                              ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const = 0;

  virtual bool empty() const = 0;

 protected:
  virtual Status Init(ExecContext* exec_context) { return Status::OK(); }
};

template <typename Impl>
struct TrivialBodyMask : public BodyMask {
  static Result<std::shared_ptr<Impl>> Make(std::shared_ptr<const BranchMask> branch_mask,
                                            ExecContext* exec_context) {
    auto body_mask = MakeShared<Impl>(std::move(branch_mask));
    RETURN_NOT_OK(body_mask->Init(exec_context));
    return body_mask;
  }

 protected:
  explicit TrivialBodyMask(std::shared_ptr<const BranchMask> branch_mask)
      : branch_mask_(std::move(branch_mask)) {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(TrivialBodyMask);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(TrivialBodyMask);

 protected:
  std::shared_ptr<const BranchMask> branch_mask_;
};

struct AllNullBodyMask : public TrivialBodyMask<AllNullBodyMask> {
  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::Apply should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const override {
    return AllFailBranchMask::Make(exec_context);
  }

  bool empty() const override { return true; }

 protected:
  using TrivialBodyMask::TrivialBodyMask;
};

struct AllPassBodyMask : public TrivialBodyMask<AllPassBodyMask> {
  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    return branch_mask_->Apply(expr, input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return branch_mask_->GetSelectionVector(exec_context);
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const override {
    return AllFailBranchMask::Make(exec_context);
  }

  bool empty() const override { return false; }

 protected:
  using TrivialBodyMask::TrivialBodyMask;
};

struct AllFailBodyMask : public TrivialBodyMask<AllFailBodyMask> {
  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::Apply should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const override {
    return branch_mask_;
  }

  bool empty() const override { return true; }

 protected:
  using TrivialBodyMask::TrivialBodyMask;
};

struct SparseBranchMask : public BranchMask {
  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = selection_vector_;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return selection_vector_;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const override;

  bool empty() const override { return selection_vector_->length() == 0; }

 protected:
  SparseBranchMask(std::shared_ptr<BooleanArray> bitmap) : bitmap_(std::move(bitmap)) {}

  SparseBranchMask(std::shared_ptr<SelectionVector> selection_vector)
      : selection_vector_(std::move(selection_vector)) {}

 protected:
  std::shared_ptr<BooleanArray> bitmap_ = nullptr;
  std::shared_ptr<SelectionVector> selection_vector_ = nullptr;
};

struct SparseBitmapBranchMask : public SparseBranchMask {
  static Result<std::shared_ptr<SparseBitmapBranchMask>> Make(
      std::shared_ptr<BooleanArray> bitmap, ExecContext* exec_context) {
    auto branch_mask = MakeShared<SparseBitmapBranchMask>(std::move(bitmap));
    RETURN_NOT_OK(branch_mask->Init(exec_context));
    return branch_mask;
  }

 protected:
  explicit SparseBitmapBranchMask(std::shared_ptr<BooleanArray> bitmap)
      : SparseBranchMask(std::move(bitmap)) {}

  Status Init(ExecContext* exec_context) override {
    ARROW_ASSIGN_OR_RAISE(selection_vector_, SelectionVector::FromMask(
                                                 *bitmap_, exec_context->memory_pool()));
    return Status::OK();
  }
};

struct SparseSelectionVectorBranchMask : public SparseBranchMask {
  static Result<std::shared_ptr<SparseSelectionVectorBranchMask>> Make(
      std::shared_ptr<SelectionVector> selection_vector, int64_t length,
      ExecContext* exec_context) {
    auto branch_mask =
        MakeShared<SparseSelectionVectorBranchMask>(std::move(selection_vector), length);
    RETURN_NOT_OK(branch_mask->Init(exec_context));
    return branch_mask;
  }

 protected:
  SparseSelectionVectorBranchMask(std::shared_ptr<SelectionVector> selection_vector,
                                  int64_t length)
      : SparseBranchMask(std::move(selection_vector)), length_(length) {}

  Status Init(ExecContext* exec_context) override {
    ARROW_ASSIGN_OR_RAISE(
        bitmap_, selection_vector_->ToMask(length_, exec_context->memory_pool()));
    return Status::OK();
  }

 private:
  int64_t length_;
};

struct SparseBodyMask : public BodyMask {
  static Result<std::shared_ptr<SparseBodyMask>> Make(
      std::shared_ptr<BooleanArray> cond_bitmap,
      std::shared_ptr<BooleanArray> body_bitmap, ExecContext* exec_context) {
    DCHECK_EQ(cond_bitmap->length(), body_bitmap->length());
    ARROW_ASSIGN_OR_RAISE(auto datum, And(*cond_bitmap, *body_bitmap, exec_context));
    auto bitmap = datum.array_as<BooleanArray>();
    return Make(std::move(bitmap), exec_context);
  }

  static Result<std::shared_ptr<SparseBodyMask>> Make(
      std::shared_ptr<BooleanArray> bitmap, ExecContext* exec_context) {
    auto body_mask = MakeShared<SparseBodyMask>(std::move(bitmap));
    RETURN_NOT_OK(body_mask->Init(exec_context));
    return body_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = selection_vector_;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return selection_vector_;
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto datum, Invert(*bitmap_, exec_context));
    auto inverted = datum.array_as<BooleanArray>();
    return SparseBitmapBranchMask::Make(std::move(inverted), exec_context);
  }

  bool empty() const override { return bitmap_->true_count() == 0; }

 protected:
  SparseBodyMask(std::shared_ptr<BooleanArray> bitmap) : bitmap_(std::move(bitmap)) {}

  Status Init(ExecContext* exec_context) override {
    ARROW_ASSIGN_OR_RAISE(selection_vector_, SelectionVector::FromMask(
                                                 *bitmap_, exec_context->memory_pool()));
    return Status::OK();
  }

 private:
  std::shared_ptr<BooleanArray> bitmap_ = nullptr;
  std::shared_ptr<SelectionVector> selection_vector_ = nullptr;
};

Result<std::shared_ptr<BodyMask>> BodyMaskFromScalar(
    const BooleanScalar& scalar, std::shared_ptr<const BranchMask> branch_mask,
    ExecContext* exec_context) {
  if (!scalar.is_valid) {
    return AllNullBodyMask::Make(std::move(branch_mask), exec_context);
  } else if (scalar.value) {
    return AllPassBodyMask::Make(std::move(branch_mask), exec_context);
  } else {
    return AllFailBodyMask::Make(std::move(branch_mask), exec_context);
  }
}

Result<std::shared_ptr<const BodyMask>> SparseAllPassBranchMask::MakeBodyMask(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK_EQ(datum.type()->id(), Type::BOOL);
  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
  }
  auto bitmap = datum.array_as<BooleanArray>();
  return SparseBodyMask::Make(std::move(bitmap), exec_context);
}

Result<std::shared_ptr<const BodyMask>> SparseBranchMask::MakeBodyMask(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK_EQ(datum.type()->id(), Type::BOOL);
  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
  }
  auto body_bitmap = datum.array_as<BooleanArray>();
  return SparseBodyMask::Make(bitmap_, std::move(body_bitmap), exec_context);
}

struct DenseAllPassBranchMask : public BranchMask {
  static Result<std::shared_ptr<DenseAllPassBranchMask>> Make(int64_t length,
                                                              ExecContext* exec_context) {
    auto branch_mask = MakeShared<DenseAllPassBranchMask>(length);
    RETURN_NOT_OK(branch_mask->Init(exec_context));
    return branch_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    DCHECK_EQ(input.length, length_);
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = nullptr;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return nullptr;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const override;

  bool empty() const override { return false; }

 protected:
  explicit DenseAllPassBranchMask(int64_t length) : length_(length) {}

 private:
  int64_t length_;
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

struct DenseBranchMask : public BranchMask {
  static Result<std::shared_ptr<DenseBranchMask>> Make(
      std::shared_ptr<SelectionVector> selection_vector, ExecContext* exec_context) {
    auto branch_mask = MakeShared<DenseBranchMask>(std::move(selection_vector));
    RETURN_NOT_OK(branch_mask->Init(exec_context));
    return branch_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(
        auto dense_input,
        TakeBySelectionVector(input, *selection_vector_->data(), exec_context));
    return ExecuteScalarExpression(expr, dense_input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return selection_vector_;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMask(
      const Datum& datum, ExecContext* exec_context) const override;

  bool empty() const override { return selection_vector_->length() == 0; }

 protected:
  explicit DenseBranchMask(std::shared_ptr<SelectionVector> selection_vector)
      : selection_vector_(std::move(selection_vector)) {}

 private:
  std::shared_ptr<SelectionVector> selection_vector_;
};

struct DenseBodyMask : public BodyMask {
  static Result<std::shared_ptr<DenseBodyMask>> Make(
      std::shared_ptr<SelectionVector> passed, std::shared_ptr<SelectionVector> failed,
      ExecContext* exec_context) {
    auto body_mask = MakeShared<DenseBodyMask>(std::move(passed), std::move(failed));
    RETURN_NOT_OK(body_mask->Init(exec_context));
    return body_mask;
  }

  Result<Datum> Apply(const Expression& expr, const ExecBatch& input,
                      ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto dense_input,
                          TakeBySelectionVector(input, *passed_->data(), exec_context));
    return ExecuteScalarExpression(expr, dense_input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector(
      ExecContext* exec_context) const override {
    return passed_;
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask(
      ExecContext* exec_context) const override {
    return DenseBranchMask::Make(failed_, exec_context);
  }

  bool empty() const override { return passed_->length() == 0; }

 protected:
  DenseBodyMask(std::shared_ptr<SelectionVector> passed,
                std::shared_ptr<SelectionVector> failed)
      : passed_(std::move(passed)), failed_(std::move(failed)) {}

 private:
  std::shared_ptr<SelectionVector> passed_;
  std::shared_ptr<SelectionVector> failed_;
};

Result<std::shared_ptr<const BodyMask>> DenseAllPassBranchMask::MakeBodyMask(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK_EQ(datum.type()->id(), Type::BOOL);

  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
  }

  auto bitmap = datum.array_as<BooleanArray>();
  DCHECK_EQ(bitmap->length(), length_);
  Int32Builder passed_builder(exec_context->memory_pool());
  Int32Builder failed_builder(exec_context->memory_pool());
  RETURN_NOT_OK(passed_builder.Reserve(bitmap->length()));
  RETURN_NOT_OK(failed_builder.Reserve(bitmap->length()));
  ArraySpan span(*bitmap->data());
  int32_t i = 0;
  RETURN_NOT_OK(VisitArraySpanInline<BooleanType>(
      span,
      [&](bool mask) -> Status {
        if (mask) {
          RETURN_NOT_OK(passed_builder.Append(i));
        } else {
          RETURN_NOT_OK(failed_builder.Append(i));
        }
        ++i;
        return Status::OK();
      },
      [&]() {
        ++i;
        return Status::OK();
      }));
  ARROW_ASSIGN_OR_RAISE(auto passed, passed_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto failed, failed_builder.Finish());
  auto passed_sv = std::make_shared<SelectionVector>(passed->data());
  auto failed_sv = std::make_shared<SelectionVector>(failed->data());
  return DenseBodyMask::Make(std::move(passed_sv), std::move(failed_sv), exec_context);
}

Result<std::shared_ptr<const BodyMask>> DenseBranchMask::MakeBodyMask(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK_EQ(datum.type()->id(), Type::BOOL);

  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    return BodyMaskFromScalar(scalar, shared_from_this(), exec_context);
  }
  auto bitmap = datum.array_as<BooleanArray>();
  DCHECK_EQ(bitmap->length(), selection_vector_->length());
  Int32Builder passed_builder(exec_context->memory_pool());
  Int32Builder failed_builder(exec_context->memory_pool());
  RETURN_NOT_OK(passed_builder.Reserve(bitmap->length()));
  RETURN_NOT_OK(failed_builder.Reserve(bitmap->length()));
  ArraySpan span(*bitmap->data());
  int32_t i = 0;
  RETURN_NOT_OK(VisitArraySpanInline<BooleanType>(
      span,
      [&](bool mask) -> Status {
        if (mask) {
          RETURN_NOT_OK(passed_builder.Append(selection_vector_->indices()[i]));
        } else {
          RETURN_NOT_OK(failed_builder.Append(selection_vector_->indices()[i]));
        }
        ++i;
        return Status::OK();
      },
      [&]() {
        ++i;
        return Status::OK();
      }));
  ARROW_ASSIGN_OR_RAISE(auto passed, passed_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto failed, failed_builder.Finish());
  auto passed_sv = std::make_shared<SelectionVector>(passed->data());
  auto failed_sv = std::make_shared<SelectionVector>(failed->data());
  return DenseBodyMask::Make(std::move(passed_sv), std::move(failed_sv), exec_context);
}

struct Branch {
  Expression cond;
  Expression body;
};

template <typename Impl>
struct ConditionalExecutor {
  explicit ConditionalExecutor(std::vector<Branch> branches,
                               std::shared_ptr<DataType> result_type)
      : branches(std::move(branches)), result_type_(std::move(result_type)) {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(ConditionalExecutor);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ConditionalExecutor);

  Result<Datum> Execute(const ExecBatch& input, ExecContext* exec_context) const {
    BranchResults results;
    results.Reserve(branches.size());
    ARROW_ASSIGN_OR_RAISE(
        auto branch_mask,
        static_cast<const Impl*>(this)->InitBranchMask(input, exec_context));
    for (const auto& branch : branches) {
      if (branch_mask->empty()) {
        break;
      }
      ARROW_ASSIGN_OR_RAISE(auto cond,
                            branch_mask->Apply(branch.cond, input, exec_context));
      ARROW_ASSIGN_OR_RAISE(auto body_mask,
                            branch_mask->MakeBodyMask(cond, exec_context));
      if (body_mask->empty()) {
        ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask(exec_context));
        continue;
      }
      ARROW_ASSIGN_OR_RAISE(auto value,
                            body_mask->Apply(branch.body, input, exec_context));
      DCHECK(value.type()->Equals(result_type_));
      ARROW_ASSIGN_OR_RAISE(auto selection_vector,
                            body_mask->GetSelectionVector(exec_context));
      results.Emplace(std::move(value), std::move(selection_vector));
      ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask(exec_context));
    }
    return static_cast<const Impl*>(this)->MultiplexBranchResults(input, results,
                                                                  exec_context);
  }

 protected:
  struct BranchResults {
    void Reserve(int64_t size) {
      values_.reserve(size);
      selection_vectors_.reserve(size);
    }

    void Emplace(Datum value, std::shared_ptr<SelectionVector> selection_vector) {
      values_.emplace_back(std::move(value));
      selection_vectors_.emplace_back(std::move(selection_vector));
    }

    bool empty() const { return values_.empty(); }

    size_t size() const { return values_.size(); }

    const std::vector<Datum>& values() const { return values_; }

    const std::vector<std::shared_ptr<SelectionVector>>& selection_vectors() const {
      return selection_vectors_;
    }

   private:
    std::vector<Datum> values_;
    std::vector<std::shared_ptr<SelectionVector>> selection_vectors_;
  };

 protected:
  std::vector<Branch> branches;
  std::shared_ptr<DataType> result_type_;
};

struct SparseConditionalExecutor : public ConditionalExecutor<SparseConditionalExecutor> {
  using ConditionalExecutor<SparseConditionalExecutor>::ConditionalExecutor;

  Result<std::shared_ptr<const BranchMask>> InitBranchMask(
      const ExecBatch& input, ExecContext* exec_context) const {
    if (input.selection_vector) {
      return SparseSelectionVectorBranchMask::Make(input.selection_vector, input.length,
                                                   exec_context);
    }
    return SparseAllPassBranchMask::Make(input.length, exec_context);
  }

  Result<Datum> MultiplexBranchResults(const ExecBatch& input,
                                       const BranchResults& results,
                                       ExecContext* exec_context) const {
    if (results.empty()) {
      return MakeArrayOfNull(result_type_, input.length, exec_context->memory_pool());
    }

    if (results.size() == 1) {
      return results.values()[0];
    }

    std::vector<Datum> choose_args;
    choose_args.reserve(results.size() + 1);
    ARROW_ASSIGN_OR_RAISE(auto indices, ComputeChooseIndices(results.selection_vectors(),
                                                             input.length, exec_context));
    choose_args.emplace_back(std::move(indices));
    choose_args.insert(choose_args.end(), results.values().begin(),
                       results.values().end());
    return CallFunction("choose", choose_args, exec_context);
  }

 private:
  Result<Datum> ComputeChooseIndices(
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

  Result<std::shared_ptr<const BranchMask>> InitBranchMask(
      const ExecBatch& input, ExecContext* exec_context) const {
    if (input.selection_vector) {
      return DenseBranchMask::Make(input.selection_vector, exec_context);
    }
    return DenseAllPassBranchMask::Make(input.length, exec_context);
  }

  Result<Datum> MultiplexBranchResults(const ExecBatch& input,
                                       const BranchResults& results,
                                       ExecContext* exec_context) const {
    if (results.empty()) {
      return MakeArrayOfNull(result_type_, input.length, exec_context->memory_pool());
    }

    if (results.size() == 1) {
      if (!results.selection_vectors()[0] ||
          results.selection_vectors()[0]->length() == input.length) {
        return results.values()[0];
      }
      return Scatter({results.values()[0]},
                     {MakeArray(results.selection_vectors()[0]->data())},
                     ScatterOptions{/*max_index=*/input.length - 1});
    }

    ARROW_ASSIGN_OR_RAISE(
        auto values,
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
        Scatter(values, indices, ScatterOptions{/*max_index=*/input.length - 1}));
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
          chunk_func(results.values()[i], results.selection_vectors()[i], chunks));
    }
    return std::make_shared<ChunkedArray>(std::move(chunks));
  }
};

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
  DCHECK_EQ(arguments.size(), 3);
  const auto& cond_expr = arguments[0];
  DCHECK_EQ(cond_expr.type()->id(), Type::BOOL);
  const auto& if_true_expr = arguments[1];
  const auto& if_false_expr = arguments[2];
  DCHECK_EQ(if_true_expr.type()->id(), if_false_expr.type()->id());
  auto result_type = if_true_expr.type()->GetSharedPtr();

  std::vector<Branch> branches = {
      {cond_expr, if_true_expr},
      {literal(true), if_false_expr},
  };
  if (IsSelectionVectorAwarePathAvailable({cond_expr, if_true_expr, if_false_expr}, input,
                                          exec_context)) {
    return SparseConditionalExecutor(std::move(branches), std::move(result_type))
        .Execute(input, exec_context);
  } else {
    return DenseConditionalExecutor(std::move(branches), std::move(result_type))
        .Execute(input, exec_context);
  }
}

// std::shared_ptr<ChunkedArray> ChunkedArrayFromDatums(const std::vector<Datum>& datums)
// {
//   std::vector<std::shared_ptr<Array>> chunks;
//   for (const auto& datum : datums) {
//     DCHECK(datum.is_arraylike());
//     if (datum.is_array() && datum.length() > 0) {
//       chunks.push_back(datum.make_array());
//     } else {
//       DCHECK(datum.is_chunked_array());
//       for (const auto& chunk : datum.chunked_array()->chunks()) {
//         if (chunk->length() > 0) {
//           chunks.push_back(chunk);
//         }
//       }
//     }
//   }
//   return std::make_shared<ChunkedArray>(std::move(chunks));
// }

// Result<Datum> IfElseSpecialForm::Execute(const std::vector<Expression>& arguments,
//                                          const ExecBatch& input,
//                                          ExecContext* exec_context) const {
//   DCHECK_EQ(input.selection_vector, nullptr);
//   DCHECK_EQ(arguments.size(), 3);

//   const auto& cond_expr = arguments[0];
//   DCHECK_EQ(cond_expr.type()->id(), Type::BOOL);
//   const auto& if_true_expr = arguments[1];
//   const auto& if_false_expr = arguments[2];
//   DCHECK_EQ(if_true_expr.type()->id(), if_false_expr.type()->id());

//   ARROW_ASSIGN_OR_RAISE(auto cond,
//                         ExecuteScalarExpression(cond_expr, input, exec_context));
//   DCHECK_EQ(cond.type()->id(), Type::BOOL);

//   if (cond.is_scalar()) {
//     auto cond_scalar = cond.scalar_as<BooleanScalar>();
//     if (!cond_scalar.is_valid) {
//       // Always eagerly return minimal "shape" whenever possible because special form
//       has
//       // all the power to shortcut the computation. This means we don't really respect
//       the
//       // shape of the input (scalar/array/chunked array) as what normal expression
//       // evaluation does.
//       return MakeNullScalar(if_true_expr.type()->GetSharedPtr());
//     } else if (cond_scalar.value) {
//       return ExecuteScalarExpression(if_true_expr, input, exec_context);
//     } else {
//       return ExecuteScalarExpression(if_false_expr, input, exec_context);
//     }
//   }

//   Datum if_true = MakeNullScalar(if_true_expr.type()->GetSharedPtr());
//   Datum if_false = MakeNullScalar(if_false_expr.type()->GetSharedPtr());

//   if (IsSelectionVectorAwarePathAvailable({if_true_expr, if_false_expr}, input,
//                                           exec_context)) {
//     DCHECK(cond.is_array());
//     auto boolean_cond = cond.array_as<BooleanArray>();
//     if (boolean_cond->null_count() == boolean_cond->length()) {
//       return MakeArrayOfNull(if_true_expr.type()->GetSharedPtr(), input.length,
//                              exec_context->memory_pool());
//     }

//     if (boolean_cond->true_count() == boolean_cond->length()) {
//       return ExecuteScalarExpression(if_true_expr, input, exec_context);
//     }

//     if (boolean_cond->false_count() == boolean_cond->length()) {
//       return ExecuteScalarExpression(if_false_expr, input, exec_context);
//     }

//     ARROW_ASSIGN_OR_RAISE(auto sel_true, SelectionVector::FromMask(
//                                              *boolean_cond,
//                                              exec_context->memory_pool()));
//     if (sel_true->length() > 0) {
//       ExecBatch input_true = input;
//       input_true.selection_vector = sel_true;
//       ARROW_ASSIGN_OR_RAISE(
//           if_true, ExecuteScalarExpression(if_true_expr, input_true, exec_context));
//     }

//     ARROW_ASSIGN_OR_RAISE(auto cond_inverted,
//                           CallFunction("invert", {cond}, exec_context));
//     DCHECK(cond_inverted.is_array());
//     auto boolean_cond_inverted = cond_inverted.array_as<BooleanArray>();
//     ARROW_ASSIGN_OR_RAISE(
//         auto sel_false,
//         SelectionVector::FromMask(*boolean_cond_inverted,
//         exec_context->memory_pool()));
//     if (sel_false->length() > 0) {
//       ExecBatch input_false = input;
//       input_false.selection_vector = sel_false;
//       ARROW_ASSIGN_OR_RAISE(
//           if_false, ExecuteScalarExpression(if_false_expr, input_false, exec_context));
//     }

//     return CallFunction("if_else", {cond, if_true, if_false}, exec_context);
//   }

//   ARROW_ASSIGN_OR_RAISE(auto sel_true,
//                         CallFunction("indices_nonzero", {cond}, exec_context));
//   ARROW_ASSIGN_OR_RAISE(auto cond_inverted, CallFunction("invert", {cond},
//   exec_context)); ARROW_ASSIGN_OR_RAISE(auto sel_false,
//                         CallFunction("indices_nonzero", {cond_inverted},
//                         exec_context));

//   if (sel_true.length() == 0 && sel_false.length() == 0) {
//     return MakeArrayOfNull(if_true_expr.type()->GetSharedPtr(), input.length,
//                            exec_context->memory_pool());
//   }

//   if (sel_true.length() == input.length) {
//     return ExecuteScalarExpression(if_true_expr, input, exec_context);
//   }

//   if (sel_false.length() == input.length) {
//     return ExecuteScalarExpression(if_false_expr, input, exec_context);
//   }

//   if (sel_true.length() > 0) {
//     ARROW_ASSIGN_OR_RAISE(auto input_true,
//                           TakeBySelectionVector(input, sel_true, exec_context));
//     ARROW_ASSIGN_OR_RAISE(
//         if_true, ExecuteScalarExpression(if_true_expr, input_true, exec_context));
//   }

//   if (sel_false.length() > 0) {
//     ARROW_ASSIGN_OR_RAISE(auto input_false,
//                           TakeBySelectionVector(input, sel_false, exec_context));
//     ARROW_ASSIGN_OR_RAISE(
//         if_false, ExecuteScalarExpression(if_false_expr, input_false, exec_context));
//   }

//   auto if_true_false = ChunkedArrayFromDatums({if_true, if_false});
//   auto sel = ChunkedArrayFromDatums({sel_true, sel_false});
//   ARROW_ASSIGN_OR_RAISE(
//       auto result_datum,
//       Scatter(if_true_false, sel, ScatterOptions{/*max_index=*/input.length - 1}));
//   DCHECK(result_datum.is_arraylike());
//   if (result_datum.is_chunked_array() &&
//       result_datum.chunked_array()->num_chunks() == 1) {
//     return result_datum.chunked_array()->chunk(0);
//   } else {
//     return result_datum;
//   }
// }

}  // namespace arrow::compute
