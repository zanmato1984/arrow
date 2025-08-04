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
#include "arrow/chunk_resolver.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/expression_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/util/logging_internal.h"

namespace arrow::compute {

namespace {

struct BodyMask;

struct BranchMask : public std::enable_shared_from_this<BranchMask> {
  virtual ~BranchMask() = default;

  Result<std::shared_ptr<const BodyMask>> ApplyCond(const Expression& expr,
                                                    const ExecBatch& input,
                                                    ExecContext* exec_context) const {
    ARROW_ASSIGN_OR_RAISE(auto datum, DoApplyCond(expr, input, exec_context));
    return MakeBodyMaskFromDatum(datum, exec_context);
  }

  virtual bool empty() const = 0;

 protected:
  virtual Result<Datum> DoApplyCond(const Expression& expr, const ExecBatch& input,
                                    ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const = 0;

 private:
  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromDatum(
      const Datum& datum, ExecContext* exec_context) const;

  friend struct DelegateBodyMask;
};

struct BodyMask : public std::enable_shared_from_this<BodyMask> {
  virtual ~BodyMask() = default;

  virtual bool empty() const = 0;

  virtual Result<Datum> ApplyCond(const Expression& expr, const ExecBatch& input,
                                  ExecContext* exec_context) const = 0;

  virtual Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const = 0;

  virtual Result<std::shared_ptr<const BranchMask>> NextBranchMask() const = 0;
};

struct AllPassBranchMask : public BranchMask {
  explicit AllPassBranchMask(int64_t length) : length_(length) {}

  bool empty() const override { return false; }

 protected:
  Result<Datum> DoApplyCond(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK_EQ(input.length, length_);
    auto input_with_sel_vec = input;
    input_with_sel_vec.selection_vector = nullptr;
    return ExecuteScalarExpression(expr, input_with_sel_vec, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return nullptr;
  }

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<BooleanArray>& bitmap,
      ExecContext* exec_context) const override;

  Result<std::shared_ptr<const BodyMask>> MakeBodyMaskFromBitmap(
      const std::shared_ptr<ChunkedArray>& bitmap,
      ExecContext* exec_context) const override;

 private:
  int64_t length_;
};

struct AllFailBranchMask : public BranchMask {
  AllFailBranchMask() = default;

  bool empty() const override { return true; }

 protected:
  Result<Datum> DoApplyCond(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::DoApplyCond should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBranchMask::GetSelectionVector should not be called");
  }

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

struct DelegateBodyMask : public BodyMask {
  explicit DelegateBodyMask(std::shared_ptr<const BranchMask> branch_mask)
      : branch_mask_(std::move(branch_mask)) {}

 protected:
  Result<Datum> DelegateApplyCond(const Expression& expr, const ExecBatch& input,
                                  ExecContext* exec_context) const {
    return branch_mask_->DoApplyCond(expr, input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> DelegateGetSelectionVector() const {
    return branch_mask_->GetSelectionVector();
  }

 protected:
  std::shared_ptr<const BranchMask> branch_mask_;
};

struct AllNullBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return true; }

  Result<Datum> ApplyCond(const Expression& expr, const ExecBatch& input,
                          ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::ApplyCond should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllNullBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

struct AllPassBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return false; }

  Result<Datum> ApplyCond(const Expression& expr, const ExecBatch& input,
                          ExecContext* exec_context) const override {
    return DelegateApplyCond(expr, input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return DelegateGetSelectionVector();
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return std::make_shared<AllFailBranchMask>();
  }
};

struct AllFailBodyMask : public DelegateBodyMask {
  using DelegateBodyMask::DelegateBodyMask;

  bool empty() const override { return true; }

  Result<Datum> ApplyCond(const Expression& expr, const ExecBatch& input,
                          ExecContext* exec_context) const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::ApplyCond should not be called");
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    DCHECK(false);
    return Status::Invalid("AllFailBodyMask::GetSelectionVector should not be called");
  }

  Result<std::shared_ptr<const BranchMask>> NextBranchMask() const override {
    return branch_mask_;
  }
};

Result<std::shared_ptr<const BodyMask>> BranchMask::MakeBodyMaskFromDatum(
    const Datum& datum, ExecContext* exec_context) const {
  DCHECK(datum.type()->id() == Type::BOOL);
  if (datum.is_scalar()) {
    auto scalar = datum.scalar_as<BooleanScalar>();
    if (!scalar.is_valid) {
      return std::make_shared<AllNullBodyMask>(shared_from_this());
    } else if (scalar.value) {
      return std::make_shared<AllPassBodyMask>(shared_from_this());
    } else {
      return std::make_shared<AllFailBodyMask>(shared_from_this());
    }
  }
  if (datum.is_array()) {
    return MakeBodyMaskFromBitmap(datum.array_as<BooleanArray>(), exec_context);
  }
  DCHECK(datum.is_chunked_array());
  return MakeBodyMaskFromBitmap(datum.chunked_array(), exec_context);
}

struct ConditionalBranchMask : public BranchMask {
  ConditionalBranchMask(std::shared_ptr<SelectionVector> selection_vector, int64_t length)
      : selection_vector_(std::move(selection_vector)), length_(length) {}

  bool empty() const override { return selection_vector_->length() == 0; }

 protected:
  Result<Datum> DoApplyCond(const Expression& expr, const ExecBatch& input,
                            ExecContext* exec_context) const override {
    auto sparse_input = input;
    sparse_input.selection_vector = selection_vector_;
    return ExecuteScalarExpression(expr, sparse_input, exec_context);
  }

  Result<std::shared_ptr<SelectionVector>> GetSelectionVector() const override {
    return selection_vector_;
  }

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

struct ConditionalBodyMask : public BodyMask {
  ConditionalBodyMask(std::shared_ptr<SelectionVector> body,
                      std::shared_ptr<SelectionVector> rest, int64_t length)
      : body_(std::move(body)), rest_(std::move(rest)), length_(length) {}

  bool empty() const override { return body_->length() == 0; }

  Result<Datum> ApplyCond(const Expression& expr, const ExecBatch& input,
                          ExecContext* exec_context) const override {
    auto sparse_input = input;
    sparse_input.selection_vector = body_;
    return ExecuteScalarExpression(expr, sparse_input, exec_context);
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

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromBitmap(
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

Result<std::shared_ptr<const BodyMask>> AllPassBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(rest_builder.Reserve(length_));

  int32_t i = 0;
  for (const auto& chunk : bitmap->chunks()) {
    DCHECK_EQ(chunk->type()->id(), Type::BOOL);
    ArraySpan span(*chunk->data());
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
  }

  ARROW_ASSIGN_OR_RAISE(auto body_arr, body_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto rest_arr, rest_builder.Finish());
  auto body = std::make_shared<SelectionVector>(body_arr->data());
  auto rest = std::make_shared<SelectionVector>(rest_arr->data());
  return std::make_shared<ConditionalBodyMask>(std::move(body), std::move(rest), length_);
}

Result<std::shared_ptr<const BodyMask>> ConditionalBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<BooleanArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(rest_builder.Reserve(length_));

  for (int64_t i = 0; i < selection_vector_->length(); ++i) {
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

Result<std::shared_ptr<const BodyMask>> ConditionalBranchMask::MakeBodyMaskFromBitmap(
    const std::shared_ptr<ChunkedArray>& bitmap, ExecContext* exec_context) const {
  DCHECK_EQ(bitmap->length(), length_);

  std::vector<const BooleanArray*> boolean_arrays(bitmap->num_chunks());
  std::transform(bitmap->chunks().begin(), bitmap->chunks().end(), boolean_arrays.begin(),
                 [](const auto& chunk) {
                   DCHECK_EQ(chunk->type()->id(), Type::BOOL);
                   return checked_cast<const BooleanArray*>(chunk.get());
                 });

  Int32Builder body_builder(exec_context->memory_pool());
  Int32Builder rest_builder(exec_context->memory_pool());
  RETURN_NOT_OK(body_builder.Reserve(length_));
  RETURN_NOT_OK(rest_builder.Reserve(length_));

  ChunkResolver resolver(bitmap->chunks());
  ChunkLocation location;
  for (int64_t i = 0; i < selection_vector_->length(); ++i) {
    auto index = selection_vector_->indices()[i];
    location = resolver.ResolveWithHint(index, location);
    if (boolean_arrays[location.chunk_index]->IsValid(location.index_in_chunk)) {
      if (boolean_arrays[location.chunk_index]->Value(location.index_in_chunk)) {
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

struct Branch {
  Expression cond;
  Expression body;
};

struct ConditionalExecutor {
  ConditionalExecutor(const std::vector<Branch>& branches, const TypeHolder& result_type)
      : branches(branches), result_type(result_type) {}

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
                            ApplyBody(body_mask, branch.body, input, exec_context));
      DCHECK(body_result.type()->Equals(*result_type));
      ARROW_ASSIGN_OR_RAISE(auto selection_vector, body_mask->GetSelectionVector());
      results.Emplace(std::move(body_result), std::move(selection_vector));
      ARROW_ASSIGN_OR_RAISE(branch_mask, body_mask->NextBranchMask());
    }
    return MultiplexResults(input, results, exec_context);
  }

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
      return std::make_shared<ConditionalBranchMask>(input.selection_vector,
                                                     input.length);
    }
    return std::make_shared<AllPassBranchMask>(input.length);
  }

  Result<std::shared_ptr<const BodyMask>> ApplyCond(
      const std::shared_ptr<const BranchMask>& branch_mask, const Expression& cond,
      const ExecBatch& input, ExecContext* exec_context) const {
    return branch_mask->ApplyCond(cond, input, exec_context);
  }

  Result<Datum> ApplyBody(const std::shared_ptr<const BodyMask>& body_mask,
                          const Expression& body, const ExecBatch& input,
                          ExecContext* exec_context) const {
    return body_mask->ApplyCond(body, input, exec_context);
  }

  Result<Datum> MultiplexResults(const ExecBatch& input, const BranchResults& results,
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
      for (int64_t i = 0; i < selection_vectors[index]->length(); ++i) {
        const int32_t row_id = row_ids[i];
        DCHECK_EQ(bit_util::GetBit(validity_data, row_id), false);
        bit_util::SetBitTo(validity_data, row_id, true);
        indices_data[row_id] = index;
      }
    }

    return ArrayData::Make(int32(), length,
                           {std::move(validity_buf), std::move(indices_buf)});
  }

 protected:
  const std::vector<Branch>& branches;
  const TypeHolder& result_type;
};

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

    branches[0] = {cond, if_true};
    branches[1] = {literal(true), if_false};

    return type;
  }

  Result<Datum> Execute(const ExecBatch& input,
                        ExecContext* exec_context) const override {
    return ConditionalExecutor(branches, type).Execute(input, exec_context);
  }

 private:
  // For Bind.
  std::shared_ptr<Function> function;
  const Kernel* kernel;

  // Post-bind, for Execute.
  TypeHolder type;
  std::vector<Branch> branches;
};

class IfElseSpecialForm : public SpecialForm {
 public:
  IfElseSpecialForm() : SpecialForm(/*name=*/"if_else") {}

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
