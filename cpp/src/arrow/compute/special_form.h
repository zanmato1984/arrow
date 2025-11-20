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

#include "arrow/compute/expression.h"

namespace arrow::compute {

/// @brief A bound representation of a special form that can be invoked directly on an
/// ExecBatch to produce a result Datum during expression evaluation. Conceptually, a
/// SpecialExecutor plays a role similar to a function kernel: it encapsulates the
/// concrete execution logic of a special form after binding. All child expressions,
/// as well as the special form itself, have already been bound to the input schema
/// and argument types at this point.
///
/// Unlike regular expression evaluation under the default call‑by‑value strategy [1],
/// the child expressions of a special form are not evaluated ahead of time. A
/// SpecialExecutor is therefore free to choose its own evaluation strategy, such as
/// call-by-name [2], deciding when and whether to evaluate individual child expressions
/// according to the semantics of the special form, such as conditional branching or
/// boolean short-circuiting. Implementations may leverage mechanisms such as
/// selection-vector-based masked execution to cope with the vectorized execution.
///
/// [1] https://en.wikipedia.org/wiki/Evaluation_strategy#Call_by_value
/// [2] https://en.wikipedia.org/wiki/Evaluation_strategy#Call_by_name
class ARROW_EXPORT SpecialExecutor {
 public:
  explicit SpecialExecutor(TypeHolder out_type,
                           std::shared_ptr<FunctionOptions> options = NULLPTR)
      : out_type_(std::move(out_type)), options_(std::move(options)) {}

  virtual ~SpecialExecutor() = default;

  const TypeHolder& out_type() const { return out_type_; }
  const std::shared_ptr<FunctionOptions> options() const { return options_; }

  virtual Result<Datum> Execute(const ExecBatch& input,
                                ExecContext* exec_context) const = 0;

 protected:
  const TypeHolder out_type_;
  const std::shared_ptr<FunctionOptions> options_;
};

/// @brief An unbound representation of a special form, which can be bound to produce
/// a SpecialExecutor for execution during expression evaluation.
///
/// A SpecialForm is intentionally immutable and independent of any concrete input schema,
/// argument types, options, or data. It defines how a special form should be bound to a
/// given set of bound arguments and options, to produce a concrete SpecialExecutor for
/// one particular invocation. Therefor implementations are naturally stateless and may be
/// modeled as singletons.
class ARROW_EXPORT SpecialForm {
 public:
  explicit SpecialForm(std::string name) : name_(std::move(name)) {}

  virtual ~SpecialForm() = default;

  const std::string& name() const { return name_; }

  virtual Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, std::shared_ptr<FunctionOptions> options,
      ExecContext* exec_context) const = 0;

 private:
  std::string name_;
};

}  // namespace arrow::compute
