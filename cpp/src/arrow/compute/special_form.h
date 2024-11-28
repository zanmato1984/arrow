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

#include "arrow/compute/expression.h"

namespace arrow::compute {

class ARROW_EXPORT SpecialFormExec {
 public:
  virtual ~SpecialFormExec() = default;

  virtual Result<TypeHolder> Bind(const std::vector<Expression>& arguments,
                                  ExecContext* exec_context) = 0;

  virtual Result<Datum> Execute(const ExecBatch& input,
                                ExecContext* exec_context) const = 0;
};

class ARROW_EXPORT SpecialForm {
 public:
  explicit SpecialForm(std::string name, bool selection_vector_aware = false)
      : name(std::move(name)), selection_vector_aware(selection_vector_aware) {}

  virtual ~SpecialForm() = default;

  virtual Result<std::unique_ptr<SpecialFormExec>> DispatchExact(
      const std::vector<TypeHolder>& types, ExecContext* exec_context) const = 0;

  virtual Result<std::unique_ptr<SpecialFormExec>> DispatchBest(
      std::vector<TypeHolder>* types, ExecContext* exec_context) const = 0;

 public:
  const std::string name;
  const bool selection_vector_aware = false;
};

std::shared_ptr<SpecialForm> GetIfElseSpecialForm();

}  // namespace arrow::compute
