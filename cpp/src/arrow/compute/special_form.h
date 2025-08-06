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

class ARROW_EXPORT SpecialExecutor {
 public:
  explicit SpecialExecutor(TypeHolder out_type) : out_type_(std::move(out_type)) {}

  virtual ~SpecialExecutor() = default;

  const TypeHolder& out_type() const { return out_type_; }

  virtual Result<Datum> Execute(const ExecBatch& input,
                                ExecContext* exec_context) const = 0;

 private:
  const TypeHolder out_type_;
};

class ARROW_EXPORT SpecialForm {
 public:
  explicit SpecialForm(std::string name) : name_(std::move(name)) {}

  virtual ~SpecialForm() = default;

  const std::string& name() const { return name_; }

  virtual Result<std::unique_ptr<SpecialExecutor>> Bind(
      std::vector<Expression>& arguments, ExecContext* exec_context) = 0;

 private:
  std::string name_;
};

std::shared_ptr<SpecialForm> GetIfElseSpecialForm();

}  // namespace arrow::compute
