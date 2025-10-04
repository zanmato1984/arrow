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

#include "arrow/compute/special/conditional_internal.h"

namespace arrow::compute {

namespace {

using internal::Branch;
using internal::ConditionalSpecialForm;

class IfElseSpecialForm : public ConditionalSpecialForm<IfElseSpecialForm> {
 public:
  IfElseSpecialForm() : ConditionalSpecialForm<IfElseSpecialForm>("if_else") {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(IfElseSpecialForm);
  ARROW_DEFAULT_MOVE_AND_ASSIGN(IfElseSpecialForm);

 protected:
  std::vector<Branch> GetBranches(std::vector<Expression> arguments) const {
    // The arity should be correct. This is guaranteed by the call binding.
    DCHECK_EQ(arguments.size(), 3);

    auto cond = std::move(arguments[0]);
    auto if_true = std::move(arguments[1]);
    auto if_false = std::move(arguments[2]);

    return std::vector<Branch>{{std::move(cond), std::move(if_true)},
                               {literal(true), std::move(if_false)}};
  }

  friend class ConditionalSpecialForm<IfElseSpecialForm>;
};

std::shared_ptr<SpecialForm> GetIfElseSpecialForm() {
  static auto instance = std::make_shared<IfElseSpecialForm>();
  return instance;
}

}  // namespace

Expression if_else_special(Expression cond, Expression if_true, Expression if_false) {
  Expression::Special special;
  special.special_form = GetIfElseSpecialForm();
  special.arguments = {std::move(cond), std::move(if_true), std::move(if_false)};
  return Expression(std::move(special));
}

}  // namespace arrow::compute
