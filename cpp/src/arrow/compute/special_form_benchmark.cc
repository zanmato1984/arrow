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

#include "benchmark/benchmark.h"

#include "arrow/compute/exec.h"
#include "arrow/compute/expression.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/random.h"

namespace arrow {

namespace compute {

namespace {

Expression if_else_regular(Expression cond, Expression if_true, Expression if_false) {
  return call("if_else", {std::move(cond), std::move(if_true), std::move(if_false)});
}

}  // namespace

template <typename... Args>
static void BM_IfElseDecent(
    benchmark::State& state,
    std::function<Expression(Expression, Expression, Expression)> if_else_func,
    Args&&...) {
  auto schema =
      arrow::schema({field("a", boolean()), field("b", int32()), field("c", int32())});
  random::RandomArrayGenerator rag(42);
  int64_t num_rows = 65536;
  auto cond = rag.Boolean(num_rows, 0.5, 0.0);
  auto if_true = rag.Int32(num_rows, 0, 42);
  auto if_false = rag.Int32(num_rows, 0, 42);
  auto if_else = if_else_func(field_ref("a"), field_ref("b"), field_ref("c"));
  auto bound = if_else.Bind(*schema);
  ExecBatch batch{
      std::vector<Datum>{std::move(cond), std::move(if_true), std::move(if_false)},
      num_rows};
  for (auto _ : state) {
    std::ignore = ExecuteScalarExpression(if_else, batch);
  }
  state.SetItemsProcessed(num_rows * state.iterations());
}

BENCHMARK_CAPTURE(BM_IfElseDecent, "if_else_regular", {if_else_regular});
BENCHMARK_CAPTURE(BM_IfElseDecent, "if_else_special", {if_else_special});

}  // namespace compute

}  // namespace arrow
