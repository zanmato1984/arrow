
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

#include <memory>
#include <optional>

#include "arrow/compute/expression_internal.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::compute::internal {

const std::shared_ptr<Schema> kBoringSchema = schema({
    field("bool", boolean()),
    field("i8", int8()),
    field("i32", int32()),
    field("i32_req", int32(), /*nullable=*/false),
    field("u32", uint32()),
    field("i64", int64()),
    field("f32", float32()),
    field("f32_req", float32(), /*nullable=*/false),
    field("f64", float64()),
    field("date64", date64()),
    field("str", utf8()),
    field("dict_str", dictionary(int32(), utf8())),
    field("dict_i32", dictionary(int32(), int32())),
    field("ts_ns", timestamp(TimeUnit::NANO)),
    field("ts_s", timestamp(TimeUnit::SECOND)),
    field("binary", binary()),
    field("ts_s_utc", timestamp(TimeUnit::SECOND, "UTC")),
});

inline Expression cast(Expression argument, std::shared_ptr<DataType> to_type) {
  return call("cast", {std::move(argument)},
              compute::CastOptions::Safe(std::move(to_type)));
}

inline Expression true_unless_null(Expression argument) {
  return call("true_unless_null", {std::move(argument)});
}

inline Expression add(Expression l, Expression r) {
  return call("add", {std::move(l), std::move(r)});
}

inline Expression sub(Expression l, Expression r) {
  return call("subtract", {std::move(l), std::move(r)});
}

inline std::string make_range_json(int start, int end) {
  std::string result = "[";
  for (int i = start; i <= end; ++i) {
    if (i > start) result += ",";
    result += std::to_string(i);
  }
  result += "]";
  return result;
}

const auto no_change = std::nullopt;

inline void ExpectBindsTo(Expression expr, std::optional<Expression> expected,
                          Expression* bound_out = nullptr,
                          const Schema& schema = *kBoringSchema) {
  if (!expected) {
    expected = expr;
  }

  ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(schema));
  EXPECT_TRUE(bound.IsBound());

  ASSERT_OK_AND_ASSIGN(expected, expected->Bind(schema));
  EXPECT_EQ(bound, *expected) << " unbound: " << expr.ToString();

  if (bound_out) {
    *bound_out = bound;
  }
}

}  // namespace arrow::compute::internal
