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

#include <gtest/gtest.h>

#include "arrow/compute/special/conditional_internal.h"
#include "arrow/compute/test_util_internal.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/checked_cast.h"

namespace arrow::compute::internal {

TEST(AllPassBranchMask, Emptiness) {
  for (auto length : {0, 42}) {
    auto branch_mask = std::make_shared<AllPassBranchMask>(length);
    EXPECT_FALSE(branch_mask->empty());
  }
}

TEST(AllPassBranchMask, ApplyCond) {
  for (auto length : {0, 42}) {
    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(auto body_mask,
                           branch_mask->ApplyCond(literal(true), ExecBatch({}, length),
                                                  default_exec_context()));
      auto casted = checked_cast<const AllPassBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(Expression expr,
                           field_ref(0).Bind(Schema({field("", boolean())})));
      ASSERT_OK_AND_ASSIGN(
          auto body_mask,
          branch_mask->ApplyCond(expr, ExecBatch({BooleanScalar(true)}, length),
                                 default_exec_context()));
      auto casted = checked_cast<const AllPassBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(auto body_mask,
                           branch_mask->ApplyCond(literal(false), ExecBatch({}, length),
                                                  default_exec_context()));
      auto casted = checked_cast<const AllFailBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(Expression expr,
                           field_ref(0).Bind(Schema({field("", boolean())})));
      ASSERT_OK_AND_ASSIGN(
          auto body_mask,
          branch_mask->ApplyCond(expr, ExecBatch({BooleanScalar(false)}, length),
                                 default_exec_context()));
      auto casted = checked_cast<const AllFailBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(
          auto body_mask,
          branch_mask->ApplyCond(literal(BooleanScalar()), ExecBatch({}, length),
                                 default_exec_context()));
      auto casted = checked_cast<const AllNullBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(Expression expr,
                           field_ref(0).Bind(Schema({field("", boolean())})));
      ASSERT_OK_AND_ASSIGN(auto body_mask, branch_mask->ApplyCond(
                                               expr, ExecBatch({BooleanScalar()}, length),
                                               default_exec_context()));
      auto casted = checked_cast<const AllNullBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }

    for (auto boolean_scalar :
         {MakeScalar(true), MakeScalar(false), MakeNullScalar(boolean())}) {
      auto branch_mask = std::make_shared<AllPassBranchMask>(length);
      ASSERT_OK_AND_ASSIGN(Expression expr,
                           field_ref(0).Bind(Schema({field("", boolean())})));
      ASSERT_OK_AND_ASSIGN(auto boolean_array,
                           gen::Constant(boolean_scalar)->Generate(length));
      ASSERT_OK_AND_ASSIGN(
          auto body_mask,
          branch_mask->ApplyCond(expr, ExecBatch({std::move(boolean_array)}, length),
                                 default_exec_context()));
      auto casted = checked_cast<const ConditionalBodyMask*>(body_mask.get());
      EXPECT_NE(casted, nullptr);
    }
  }
}

TEST(AllFailBranchMask, Emptiness) {
  auto branch_mask = std::make_shared<AllFailBranchMask>();
  EXPECT_TRUE(branch_mask->empty());
}

TEST(ConditionalBranchMask, Emptiness) {
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]")}) {
    auto branch_mask = std::make_shared<ConditionalBranchMask>(selection, /*length=*/42);
    EXPECT_EQ(branch_mask->empty(), selection->length() == 0);
  }
}

}  // namespace arrow::compute::internal
