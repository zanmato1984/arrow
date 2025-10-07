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

namespace arrow::compute::internal {

TEST(AllPassBranchMask, Emptiness) {
  for (auto length : {0, 42}) {
    auto all_pass = std::make_shared<AllPassBranchMask>(length);
    EXPECT_FALSE(all_pass->empty());
  }
}

TEST(AllFailBranchMask, Emptiness) {
  auto all_fail = std::make_shared<AllFailBranchMask>();
  EXPECT_TRUE(all_fail->empty());
}

TEST(ConditionalBranchMask, Emptiness) {
  for (const auto& selection :
       {SelectionVectorFromJSON("[]"), SelectionVectorFromJSON("[0]"),
        SelectionVectorFromJSON("[0, 41]")}) {
    auto conditional = std::make_shared<ConditionalBranchMask>(selection, /*length=*/42);
    EXPECT_EQ(conditional->empty(), selection->length() == 0);
  }
}

}  // namespace arrow::compute::internal
