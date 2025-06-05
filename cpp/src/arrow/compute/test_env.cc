#include <gtest/gtest.h>

#include "arrow/compute/initialize.h"
#include "arrow/testing/gtest_util.h"

namespace arrow::compute {

namespace {

class ComputeKernelEnvironment : public ::testing::Environment {
 public:
  // This must be done before using the compute kernels in order to
  // register them to the FunctionRegistry.
  ComputeKernelEnvironment() : ::testing::Environment() {}

  void SetUp() override { ASSERT_OK(arrow::compute::Initialize()); }
};

}  // namespace

// Initialize the compute module
::testing::Environment* compute_kernels_env =
    ::testing::AddGlobalTestEnvironment(new ComputeKernelEnvironment);

}  // namespace arrow::compute
