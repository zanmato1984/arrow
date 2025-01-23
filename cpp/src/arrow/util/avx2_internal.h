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

#include "arrow/util/simd.h"

#if !defined(ARROW_HAVE_AVX2) && !defined(ARROW_HAVE_AVX512) && \
    !defined(ARROW_HAVE_RUNTIME_AVX2) && !defined(ARROW_HAVE_RUNTIME_AVX512)
static_assert(false, "This file should only be included when AVX2 or AVX512 is enabled");
#endif

namespace arrow::util {

inline std::pair<__m256i, __m256i> ConvertInt32ToInt64(__m256i v) {
  return {_mm256_cvtepi32_epi64(_mm256_castsi256_si128(v)),
          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1))};
}

inline __m256i MulAddInt64(__m256 v, __m256 factor, __m256 addend) {
  return _mm256_add_epi64(addend, _mm256_mul_epi32(v, factor));
}

inline __m256i MulAddInt64(__m256 v, int64_t factor, int64_t addend) {
  return MulAddInt64(v, _mm256_set1_epi64x(factor), _mm256_set1_epi64x(addend));
}

}  // namespace arrow::util
