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

/// The concept of "special form" is borrowed from Lisp
/// (https://courses.cs.northwestern.edu/325/readings/special-forms.html). A special form
/// is used to implement evaluation strategies
/// (https://en.wikipedia.org/wiki/Evaluation_strategy) other than the default
/// call-by-value strategy used by Arrow expression evaluation. Velox also uses this term.
///
/// In a call-by-value strategy, all arguments are evaluated before the function
/// invocation. For example, consider a regular function call
///   if_else(a, b, c)
/// Under call-by-value semantics, the expressions `a`, `b`, and `c` are all evaluated
/// before calling `if_else`. This can lead to unintuitive behavior when subexpressions
/// have observable side effects. For instance,
///   if_else(not_equal(a, 0), divide(b, a), 0)
/// should never produce a divide-by-zero error in most programming languages. However,
/// under call-by-value semantics, `divide(b, a)` is evaluated regardless of the
/// condition, so a divide-by-zero error can still occur. To address this, a special form
/// for `if_else` would be needed, namely `if_else_special`, that follows a
/// call-by-name-like evaluation strategy, where, for each row in a batch, only one of the
/// branches is evaluated based on the corresponding value of condition.
///
/// Each API in this file is intended to refer to a concrete special form. In addition to
/// the aforementioned `if_else_special`, the design anticipates variants of conditional
/// constructs such as `case_when_special` and `coalesce_special`, as well as boolean
/// operators with short-circuit semantics, such as `and_special` and `or_special`, some
/// of which may not be implemented yet.

/// TODO: Doc.
ARROW_EXPORT
Expression if_else_special(Expression cond, Expression if_true, Expression if_false);

}  // namespace arrow::compute