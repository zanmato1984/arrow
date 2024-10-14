#include "arrow/compute/expression.h"

namespace arrow::compute {

class SpecialForm {
 public:
  SpecialForm(const std::string& name) {}

 public:
  const std::string name;
};

class IfElseSpecialForm : public SpecialForm {
 public:
  IfElseSpecialForm() : SpecialForm("if_else") {}
};

}  // namespace arrow::compute
