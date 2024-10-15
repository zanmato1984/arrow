#include "arrow/compute/expression.h"

namespace arrow::compute {

class SpecialForm {
 public:
  SpecialForm(std::string name) : name(std::move(name)) {}
  virtual ~SpecialForm() = default;

  virtual Result<TypeHolder> Resolve(std::vector<Expression>* arguments,
                                     compute::ExecContext* exec_context) const = 0;

  virtual Result<Datum> Execute(const std::vector<Expression>& arguments,
                                const ExecBatch& input,
                                compute::ExecContext* exec_context) const = 0;

 public:
  const std::string name;
};

class IfElseSpecialForm : public SpecialForm {
 public:
  IfElseSpecialForm() : SpecialForm("if_else") {}

  Result<TypeHolder> Resolve(std::vector<Expression>* arguments,
                             compute::ExecContext* exec_context) const override {
    ARROW_ASSIGN_OR_RAISE(auto function,
                          exec_context->func_registry()->GetFunction("if_else"));
    std::vector<TypeHolder> types = GetTypes(*arguments);

    // TODO: DispatchBest and implicit cast.
    ARROW_ASSIGN_OR_RAISE(auto maybe_exact_match, function->DispatchExact(types));
    compute::KernelContext kernel_context(exec_context, maybe_exact_match);
    if (maybe_exact_match->init) {
      const FunctionOptions* options = function->default_options();
      ARROW_ASSIGN_OR_RAISE(
          auto kernel_state,
          maybe_exact_match->init(&kernel_context, {maybe_exact_match, types, options}));
      kernel_context.SetState(kernel_state.get());
    }
    return maybe_exact_match->signature->out_type().Resolve(&kernel_context, types);
  }

  Result<Datum> Execute(const std::vector<Expression>& arguments, const ExecBatch& input,
                        compute::ExecContext* exec_context) const override {}
};

}  // namespace arrow::compute
