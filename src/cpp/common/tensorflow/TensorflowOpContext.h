//
// Created by LYL232 on 2021/5/31.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_OP_CONTEXT_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_OP_CONTEXT_H

#include "common/OpContext.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TensorflowOpContext : public OpContext {
public:
    TensorflowOpContext(tensorflow::OpKernelContext &context);

    StatusCode allocateOutput(const CommonTensorShape &shape, std::shared_ptr<CommonTensor> &tensor) override;

private:
    tensorflow::OpKernelContext &context_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TENSORFLOW_OP_CONTEXT_H
