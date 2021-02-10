//
// Created by LYL232 on 2021/2/6.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLREDUCEOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLREDUCEOP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {
class AllreduceOp : public tensorflow::AsyncOpKernel {
public:
    explicit AllreduceOp(tensorflow::OpKernelConstruction *context) :
            AsyncOpKernel(context) {}

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;
};
}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLREDUCEOP_H
