//
// Created by LYL232 on 2021/2/28.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RECEIVETENSOROP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RECEIVETENSOROP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class ReceiveTensorOp : public tensorflow::AsyncOpKernel {
public:
    explicit ReceiveTensorOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int sender_;
};


}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RECEIVETENSOROP_H
