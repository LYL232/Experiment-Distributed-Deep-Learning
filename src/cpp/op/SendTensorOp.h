//
// Created by LYL232 on 2021/3/1.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SENDTENSOROP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SENDTENSOROP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class SendTensorOp : public tensorflow::AsyncOpKernel {
public:
    explicit SendTensorOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int receiver_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_SENDTENSOROP_H
