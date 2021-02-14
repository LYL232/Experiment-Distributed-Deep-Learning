//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BROADCASTOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BROADCASTOP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class BroadcastOp : public tensorflow::AsyncOpKernel {
public:
    explicit BroadcastOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int rootRank_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BROADCASTOP_H
