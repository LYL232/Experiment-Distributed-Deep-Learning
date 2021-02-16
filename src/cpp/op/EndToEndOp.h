//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ENDTOENDOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ENDTOENDOP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class EndToEndOp : public tensorflow::AsyncOpKernel {
public:
    explicit EndToEndOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int sender_, receiver_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ENDTOENDOP_H
