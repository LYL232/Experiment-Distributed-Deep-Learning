//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H

#include <string>
#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class ForwardAndSendOp : public tensorflow::AsyncOpKernel {
public:
    explicit ForwardAndSendOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int receiver_;
    std::string msg_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H
