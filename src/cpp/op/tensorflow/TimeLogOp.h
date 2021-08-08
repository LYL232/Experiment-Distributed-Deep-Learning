//
// Created by LYL232 on 2021/8/8.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TIME_LOG_OP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TIME_LOG_OP_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class TimeLogOp : public tensorflow::OpKernel {
public:
    explicit TimeLogOp(tensorflow::OpKernelConstruction *context);

    void Compute(tensorflow::OpKernelContext *context) override;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_TIME_LOG_OP_H
