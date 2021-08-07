//
// Created by LYL232 on 2021/6/19.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DOBUTPASSBYOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DOBUTPASSBYOP_H

#include "tensorflow/core/framework/op_kernel.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class PassWithComputed: public tensorflow::OpKernel {
public:
    explicit PassWithComputed(tensorflow::OpKernelConstruction *context);

    void Compute(tensorflow::OpKernelContext *context) override;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_DOBUTPASSBYOP_H
