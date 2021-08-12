//
// Created by LYL232 on 2021/8/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ASYNC_OP_KERNEL_WITH_KEY_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ASYNC_OP_KERNEL_WITH_KEY_H

#include "tensorflow/core/framework/op_kernel.h"

namespace lyl232 { namespace experiment { namespace ddl {

class AsyncOpKernelWithKey : public tensorflow::AsyncOpKernel {
public:
    typedef std::function<void()> DoneCallback;

    AsyncOpKernelWithKey(tensorflow::OpKernelConstruction *context);

    const std::string key() const;

protected:
    std::string key_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ASYNC_OP_KERNEL_WITH_KEY_H
