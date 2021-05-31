//
// Created by LYL232 on 2021/5/29.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLGATHEROP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLGATHEROP_H

#include "tensorflow/core/framework/op_kernel.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class AllgatherOp : public tensorflow::AsyncOpKernel {
public:
    explicit AllgatherOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    // 为了方便传通信域对象信息, 所以在op的参数里定义communicator为整数, 其即是一个Communicator对象的指针
    Communicator::ID communicatorId_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_ALLGATHEROP_H
