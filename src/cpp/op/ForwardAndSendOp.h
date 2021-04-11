//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H

#include <string>
#include <mutex>
#include "tensorflow/core/framework/op_kernel.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class ForwardAndSendOp : public tensorflow::AsyncOpKernel {
public:
    explicit ForwardAndSendOp(tensorflow::OpKernelConstruction *context);

    void ComputeAsync(tensorflow::OpKernelContext *context, DoneCallback done) override;

private:
    int receiver_;
    std::string msg_;
    // 为了方便传通信域对象信息, 所以在op的参数里定义communicator为整数, 其即是一个Communicator对象的指针
    Communicator::ID communicatoId_;

    // 因为发送梯度分为发送信息和发送张量两步, 这两步不能打断, 否则会出问题, 所以用一个锁进行同步
    static std::mutex sendingMutex_;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_FORWARDANDSENDOP_H
