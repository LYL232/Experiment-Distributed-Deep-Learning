//
// Created by LYL232 on 2021/2/6.
//
/**
 * 此文件定义通信的抽象接口
 */

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATION_H

#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

class Communication {
};

/**
 * 全规约接口
 */
class AllreduceCommunication : virtual public Communication {
public:

    enum Operation : uchar {
        OP_ALLREDUCE_SUM = 0,
    };

    /**
     * 全规约通信
     * @param sendBuffer 传输的数据缓冲
     * @param recvBuffer 接受的数据缓冲
     * @param elements 传输的元素个数
     * @param dtype tensorflow DataType
     * @param op 进行的规约运算
     * @return
     */
    virtual StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            Operation op) const = 0;
};

}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATION_H
