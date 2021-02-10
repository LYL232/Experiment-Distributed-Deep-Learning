//
// Created by LYL232 on 2021/2/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H

#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

class CommunicationBackend {
    friend class Global;

    using uchar = unsigned char;
public:
    enum AllreduceOperation : uchar {
        ALLREDUCE_OP_SUM = 0,
    };

    CommunicationBackend() = default;

    CommunicationBackend(const CommunicationBackend &) = delete;

    CommunicationBackend(CommunicationBackend &&) = delete;

    int processes() const noexcept;

    int processRank() const noexcept;

    /**
     * 全规约通信
     * @param sendBuffer 传输的数据缓冲
     * @param recvBuffer 接受的数据缓冲
     * @param elements 传输的元素个数
     * @param dtype tensorflow::DataType
     * @param op 进行的规约运算
     * @return
     */
    virtual StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            AllreduceOperation op) const;


    virtual ~CommunicationBackend() {}

protected:
    // 要求子类继承实例化之后执行该方法, 相当于构造函数回调, 之所以这样实现
    // 是因为如MPI的初始化无法在构造函数中进行
    void initialize(int processes, int processRank);

    virtual void finalize();

private:
    int processes_ = -1, processRank_ = -1;
    bool initialzed_ = false;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H
