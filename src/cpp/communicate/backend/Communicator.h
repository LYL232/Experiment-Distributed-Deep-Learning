//
// Created by LYL232 on 2021/3/21.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATOR_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATOR_H

#include <memory>
#include <map>
#include "def.h"
#include "global/HeapMemoryManager.h"

namespace lyl232 { namespace experiment { namespace ddl {

/**
 * 通信域对象
 */
class Communicator {
public:
    typedef long long ID;

    enum AllreduceOperation : unsigned char {
        ALLREDUCE_OP_SUM = 0,
    };

    Communicator(int rank, int size);

    Communicator(const Communicator &other) = delete;

    Communicator(Communicator &&other) = delete;

    int rank() const noexcept;

    int size() const noexcept;

    /**
     * 全规约通信
     * @param sendBuffer 传输的数据缓冲
     * @param recvBuffer 接受的数据缓冲
     * @param elements 传输的元素个数
     * @param dtype 数据类型代码
     * @param op 进行的规约运算
     * @return StatusCode
     */
    virtual StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            AllreduceOperation op) const;

    /**
     * 全收集通信, 从每个进程接收不同的元素个数版本
     * @param sendBuffer 传输的数据缓冲
     * @param sendElements 发送的元素个数
     * @param recvBuffer 接受的数据缓冲
     * @param recvCounts 从每个进程中接收的元素个数,
     *  是一个数组, 大小为this->size(), 代表从每个进程(包括自身)接收的元素个数
     * @param displs 将从每个进程接收到的数据放在recvBuffer的偏移位置处,
     *  是一个数组, 大小为this->size(), 代表将从每个进程接收到的数据放在recvBuffer的偏移位置处
     * @param dtype 数据类型代码
     * @return StatusCode
     */
    virtual StatusCode allgather(
            void *sendBuffer, size_t sendElements,
            void *recvBuffer,
            const std::vector<size_t> &recvCounts,
            const std::vector<size_t> &displs,
            DataType dtype
    ) const;

    /**
     * 全收集通信, 从每个进程接收相同的元素个数版本
     * @param sendBuffer 传输的数据缓冲
     * @param sendElements 发送的元素个数
     * @param recvBuffer 接受的数据缓冲
     * @param recvElements 从每个进程接收的元素个数
     * @param dtype 数据类型代码
     * @return StatusCode
     */
    virtual StatusCode allgather(
            void *sendBuffer, size_t sendElements, void *recvBuffer, size_t recvElements,
            DataType dtype) const;

    /**
     * 广播通信
     * @param buffer 数据缓冲
     * @param elements 传输的元素个数
     * @param dtype 数据类型代码
     * @param rootRank 根节点
     * @return StatusCode
     */
    virtual StatusCode broadcast(
            void *buffer,
            size_t elements, DataType dtype,
            int rootRank) const;

    /**
     * 分割通信域
     * @param color 新通信域的color
     * @param key 控制进程在新通信域的rank大小
     * @return 新通信域
     */
    virtual std::shared_ptr<Communicator> split(int color, int key) const;

    /**
     * 获取通信域的唯一识别符, 主要用来比较通信域是否相同
     * @return
     */
    virtual ID id() const;

    bool operator==(const Communicator &other) const noexcept;

    virtual ~Communicator() = default;

protected:
    int rank_, size_;
    static std::shared_ptr<HeapMemoryManager> memManager_;
    static std::map<Communicator::ID, std::shared_ptr<Communicator>> &communicatorMap_() noexcept;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATOR_H
