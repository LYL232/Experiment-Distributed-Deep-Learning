//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H


#include "communicate/tensor/collective/controller/rtc/RingTokenCommunicateHandler.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

/**
 * 由于设计时一个RingTokenCommunicateHandler只能处理一个通信域, 所以为了能够处理所有通信域的请求,
 * 实现此类, 主要思路是用一个map记录每个通信域和其对应的Handler, 遇到新的就新建一个Handler, 否则返回已经存在的Handler,
 * 或许可以设计一个通用的
 */
class RingTokenCommunicateController : public TensorsCollectiveCommunicateController {
public:
    RingTokenCommunicateController();

    RingTokenCommunicateController(const RingTokenCommunicateController &) = delete;

    RingTokenCommunicateController(RingTokenCommunicateController &&) = delete;

    StatusCode handleRequest(const std::shared_ptr<TensorCollectiveCommunicateRequest> &request) override;

    StatusCode allreduce(const Requests &requests) override;

    StatusCode allgather(const Requests &requests) override;

    StatusCode broadcast(const Requests &requests) override;

    ~RingTokenCommunicateController() override;

protected:

    /**
     * 由子类实现的创建一个新的Handler方法
     * @return
     */
    virtual std::shared_ptr<RingTokenCommunicateHandler> newHandler(
            const std::shared_ptr<Communicator> &communicator);

private:
    RingTokenCommunicateHandler &getHandler(const std::shared_ptr<Communicator> &communicator);

    // 用来记录每个通信域的控制器, 懒加载, 按需分配
    std::map<Communicator::ID, std::shared_ptr<RingTokenCommunicateHandler>> handlerMap_;
    // 用来记录每个handler进入的顺序，因为handler不按创建的顺序释放会造成死锁
    std::map<size_t, std::shared_ptr<RingTokenCommunicateHandler>> orderedByEnteringHandlerMap_;
    // 读写锁
    pthread_rwlock_t rwlock_;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H
