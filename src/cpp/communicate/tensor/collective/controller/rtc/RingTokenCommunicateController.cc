//
// Created by LYL232 on 2021/2/8.
//

#include "global/Global.h"
#include "communicate/tensor/collective/controller/rtc/RingTokenCommunicateController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {


RingTokenCommunicateController::RingTokenCommunicateController() :
        handlerMap_(),
        orderedByEnteringHandlerMap_(),
        rwlock_(PTHREAD_RWLOCK_INITIALIZER) {}

StatusCode RingTokenCommunicateController::handleRequest(
        const std::shared_ptr<TensorCollectiveCommunicateRequest> &request) {
    return getHandler(request->communicator()).handleRequest(request);
}

StatusCode RingTokenCommunicateController::allreduce(const Requests &requests) {
    assert(!requests.empty());
    const auto &request = requests[0];
    return getHandler(request->communicator()).allreduce(requests);
}

StatusCode RingTokenCommunicateController::allgather(const Requests &requests) {
    assert(!requests.empty());
    const auto &request = requests[0];
    return getHandler(request->communicator()).allgather(requests);
}

StatusCode RingTokenCommunicateController::broadcast(const Requests &requests) {
    assert(!requests.empty());
    const auto &request = requests[0];
    return getHandler(request->communicator()).broadcast(requests);
}

RingTokenCommunicateController::~RingTokenCommunicateController() {
    GLOBAL_INFO_WITH_THREAD_ID("total handlers: " << handlerMap_.size())
    pthread_rwlock_wrlock(&rwlock_);
    // 由于是共享指针，所以可以先清除handlerMap_, 然后再按handler新建的顺序进行析构
    handlerMap_.clear();
    for (auto iter = orderedByEnteringHandlerMap_.begin(); iter != orderedByEnteringHandlerMap_.end(); ++iter) {
        GLOBAL_INFO_WITH_THREAD_ID("deleting RingTokenCommunicateHandler-" << iter->second.get())
        orderedByEnteringHandlerMap_.erase(iter);
        GLOBAL_INFO_WITH_THREAD_ID("deletind RingTokenCommunicateHandler-" << iter->second.get())
    }
    pthread_rwlock_unlock(&rwlock_);
    pthread_rwlock_destroy(&rwlock_);
}

RingTokenCommunicateHandler &RingTokenCommunicateController::getHandler(
        const std::shared_ptr<Communicator> &communicator) {
    pthread_rwlock_rdlock(&rwlock_);
    auto id = communicator->id();
    auto iter = handlerMap_.find(id);
    if (iter != handlerMap_.end()) {
        auto &handler = *iter->second;
        pthread_rwlock_unlock(&rwlock_);
        return handler;
    }
    pthread_rwlock_unlock(&rwlock_);
    // 没有需要的Handler, 需要修改Map, 重新获取写锁
    pthread_rwlock_wrlock(&rwlock_);
    iter = handlerMap_.find(id);
    // 需要重新判断是否有了适合的Handler
    if (iter != handlerMap_.end()) {
        auto &handler = *iter->second;
        pthread_rwlock_unlock(&rwlock_);
        return handler;
    }
    // 否则新建一个Handler
    auto handler = newHandler(communicator);
    handlerMap_.emplace(id, handler);
    orderedByEnteringHandlerMap_.emplace(orderedByEnteringHandlerMap_.size(), handler);
    pthread_rwlock_unlock(&rwlock_);
    return *handler;
}

std::shared_ptr<RingTokenCommunicateHandler> RingTokenCommunicateController::newHandler(
        const std::shared_ptr<Communicator> &communicator) {
    CALLING_ABSTRACT_INTERFACE_ERROR("RingTokenCommunicateController::newHandler"
                                     "(const std::shared_ptr<Communicator> &communicator)");
}


}}}}