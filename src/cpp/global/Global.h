//
// Created by LYL232 on 2021/2/5.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H

#include <pthread.h>
#include <thread>
#include <chrono>
#include "global/GlobalLog.h"
#include "global/HeapMemoryManager.h"
#include "communicate/backend/Communicator.h"


namespace lyl232 { namespace experiment { namespace ddl {

class CommunicationBackend;

class TensorsCollectiveCommunicateController;

class TensorEnd2EndCommunicateController;

class MessageController;

/**
 * 单例模式: 全局对象类
 */
class Global {
    // 开放communicatorMap_的访问权限给Communicator类, 主要用于分割通信域时能够注册分割出的通信域的ID
    friend class Communicator;

public:
    Global() = delete;

    Global(const Global &) = delete;

    Global(Global &&) = delete;

    std::shared_ptr<Communicator> worldCommunicator() const noexcept;

    CommunicationBackend &communicationBackend() const noexcept;

    TensorsCollectiveCommunicateController &collectiveCommunicateController() const noexcept;

    TensorEnd2EndCommunicateController &end2EndCommunicateController() const noexcept;

    MessageController &messageController() const noexcept;

    HeapMemoryManager &heapMemoryManager() const noexcept;

    const std::shared_ptr<Communicator> &getCommunicator(Communicator::ID) const noexcept;

    /**
     * python端不再需要一个通信域对象时调用的方法, 这将会导致communicatorMap_内的shared_ptr被析构,
     * 不再保持这个通信域
     * @param id
     */
    void detachCommunicator(Communicator::ID id) const noexcept;

    ~Global();

    static Global &get() noexcept;

    static const std::shared_ptr<GlobalLog> log;

private:
    Global(
            std::shared_ptr<HeapMemoryManager> heapMemoryManager,
            std::shared_ptr<CommunicationBackend> communicationBackend,
            std::shared_ptr<TensorsCollectiveCommunicateController> collectiveController,
            std::shared_ptr<TensorEnd2EndCommunicateController> end2EndController,
            std::shared_ptr<MessageController> messageController
    ) noexcept;

    mutable std::shared_ptr<HeapMemoryManager> heapMemoryManager_;
    mutable std::shared_ptr<CommunicationBackend> communicationBackend_;
    mutable std::shared_ptr<TensorsCollectiveCommunicateController>
            collectiveCommunicateController_;
    mutable std::shared_ptr<TensorEnd2EndCommunicateController>
            end2EndCommunicateController_;
    mutable std::shared_ptr<MessageController> messageController_;

    // 这个map设计成指针是因为这个map的析构可能在communicationBackend_之后, 这样如果
    // Communicator被析构可能会出异常, 设计成指针就可以控制其何时被析构
    // todo: 可以尝试一下写在Backend对象中的析构函数中
    mutable std::map<Communicator::ID, std::shared_ptr<Communicator>> *communicatorMap_;
    static Global instance_;
};

#define STREAM_WITH_THREAD_ID(s) \
    "[" << std::this_thread::get_id() << "]" << s

#define GLOBAL_INFO(s) { \
        auto &stream = Global::log->thisThreadLogStream(); \
        stream<< "[INFO]" << s;\
        (*Global::log)(stream); \
    }

#define GLOBAL_INFO_WITH_THREAD_ID(s) \
    GLOBAL_INFO(STREAM_WITH_THREAD_ID(s))

#define GLOBAL_DEBUG(s) { \
        auto &stream = Global::log->thisThreadLogStream(); \
        stream<< "[DEBUG]" << s;\
        (*Global::log)(stream); \
    }

#define GLOBAL_DEBUG_WITH_THREAD_ID(s) \
    GLOBAL_DEBUG(STREAM_WITH_THREAD_ID(s))

#define GLOBAL_ERROR(s) { \
        auto &stream = Global::log->thisThreadLogStream(); \
        stream<< "[ERROR]" << s;\
        (*Global::log)(stream); \
    }

#define GLOBAL_ERROR_WITH_THREAD_ID(s) \
    GLOBAL_ERROR(STREAM_WITH_THREAD_ID(s))

#define SEC_TIME_LOG(s) GLOBAL_INFO("[TIME-" << \
    std::chrono::duration_cast<std::chrono::seconds>( \
        std::chrono::system_clock::now().time_since_epoch()).count() \
        << "]: " << s)

#define MS_TIME_LOG(s) GLOBAL_INFO("[TIME-" << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
        std::chrono::system_clock::now().time_since_epoch()).count() \
        << "]: " << s)

#define US_TIME_LOG(s) GLOBAL_INFO("[TIME-" << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
        std::chrono::system_clock::now().time_since_epoch()).count() \
        << "]: " << s)

#define NS_TIME_LOG(s) GLOBAL_INFO("[TIME-" << \
    std::chrono::duration_cast<std::chrono::nanoseconds>( \
        std::chrono::system_clock::now().time_since_epoch()).count() \
        << "]: " << s)

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
