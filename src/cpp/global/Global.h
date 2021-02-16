//
// Created by LYL232 on 2021/2/5.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H

#include <pthread.h>
#include <thread>
#include "global/GlobalLog.h"


namespace lyl232 { namespace experiment { namespace ddl {

class CommunicationBackend;

class TensorsCollectiveCommunicateController;
class TensorEnd2EndCommunicateController;

/**
 * 单例模式: 全局对象类
 */
class Global {
public:
    Global() = delete;

    Global(const Global &) = delete;

    Global(Global &&) = delete;

    int processes() const noexcept;

    int processRank() const noexcept;

    CommunicationBackend &communicationBackend() const noexcept;

    TensorsCollectiveCommunicateController &collectiveCommunicateController() const noexcept;

    TensorEnd2EndCommunicateController &end2EndCommunicateController() const noexcept;

    ~Global();

    static Global &get() noexcept;

    static const std::shared_ptr<GlobalLog> log;

private:
    Global(
            std::shared_ptr<CommunicationBackend> communicationBackend,
            std::shared_ptr<TensorsCollectiveCommunicateController> collectiveController,
            std::shared_ptr<TensorEnd2EndCommunicateController> end2EndController
    );

    mutable std::shared_ptr<CommunicationBackend> communicationBackend_;
    mutable std::shared_ptr<TensorsCollectiveCommunicateController>
            collectiveCommunicateController_;
    mutable std::shared_ptr<TensorEnd2EndCommunicateController>
            end2EndCommunicateController_;

    static Global instance_;
};

#define STREAM_WITH_THREAD_ID(s) \
    "[" << std::this_thread::get_id() << "]: " << s

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
}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
