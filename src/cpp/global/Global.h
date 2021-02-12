//
// Created by LYL232 on 2021/2/5.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H

#include <pthread.h>
#include <sstream>
#include <functional>
#include <thread>
#include <map>
#include "global/GlobalLogStream.h"
#include "communicate/communication/CommunicationBackend.h"
#include "communicate/tensor/allreduce/TensorsAllreduceController.h"


namespace lyl232 { namespace experiment { namespace ddl {

class CommunicationBackend;

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

    static Global &get() noexcept;

    CommunicationBackend &communicationBackend() const noexcept;

    tensorsallreduce::TensorsAllreduceController &allreduceController() const noexcept;

    void log(std::ostringstream &stream) noexcept;

    std::ostringstream &thisThreadLogStream() const noexcept;

    ~Global();

private:

    Global(
            std::shared_ptr<CommunicationBackend> communicationBackend,
            std::shared_ptr<tensorsallreduce::TensorsAllreduceController> allreduceController,
            std::shared_ptr<GlobalLogStream> logStream
    );


    mutable std::map<std::thread::id, std::ostringstream *> threadLogStream_;
    mutable pthread_rwlock_t rwlock_;
    mutable std::shared_ptr<CommunicationBackend> communicationBackend_;
    mutable std::shared_ptr<tensorsallreduce::TensorsAllreduceController>
            allreduceController_;
    std::shared_ptr<GlobalLogStream> logStream_;

    static Global instance_;
};

#define STREAM_WITH_RANK(s, rank) \
    "[rank " << rank << "]: " << s

#define STREAM_WITH_THREAD_ID(s) \
    "[" << std::this_thread::get_id() << "]: " << s

#define GLOBAL_INFO(s) \
    Global::get().thisThreadLogStream() << "[INFO]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_INFO_WITH_RANK(s) \
    GLOBAL_INFO(STREAM_WITH_RANK(s, Global::get().processRank()))

#define GLOBAL_INFO_WITH_RANK_THREAD_ID(s) \
    GLOBAL_INFO_WITH_RANK(STREAM_WITH_THREAD_ID(s))

#define GLOBAL_DEBUG(s) \
    Global::get().thisThreadLogStream() << "[DEBUG]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_DEBUG_WITH_RANK(s) \
    GLOBAL_DEBUG(STREAM_WITH_RANK(s, Global::get().processRank()))

#define GLOBAL_DEBUG_WITH_RANK_THREAD_ID(s) \
    GLOBAL_DEBUG_WITH_RANK(STREAM_WITH_THREAD_ID(s))

#define GLOBAL_ERROR(s) \
    Global::get().thisThreadLogStream() << "[ERROR]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_ERROR_WITH_RANK(s) \
    GLOBAL_ERROR(STREAM_WITH_RANK(s, Global::get().processRank()))

#define GLOBAL_ERROR_WITH_RANK_THREAD_ID(s) \
    GLOBAL_ERROR_WITH_RANK(STREAM_WITH_THREAD_ID(s))
}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
