//
// Created by LYL232 on 2021/2/5.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H

#include "mpi.h"
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <functional>
#include <thread>
#include <map>
#include <atomic>


namespace lyl232 { namespace experiment { namespace ddl {

namespace tensorsallreduce {
class TensorsAllreduceController;
}

/**
 * 单例模式: 全局对象类
 */
class Global {
public:
    int MPIWorldSize() const { return MPIWorldSize_; }

    int MPIWorldRank() const { return MPIWorldRank_; }

    static const Global &get();

//    const TensorCommunicateManager &communicationManager() const { return *communicationManager_; }

    tensorsallreduce::TensorsAllreduceController &controller() const {
        return *controller_;
    };

    void log(std::ostringstream &stream, bool clear = true) const;

    Global(const Global &) = delete;

    ~Global();

    std::ostringstream &thisThreadLogStream() const;

private:
    int MPIWorldRank_, MPIWorldSize_;

    std::ostream *log_;
    std::function<void()> logStreamDestructor_;

    mutable std::map<std::thread::id, std::ostringstream *> threadLogStream_;

    mutable tensorsallreduce::TensorsAllreduceController *controller_ = nullptr;

    mutable std::atomic_bool initialized;

    mutable pthread_rwlock_t rwlock_ = PTHREAD_RWLOCK_INITIALIZER;

    static Global instance_;

    static void init();

    Global() {
        initialized = false;
        init();
        assert(initialized);
    };
};

#define GLOBAL_INFO(s) \
    Global::get().thisThreadLogStream() << "[INFO]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_INFO_WITH_RANK(s) \
    GLOBAL_INFO("[rank " << Global::get().MPIWorldRank() << "]: " << s)

#define GLOBAL_INFO_WITH_RANK_THREAD_ID(s) \
    GLOBAL_INFO_WITH_RANK("[" << std::this_thread::get_id() << "]: " << s)

#define GLOBAL_DEBUG(s) \
    Global::get().thisThreadLogStream() << "[DEBUG]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_DEBUG_WITH_RANK(s) \
    GLOBAL_DEBUG("[rank " << Global::get().MPIWorldRank() << "]: " << s)

#define GLOBAL_DEBUG_WITH_RANK_THREAD_ID(s) \
    GLOBAL_DEBUG_WITH_RANK("[" << std::this_thread::get_id() << "]: " << s)

#define GLOBAL_ERROR(s) \
    Global::get().thisThreadLogStream() << "[ERROR]" << s;\
    Global::get().log(Global::get().thisThreadLogStream())

#define GLOBAL_ERROR_WITH_RANK(s) \
    GLOBAL_ERROR("[rank " << Global::get().MPIWorldRank() << "]: " << s)

#define GLOBAL_ERROR_WITH_RANK_THREAD_ID(s) \
    GLOBAL_ERROR_WITH_RANK("[" << std::this_thread::get_id() << "]: " << s)
}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBAL_H
