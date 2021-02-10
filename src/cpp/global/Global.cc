//
// Created by LYL232 on 2021/2/5.
//

#include <assert.h>
#include "global/Global.h"
#include "communicate/tensor/allreduce/TensorsAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl {

Global *Global::instancePtr_ = nullptr;

Global::Global() :
        log_(nullptr), logStreamDestructor_([]() {}),
        threadLogStream_(),
        rwlock_(PTHREAD_RWLOCK_INITIALIZER),
        communication_(nullptr),
        allreduceController_(nullptr),
        initialized_(false) {
}

void Global::initialize(
        Global *singleton,
        std::ostream *log,
        std::function<void()> logStreamDestructor,
        std::shared_ptr<CommunicationBackend> communicationBackend,
        tensorsallreduce::TensorsAllreduceController *allreduceController) {
    singleton->log_ = log;
    singleton->logStreamDestructor_ = logStreamDestructor;
    singleton->communication_ = communicationBackend;
    singleton->allreduceController_ = allreduceController;
    singleton->initialized_ = true;
    instancePtr_ = singleton;
    GLOBAL_INFO_WITH_RANK("Global initialized");
}

Global::~Global() {
    GLOBAL_INFO_WITH_RANK("Global finalizing");
    // 之所以使用指针来指向, 是因为需要在delete communicationBackend_之前确认其他的通信资源得到释放(比如MPI), 而
    // delete指针可以做到这一点
    delete allreduceController_;
    communication_->finalize();
    GLOBAL_INFO_WITH_RANK("Global finalized");
    logStreamDestructor_();
    pthread_rwlock_destroy(&rwlock_);
    for (auto iter = threadLogStream_.begin(); iter != threadLogStream_.end(); ++iter) {
        delete iter->second;
    }
    instancePtr_ = nullptr;
}

void Global::log(std::ostringstream &stream) const noexcept {
    pthread_rwlock_wrlock(&rwlock_);
    *log_ << stream.str() << std::endl;
    pthread_rwlock_unlock(&rwlock_);
    stream.str(std::string());
    stream.clear();
}

std::ostringstream &Global::thisThreadLogStream() const noexcept {
    std::ostringstream *ptr;
    pthread_rwlock_rdlock(&rwlock_);
    auto iter = threadLogStream_.find(std::this_thread::get_id());
    if (iter == threadLogStream_.end()) {
        pthread_rwlock_unlock(&rwlock_);
        pthread_rwlock_wrlock(&rwlock_);
        ptr = new std::ostringstream();
        threadLogStream_.emplace(std::this_thread::get_id(), ptr);
    } else {
        ptr = iter->second;
    }
    pthread_rwlock_unlock(&rwlock_);
    return *ptr;
}

Global &Global::get() noexcept {
    while (instancePtr_ == nullptr) {
        std::this_thread::sleep_for(std::chrono::microseconds (10));
    }
    return *instancePtr_;
}

int Global::processes() const noexcept {
    return communication_->processes();
}

int Global::processRank() const noexcept {
    return communication_->processRank();
}

tensorsallreduce::TensorsAllreduceController &Global::allreduceController() const noexcept {
    assert(allreduceController_ != nullptr);
    return *allreduceController_;
}

CommunicationBackend &Global::communication() const noexcept {
    assert(communication_ != nullptr);
    return *communication_;
}

bool Global::initialized() const noexcept {
    return initialized_;
}

}}}