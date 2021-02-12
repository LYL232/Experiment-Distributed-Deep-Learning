//
// Created by LYL232 on 2021/2/5.
//

#include <assert.h>
#include "global/Global.h"
#include "global/initialize.h"

namespace lyl232 { namespace experiment { namespace ddl {


Global Global::instance_(
        communicationBackendGetter(),
        allreduceControllerGetter(),
        globalLogStreamGetter()
);

Global::Global(
        std::shared_ptr<CommunicationBackend> communicationBackend,
        std::shared_ptr<tensorsallreduce::TensorsAllreduceController> allreduceController,
        std::shared_ptr<GlobalLogStream> logStream
) : threadLogStream_(),
    rwlock_(PTHREAD_RWLOCK_INITIALIZER),
    communicationBackend_(communicationBackend),
    allreduceController_(allreduceController),
    logStream_(logStream) {
    logStream_->stream()
            << STREAM_WITH_RANK("Global initialized", communicationBackend_->processRank())
            << std::endl;
}

Global::~Global() {
    GLOBAL_INFO_WITH_RANK("Global finalizing");
    pthread_rwlock_destroy(&rwlock_);
    for (auto iter = threadLogStream_.begin(); iter != threadLogStream_.end(); ++iter) {
        delete iter->second;
    }
}

void Global::log(std::ostringstream &stream) noexcept {
    pthread_rwlock_wrlock(&rwlock_);
    logStream_->stream() << stream.str() << std::endl;
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
    return instance_;
}

int Global::processes() const noexcept {
    return communicationBackend_->processes();
}

int Global::processRank() const noexcept {
    return communicationBackend_->processRank();
}

tensorsallreduce::TensorsAllreduceController &Global::allreduceController() const noexcept {
    assert(allreduceController_ != nullptr);
    return *allreduceController_;
}

CommunicationBackend &Global::communicationBackend() const noexcept {
    return *communicationBackend_;
}

}}}