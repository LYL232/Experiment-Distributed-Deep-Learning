//
// Created by LYL232 on 2021/2/12.
//

#include "global/GlobalLog.h"

namespace lyl232 { namespace experiment { namespace ddl {

GlobalLog::GlobalLog(
        std::ostream &stream,
        std::function<void()> streamDestructor
) : stream_(stream), streamDestructor_(streamDestructor),
    rwlock_(PTHREAD_RWLOCK_INITIALIZER), threadLogStream_() {
    stream_ << "GlobalLog initialized" << std::endl;
}

std::ostringstream &GlobalLog::thisThreadLogStream() const noexcept {
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


GlobalLog::~GlobalLog() {
    stream_ << "GlobalLog finalized" << std::endl;
    streamDestructor_();
    pthread_rwlock_destroy(&rwlock_);
    for (auto iter = threadLogStream_.begin(); iter != threadLogStream_.end(); ++iter) {
        delete iter->second;
    }
}

void GlobalLog::operator()(std::ostringstream &stream) const noexcept {
    pthread_rwlock_wrlock(&rwlock_);
    stream_ << stream.str() << std::endl;
    pthread_rwlock_unlock(&rwlock_);
    stream.str(std::string());
    stream.clear();
}

}}}