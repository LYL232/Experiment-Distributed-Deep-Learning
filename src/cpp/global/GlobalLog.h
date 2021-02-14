//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOG_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOG_H

#include <memory>
#include <map>
#include <thread>
#include <ostream>
#include <sstream>
#include <functional>

namespace lyl232 { namespace experiment { namespace ddl {

class GlobalLog {
public:
    GlobalLog(std::ostream &stream, std::function<void()> streamDestructor);

    std::ostringstream &thisThreadLogStream() const noexcept;

    ~GlobalLog();

    void operator()(std::ostringstream &stream) const noexcept;

private:
    std::ostream &stream_;
    std::function<void()> streamDestructor_;

    mutable pthread_rwlock_t rwlock_;
    mutable std::map<std::thread::id, std::ostringstream *> threadLogStream_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOG_H
