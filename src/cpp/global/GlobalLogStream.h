//
// Created by LYL232 on 2021/2/12.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOGSTREAM_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOGSTREAM_H

#include <memory>
#include <ostream>
#include <functional>

class GlobalLogStream {
public:
    GlobalLogStream(
            std::shared_ptr<std::ostream> stream,
            std::function<void()> streamDestructor
    ) : stream_(stream), streamDestructor_(streamDestructor) {};

    std::ostream &stream();

    virtual ~GlobalLogStream();

private:
    std::shared_ptr<std::ostream> stream_;
    std::function<void()> streamDestructor_;
};


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_GLOBALLOGSTREAM_H
