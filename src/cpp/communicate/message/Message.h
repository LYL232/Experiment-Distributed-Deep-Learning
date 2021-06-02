//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H

#include <cstring>
#include <iostream>
#include "global/Global.h"

namespace lyl232 { namespace experiment { namespace ddl {

struct Message {
    char *msg_ptr;
    int senderRank;
    size_t length;

    Message(const char *msg, int sender, size_t len);

    Message(const Message &other) = delete;

    Message(Message &&other) noexcept;

    ~Message();

private:
    static std::shared_ptr<HeapMemoryManager> memManager_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
