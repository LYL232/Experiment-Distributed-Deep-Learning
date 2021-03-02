//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H

#include <cstring>
#include <iostream>

namespace lyl232 { namespace experiment { namespace ddl {

struct Message {
    char *msg_ptr;
    int senderRank;

    Message(const char *msg, int sender) : msg_ptr(nullptr), senderRank(sender) {
        auto len = strlen(msg);
        msg_ptr = new char[len + 1];
        memcpy(msg_ptr, msg, len);
        msg_ptr[len] = 0;
    }

    Message(const Message &other) = delete;

    Message(Message &&other) : msg_ptr(other.msg_ptr), senderRank(other.senderRank) {
        other.msg_ptr = nullptr;
    }

    ~Message() {
        delete[] msg_ptr;
    }
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
