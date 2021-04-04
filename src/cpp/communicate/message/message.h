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
    size_t length;

    Message(const char *msg, int sender, size_t len) :
            msg_ptr(nullptr), senderRank(sender), length(len) {
        msg_ptr = new char[length + 1];
        memcpy(msg_ptr, msg, length);
        msg_ptr[length] = 0;
    }

    Message(const Message &other) = delete;

    Message(Message &&other) noexcept:
            msg_ptr(other.msg_ptr), senderRank(other.senderRank), length(other.length) {
        other.msg_ptr = nullptr;
    }

    ~Message() {
        delete[] msg_ptr;
    }
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
