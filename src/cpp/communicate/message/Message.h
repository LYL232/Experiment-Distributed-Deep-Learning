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

    /**
     * 这个构造函数会将msg的内容复制出来，不会影响msg
     * @param msg
     * @param sender
     * @param len
     */
    Message(const char *msg, int sender, size_t len);

    /**
     * 这个构造函数会在析构时释放msg指针
     * @param sender
     * @param len
     * @param msg
     */
    Message(int sender, size_t len, char *msg);

    Message(const Message &other) = delete;

    Message(Message &&other) noexcept;

    ~Message();

private:
    static std::shared_ptr<HeapMemoryManager> memManager_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MESSAGE_H
