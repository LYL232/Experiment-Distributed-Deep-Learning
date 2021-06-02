//
// Created by LYL232 on 2021/6/2.
//

#include "communicate/message/Message.h"
#include "global/initialize.h"

namespace lyl232 { namespace experiment { namespace ddl {

std::shared_ptr<HeapMemoryManager> Message::memManager_(heapMemoryManagerGetter());

Message::Message(const char *msg, int sender, size_t len) :
        msg_ptr(nullptr), senderRank(sender), length(len) {
    msg_ptr = (char *) memManager_->allocateBytes(length + 1);
    memcpy(msg_ptr, msg, length);
    msg_ptr[length] = 0;
}

Message::Message(Message &&other) noexcept:
        msg_ptr(other.msg_ptr), senderRank(other.senderRank), length(other.length) {
    other.msg_ptr = nullptr;
}

Message::~Message() {
    memManager_->deallocateBytes(msg_ptr);
}

}}}