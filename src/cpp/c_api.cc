//
// Created by LYL232 on 2021/2/7.
//
#include "global/Global.h"
#include "communicate/message/MessageController.h"
#include "c_api.h"

namespace lyl232 { namespace experiment { namespace ddl {


int communicator_rank(Communicator::ID id) {
    return Global::get().getCommunicator(id)->rank();
}

int communicator_size(Communicator::ID id) {
    return Global::get().getCommunicator(id)->size();
}

Communicator::ID world_communicator() {
    return Global::get().worldCommunicator()->id();
}

void destroy_message(Message *messagePtr) {
    delete messagePtr;
}

Message *listen_message(Communicator::ID id) {
    auto &global = Global::get();
    return global.messageController().listen(*global.getCommunicator(id));
}

void send_message(const char *msg, int receiverRank, Communicator::ID id, size_t len) {
    auto &global = Global::get();
    const auto &comm = global.getCommunicator(id);
    global.messageController().sendMessage(
            Message(msg, comm->rank(), len),
            receiverRank, comm
    );
}

Communicator::ID split_communicator(Communicator::ID id, int color, int key) {
    return Global::get().getCommunicator(id)->split(color, key)->id();
}

void detach_communicator(Communicator::ID id) {
    Global::get().detachCommunicator(id);
}

void py_info(const char *logStr) {
    GLOBAL_INFO_WITH_THREAD_ID("[py]: " << logStr)
}

void py_debug(const char *logStr) {
    GLOBAL_DEBUG_WITH_THREAD_ID("[py]: " << logStr)
}

void py_error(const char *logStr) {
    GLOBAL_ERROR_WITH_THREAD_ID("[py]: " << logStr)
}

}}}