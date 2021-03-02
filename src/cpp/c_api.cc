//
// Created by LYL232 on 2021/2/7.
//
#include "global/Global.h"
#include "communicate/message/MessageController.h"
#include "c_api.h"

namespace lyl232 { namespace experiment { namespace ddl {
int processes() {
    return Global::get().processes();
}

int process_rank() {
    return Global::get().processRank();
}

Message *listen_message() {
    return Global::get().messageController().listen();
}

void destroy_message(Message *messagePtr) {
    delete messagePtr;
}

void send_message(const char *msg, int receiverRank) {
    auto &global = Global::get();
    global.messageController().sendMessage(
            Message(msg, global.processRank()),
            receiverRank
    );
}

}}}