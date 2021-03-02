//
// Created by LYL232 on 2021/2/27.
//

#include "def.h"
#include "communicate/message/MessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

void MessageController::sendMessage(const Message &message, int receiver) {
    CALLING_ABSTRACT_INTERFACE_ERROR("MessageController::sendMessage(const Message &message, int receiver)");
}

Message* MessageController::listen() {
    CALLING_ABSTRACT_INTERFACE_ERROR("MessageController::listen()");
}

}}}