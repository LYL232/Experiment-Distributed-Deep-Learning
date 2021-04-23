//
// Created by LYL232 on 2021/2/27.
//

#include "def.h"
#include "communicate/message/MessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

void MessageController::sendMessage(
        const Message &message, int receiver,
        const std::shared_ptr<Communicator> &communicator) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "MessageController::sendMessage(const Message &message, int receiver),"
            " const Message &message, int receiver, const std::shared_ptr<Communicator> &communicator");
}

Message *MessageController::listen(const Communicator &communicator) {
    CALLING_ABSTRACT_INTERFACE_ERROR("MessageController::listen(const Communicator &communicator)");
}

Message *MessageController::broadcastMessage(
        const Message &message, int root, const std::shared_ptr<Communicator> &communicator) {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "broadcastMessage("
            "const Message &message, int root, const std::shared_ptr<Communicator> &communicator)");
}

}}}