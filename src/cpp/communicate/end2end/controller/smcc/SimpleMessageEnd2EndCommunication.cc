//
// Created by LYL232 on 2021/2/18.
//

#include "communicate/end2end/controller/smcc/SimpleMessageEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace smcc {

StatusCode SimpleMessageEnd2EndCommunication::sendMessage(const Message &message) {
    CALLING_ABSTRACT_INTERFACE_ERROR("SimpleMessageEnd2EndCommunication::sendMessage(const Message &message)");
}

SimpleMessageEnd2EndCommunication::SharedMessage SimpleMessageEnd2EndCommunication::listen() {
    CALLING_ABSTRACT_INTERFACE_ERROR("SimpleMessageEnd2EndCommunication::listen()");
}

}}}}