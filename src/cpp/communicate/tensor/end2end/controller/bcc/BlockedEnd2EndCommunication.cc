//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

StatusCode BlockedEnd2EndCommunication::send(
        const TensorSendCommunicateRequest &request) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "BlockedEnd2EndCommunication::"
            "send(const TensorSendCommunicateRequest &request)");
}

StatusCode BlockedEnd2EndCommunication::receive(
        const TensorReceiveCommunicateRequest &request) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "BlockedEnd2EndCommunication::"
            "receive(const TensorReceiveCommunicateRequest &request)");
}

}}}}