//
// Created by LYL232 on 2021/2/16.
//

#include "communicate/end2end/controller/bcc/BlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

StatusCode BlockedEnd2EndCommunication::sendOrReceiveRequest(
        const TensorEnd2EndCommunicateRequest &request) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "BlockedEnd2EndCommunication::"
            "sendOrReceiveRequest(const TensorEnd2EndCommunicateRequest &request)");
}
}}}}