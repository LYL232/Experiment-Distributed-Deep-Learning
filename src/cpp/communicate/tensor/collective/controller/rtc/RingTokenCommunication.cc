//
// Created by LYL232 on 2021/2/13.
//

#include "communicate/tensor/collective/controller/rtc/RingTokenCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

RingTokenCommunication::RingTokenCommunication(std::shared_ptr<Communicator> communicator) :
        communicator_(std::move(communicator)) {}

void
RingTokenCommunication::communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenCommunication::communicationSendTokenTo"
            "(int receiver, const std::shared_ptr<Token> &token)");
}

std::shared_ptr<Token> RingTokenCommunication::communicationReceiveTokenFrom(int sender) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenCommunication::"
            "communicationReceiveTokenFrom(int sender)");
}

StatusCode
RingTokenCommunication::allreduceRequests(const Requests &requests) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenCommunication::allreduceRequests("
            "const std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>>"
            " &requests)"
    );
}

StatusCode
RingTokenCommunication::broadcastRequests(const Requests &requests) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenCommunication::broadcastRequests("
            "const std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>>"
            " &requests)"
    );
}


}}}}