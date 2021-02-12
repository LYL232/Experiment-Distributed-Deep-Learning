//
// Created by LYL232 on 2021/2/12.
//

#include "RingTokenAllreduceCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

void
RingTokenAllreduceCommunication::communicationSendTokenTo(
        int receiver, const std::shared_ptr<Token> &token) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenAllreduceCommunication::communicationSendTokenTo"
            "(int receiver, const std::shared_ptr<Token> &token)");
}

std::shared_ptr<Token> RingTokenAllreduceCommunication::communicationReceiveTokenFrom(int sender) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenAllreduceCommunication::"
            "communicationReceiveTokenFrom(int sender)");
}

StatusCode
RingTokenAllreduceCommunication::allreduceRequests(
        const std::map<std::string, TensorAllreduceRequest *> &requests,
        size_t elements, size_t byteSize) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenAllreduceCommunication::allreduceRequests("
            "const std::map<std::string, TensorAllreduceRequest *> &requests,"
            "size_t elements, size_t byteSize)"
    );
}
}}}}}