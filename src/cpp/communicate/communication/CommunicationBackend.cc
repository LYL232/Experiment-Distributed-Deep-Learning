//
// Created by LYL232 on 2021/2/11.
//

#include <assert.h>
#include "communicate/communication/CommunicationBackend.h"
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

CommunicationBackend::CommunicationBackend() {}

CommunicationBackend::~CommunicationBackend() {}

int CommunicationBackend::processes() const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "CommunicationBackend::processes()");
}

int CommunicationBackend::processRank() const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "CommunicationBackend::processRank()");
}

StatusCode CommunicationBackend::allreduce(
        void *sendBuffer, void *recvBuffer, size_t elements, DataType dtype,
        AllreduceOperation op) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "CommunicationBackend::allreduce("
            "void *sendBuffer, void *recvBuffer,"
            " size_t elements, DataType dtype, "
            " Operation op)");
}

StatusCode CommunicationBackend::broadcast(
        void *buffer, size_t elements, DataType dtype,
        int rootRank) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "CommunicationBackend::broadcast("
            "void *buffer,"
            " size_t elements, DataType dtype, "
            " int rootRank)");
}

}}}