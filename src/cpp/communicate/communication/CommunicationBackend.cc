//
// Created by LYL232 on 2021/2/11.
//

#include <assert.h>
#include "CommunicationBackend.h"
#include "def.h"

namespace lyl232 { namespace experiment { namespace ddl {

int CommunicationBackend::processes() const noexcept {
    assert(initialzed_);
    return processes_;
}

int CommunicationBackend::processRank() const noexcept {
    assert(initialzed_);
    return processRank_;
}

void CommunicationBackend::initialize(int processes, int processRank) {
    processes_ = processes;
    processRank_ = processRank;
    initialzed_ = true;
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

void CommunicationBackend::finalize() {
    CALLING_ABSTRACT_INTERFACE_ERROR("CommunicationBackend::finalize()");
}
}}}