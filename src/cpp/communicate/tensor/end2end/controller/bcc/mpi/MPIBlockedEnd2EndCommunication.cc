//
// Created by LYL232 on 2021/2/16.
//

#include "mpi.h"
#include "global/Global.h"
#include "communicate/tensor/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"
#include "communicate/backend/mpi/MPICommunicator.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

double MPIBlockedEnd2EndCommunication::inflateFactor_ = 1.5;

MPIBlockedEnd2EndCommunication::MPIBlockedEnd2EndCommunication(
        std::shared_ptr<MPIBackend> backend) :
        receiveBuffer_(nullptr), receiveBufferSize_(0),
        backend_(std::move(backend)), statusBuffer_(),
        sendMutex_(PTHREAD_MUTEX_INITIALIZER),
        receiveMutex_(PTHREAD_MUTEX_INITIALIZER) {}

StatusCode MPIBlockedEnd2EndCommunication::send(
        const TensorSendCommunicateRequest &request) const {
    auto &tensor = *request.requestingTensor();
    pthread_mutex_lock(&sendMutex_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sending tensor: " << request.key() << " to rank: " << request.receiver())
#endif
    const auto &communicator = dynamic_cast<const MPICommunicator &>(*request.communicator());
    MPI_Send(
            request.requestingTensor()->data(),
            tensor.elements(),
            MPIBackend::DataType2MPIType(tensor.dtype()),
            request.receiver(),
            MPIBackend::MPI_TAG_BCC_COMMUNICATE,
            communicator.mpiComm()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent tensor: " << request.key() << " to rank: " << request.receiver())
#endif
    pthread_mutex_unlock(&sendMutex_);
    // todo: check status
    request.done(STATUS_OK);
    return STATUS_OK;
}

StatusCode MPIBlockedEnd2EndCommunication::receive(
        const TensorReceiveCommunicateRequest &request) const {
    auto &tensor = *request.requestingTensor();
    pthread_mutex_lock(&receiveMutex_);
    checkReceiveBuffer_(tensor.byteSize());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("mpi receiving Tensor: " << request.key())
#endif
    const auto &communicator = dynamic_cast<const MPICommunicator &>(*request.communicator());
    MPI_Recv(
            receiveBuffer_,
            tensor.elements(),
            MPIBackend::DataType2MPIType(tensor.dtype()),
            request.sender(),
            MPIBackend::MPI_TAG_BCC_COMMUNICATE,
            communicator.mpiComm(),
            &statusBuffer_
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID(
            "mpi received Tensor: " << request.key() <<
                                    ", copying memory from receive buffer to output tensor")
#endif
    memcpy(tensor.data(), receiveBuffer_, tensor.byteSize());
    pthread_mutex_unlock(&receiveMutex_);
    // todo: check status
    request.done(STATUS_OK);
    return STATUS_OK;
}

void MPIBlockedEnd2EndCommunication::checkReceiveBuffer_(size_t bytesRequire) const {
    if (receiveBufferSize_ < bytesRequire) {
        memManager_->deallocateBytes(receiveBuffer_);
        receiveBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        receiveBuffer_ = (char *) memManager_->allocateBytes(receiveBufferSize_);
    }
}


MPIBlockedEnd2EndCommunication::~MPIBlockedEnd2EndCommunication() {
    memManager_->deallocateBytes(receiveBuffer_);
}

}}}}