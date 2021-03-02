//
// Created by LYL232 on 2021/2/16.
//

#include "mpi.h"
#include "global/Global.h"
#include "communicate/tensor/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

double MPIBlockedEnd2EndCommunication::inflateFactor_ = 1.5;

MPIBlockedEnd2EndCommunication::MPIBlockedEnd2EndCommunication(
        std::shared_ptr<MPIBackend> backend) :
        sendBuffer_(nullptr), receiveBuffer_(nullptr),
        sendBufferSize_(0), receiveBufferSize_(0),
        backend_(backend), statusBuffer_(),
        sendMutex_(PTHREAD_MUTEX_INITIALIZER),
        receiveMutex_(PTHREAD_MUTEX_INITIALIZER) {}

StatusCode MPIBlockedEnd2EndCommunication::send(
        const TensorSendCommunicateRequest &request) const {
    pthread_mutex_lock(&sendMutex_);
    checkSendBuffer_(request.tensorSize());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensor to send buffer")
#endif
    memcpy(sendBuffer_, request.requestingTensorData(), request.tensorSize());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sending tensor: " << request.key() << " to rank: " << request.receiver())
#endif
    MPI_Send(
            sendBuffer_,
            request.elements(),
            MPIBackend::DataType2MPIType(request.dtype()),
            request.receiver(),
            MPIBackend::MPI_TAG_BCC_COMMUNICATE,
            MPI_COMM_WORLD
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
    pthread_mutex_lock(&receiveMutex_);
    checkReceiveBuffer_(request.tensorSize());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("mpi receiving Tensor: " << request.key())
#endif
    MPI_Recv(
            receiveBuffer_,
            request.elements(),
            MPIBackend::DataType2MPIType(request.dtype()),
            request.sender(),
            MPIBackend::MPI_TAG_BCC_COMMUNICATE,
            MPI_COMM_WORLD,
            &statusBuffer_
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID(
            "mpi received Tensor: " << request.key() <<
                                    ", copying memory from receive buffer to output tensor")
#endif
    memcpy(request.requestingTensorData(), receiveBuffer_, request.tensorSize());
    pthread_mutex_unlock(&receiveMutex_);
    // todo: check status
    request.done(STATUS_OK);
    return STATUS_OK;
}

void MPIBlockedEnd2EndCommunication::checkSendBuffer_(size_t bytesRequire) const {
    if (sendBufferSize_ < bytesRequire) {
        delete[]sendBuffer_;
        sendBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        sendBuffer_ = new char[sendBufferSize_];
    }
}

void MPIBlockedEnd2EndCommunication::checkReceiveBuffer_(size_t bytesRequire) const {
    if (receiveBufferSize_ < bytesRequire) {
        delete[]receiveBuffer_;
        receiveBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        receiveBuffer_ = new char[receiveBufferSize_];
    }
}


MPIBlockedEnd2EndCommunication::~MPIBlockedEnd2EndCommunication() {
    delete[]sendBuffer_;
    delete[]receiveBuffer_;
}

}}}}