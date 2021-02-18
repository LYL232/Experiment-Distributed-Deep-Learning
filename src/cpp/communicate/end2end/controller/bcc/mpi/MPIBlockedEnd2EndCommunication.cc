//
// Created by LYL232 on 2021/2/16.
//

#include "mpi.h"
#include "global/Global.h"
#include "communicate/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

double MPIBlockedEnd2EndCommunication::inflateFactor_ = 1.5;

MPIBlockedEnd2EndCommunication::MPIBlockedEnd2EndCommunication(
        std::shared_ptr<MPIBackend> backend) :
        buffer_(nullptr), bufferSize_(0), backend_(backend), statusBuffer_(),
        mutex_(PTHREAD_MUTEX_INITIALIZER) {}

StatusCode MPIBlockedEnd2EndCommunication::sendOrReceiveRequest(
        const TensorEnd2EndCommunicateRequest &request) const {
    pthread_mutex_lock(&mutex_);
    if (backend_->processRank() == request.sender()) {
        checkBuffer_(request.tensorSize());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
        GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensor to end2end communicate buffer")
#endif
        memcpy(buffer_, request.requestingTensorData(), request.tensorSize());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID("mpi sending tensor: " << request.key() << " to rank: " << request.receiver())
#endif
        MPI_Send(
                buffer_,
                request.elements(),
                MPIBackend::DataType2MPIType(request.dtype()),
                request.receiver(),
                MPIBackend::MPI_TAG_BCC_COMMUNICATE_AS_SENDER,
                MPI_COMM_WORLD
        );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID("mpi sent tensor: " << request.key() << " to rank: " << request.receiver())
#endif
        pthread_mutex_unlock(&mutex_);
    } else if (backend_->processRank() == request.receiver()) {
        checkBuffer_(request.tensorSize());
        MPI_Recv(
                buffer_,
                request.elements(),
                MPIBackend::DataType2MPIType(request.dtype()),
                request.sender(),
                MPIBackend::MPI_TAG_BCC_COMMUNICATE_AS_RECEIVER,
                MPI_COMM_WORLD,
                &statusBuffer_
        );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
        GLOBAL_INFO_WITH_THREAD_ID("copying memory from end2end communicate buffer to output tensor")
#endif
        memcpy(request.requestingTensorData(), buffer_, request.tensorSize());
        pthread_mutex_unlock(&mutex_);
    } else {
        using namespace std;
        string msg("received irrelevant request, sender: ");
        msg.append(to_string(request.sender())).append("receiver: ")
                .append(to_string(request.receiver())).append(", while handling rank is ")
                .append(to_string(backend_->processRank()));
        pthread_mutex_unlock(&mutex_);
        throw runtime_error(msg);
    }
    // todo: check status
    request.done(STATUS_OK);
    return STATUS_OK;
}

void MPIBlockedEnd2EndCommunication::checkBuffer_(size_t bytesRequire) const {
    if (bufferSize_ < bytesRequire) {
        delete[]buffer_;
        bufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        buffer_ = new char[bufferSize_];
    }
}


MPIBlockedEnd2EndCommunication::~MPIBlockedEnd2EndCommunication() {
    delete[]buffer_;
}

}}}}