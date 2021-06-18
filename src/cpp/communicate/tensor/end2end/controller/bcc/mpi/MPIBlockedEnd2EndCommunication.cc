//
// Created by LYL232 on 2021/2/16.
//

#include "mpi.h"
#include "global/Global.h"
#include "communicate/tensor/end2end/controller/bcc/mpi/MPIBlockedEnd2EndCommunication.h"
#include "communicate/backend/mpi/MPICommunicator.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

MPIBlockedEnd2EndCommunication::MPIBlockedEnd2EndCommunication(
        std::shared_ptr<MPIBackend> backend) :
        backend_(std::move(backend)), statusBuffer_() {}

StatusCode MPIBlockedEnd2EndCommunication::send(
        const TensorSendCommunicateRequest &request) const {
    auto &tensor = *request.requestingTensor();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sending tensor: " << request.key() << " to rank: " << request.receiver())
#endif
    const auto &communicator = dynamic_cast<const MPICommunicator &>(*request.communicator());
    MPI_Send(
            request.requestingTensor()->data(),
            tensor.elements(),
            MPIBackend::DataType2MPIType(tensor.dtype()),
            request.receiver(),
            request.tag() + MPIBackend::MPI_CUSTOM_TAG_BEGIN,
            communicator.mpiComm()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent tensor: " << request.key() << " to rank: " << request.receiver())
#endif
    // todo: check status
    request.done(STATUS_OK);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
    GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request.key() << " done send")
#endif
    return STATUS_OK;
}

StatusCode MPIBlockedEnd2EndCommunication::receive(
        const TensorReceiveCommunicateRequest &request) const {
    auto &tensor = *request.requestingTensor();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("mpi receiving Tensor: " << request.key())
#endif
    const auto &communicator = dynamic_cast<const MPICommunicator &>(*request.communicator());
    MPI_Recv(
            tensor.data(),
            tensor.elements(),
            MPIBackend::DataType2MPIType(tensor.dtype()),
            request.sender(),
            request.tag() + MPIBackend::MPI_CUSTOM_TAG_BEGIN,
            communicator.mpiComm(),
            &statusBuffer_
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_BLOCKED_END2END_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("mpi received Tensor: " << request.key() << "\n")
#endif
    // todo: check status
    request.done(STATUS_OK);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
    GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request.key() << " done receive")
#endif
    return STATUS_OK;
}


MPIBlockedEnd2EndCommunication::~MPIBlockedEnd2EndCommunication() {}

}}}}