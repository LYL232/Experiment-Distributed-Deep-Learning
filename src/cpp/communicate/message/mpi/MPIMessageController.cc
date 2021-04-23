//
// Created by LYL232 on 2021/2/27.
//

#include <cstring>
#include "global/Global.h"
#include "communicate/message/mpi/MPIMessageController.h"
#include "communicate/backend/mpi/MPICommunicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

double MPIMessageController::inflateFactor_ = 1.5;

MPIMessageController::MPIMessageController(std::shared_ptr<MPIBackend> backend) :
        mutex_(PTHREAD_MUTEX_INITIALIZER), backend_(std::move(backend)),
        buffer_(nullptr), bufferSize_(0), statusBuffer_() {}

Message *MPIMessageController::broadcastMessage(
        const Message &message, int root, const std::shared_ptr<Communicator> &communicator) {

    size_t len = message.length;
    if (communicator->rank() == root) {
        memcpy(buffer_, message.msg_ptr, len);
    }

    pthread_mutex_lock(&mutex_);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID(
            "before broadcasting message: " << message.msg_ptr << ", root " << root)
#endif
    const auto &comm = dynamic_cast<const MPICommunicator &>(*communicator);
    // 先传输字符串的长度
    MPI_Bcast(&len, sizeof(size_t), MPI_BYTE, root, comm.mpiComm());
    checkBuffer_(len);
    MPI_Bcast(buffer_, len, MPI_CHAR, root, comm.mpiComm());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi brocasted Message")
#endif
    auto *res = new Message(buffer_, root, len);
    pthread_mutex_unlock(&mutex_);
    return res;
}

void MPIMessageController::sendMessage(
        const Message &message, int receiver,
        const std::shared_ptr<Communicator> &communicator) {
    pthread_mutex_lock(&mutex_);
    size_t len = message.length;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID(
            "before sending message: " << message.msg_ptr << ", to " << receiver)
#endif
    const auto &comm = dynamic_cast<const MPICommunicator &>(*communicator);

    // 先传输字符串的长度
    MPI_Send(
            &len, sizeof(size_t), MPI_BYTE, receiver,
            MPIBackend::MPI_TAG_MESSAGE_META, comm.mpiComm()
    );
    MPI_Send(
            message.msg_ptr, len, MPI_CHAR, receiver,
            MPIBackend::MPI_TAG_MESSAGE_MSG, comm.mpiComm()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent Message")
#endif
    pthread_mutex_unlock(&mutex_);
}

Message *MPIMessageController::listen(const Communicator &communicator) {
    size_t len;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID("listen message")
#endif
    const auto &comm = dynamic_cast<const MPICommunicator &>(communicator);
    MPI_Recv(
            &len, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE,
            MPIBackend::MPI_TAG_MESSAGE_META,
            comm.mpiComm(), &statusBuffer_
    );

    int sender = statusBuffer_.MPI_SOURCE;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID(
            "mpi received message meta: {" << "len: " << len << ", sender: " << sender << "}")
#endif

    checkBuffer_(len);

    MPI_Recv(
            buffer_, len, MPI_CHAR, sender,
            MPIBackend::MPI_TAG_MESSAGE_MSG,
            comm.mpiComm(), &statusBuffer_
    );

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    buffer_[len] = 0;
    GLOBAL_INFO_WITH_THREAD_ID("received message msg: " << (const char *) buffer_ << ", sender: " << sender)
#endif
    auto *res = new Message(buffer_, sender, len);
    pthread_mutex_unlock(&mutex_);
    return res;
}

void MPIMessageController::checkBuffer_(size_t byteSize) {
    if (bufferSize_ < byteSize) {
        delete[]buffer_;
        buffer_ = new char[bufferSize_ = (size_t) ((double) byteSize * inflateFactor_)];
    }
}

MPIMessageController::~MPIMessageController() {
    delete[]buffer_;
    pthread_mutex_destroy(&mutex_);
}


}}}