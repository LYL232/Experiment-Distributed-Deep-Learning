//
// Created by LYL232 on 2021/2/27.
//

#include <cstring>
#include "global/Global.h"
#include "communicate/message/mpi/MPIMessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

double MPIMessageController::inflateFactor_ = 1.5;

MPIMessageController::MPIMessageController(std::shared_ptr<MPIBackend> backend) :
        backend_(backend), buffer_(nullptr), bufferSize_(0), statusBuffer_() {}

void MPIMessageController::sendMessage(const Message &message, int receiver) {
    size_t len = strlen(message.msg_ptr);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID(
            "before sending message: " << message.msg_ptr << ", to " << receiver)
#endif
    checkBuffer_(len);
    // 先传输字符串的长度
    MPI_Send(
            &len, sizeof(size_t), MPI_BYTE, receiver,
            MPIBackend::MPI_TAG_MESSAGE_META, MPI_COMM_WORLD
    );
    MPI_Send(
            message.msg_ptr, len, MPI_CHAR, receiver,
            MPIBackend::MPI_TAG_MESSAGE_MSG, MPI_COMM_WORLD
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent Message")
#endif
}

Message *MPIMessageController::listen() {
    size_t len;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID("listen message")
#endif

    MPI_Recv(
            &len, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE,
            MPIBackend::MPI_TAG_MESSAGE_META,
            MPI_COMM_WORLD, &statusBuffer_
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
            MPI_COMM_WORLD, &statusBuffer_
    );

    char *msg = new char[len + 1];
    memcpy(msg, buffer_, len);
    msg[len] = 0;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID("received message msg: " << msg << ", sender: " << sender)
#endif

    return new Message(msg, sender);
}

void MPIMessageController::checkBuffer_(size_t byteSize) {
    if (bufferSize_ < byteSize) {
        delete[]buffer_;
        buffer_ = new char[bufferSize_ = (size_t) ((double) byteSize * inflateFactor_)];
    }
}

MPIMessageController::~MPIMessageController() {
    delete[]buffer_;
}


}}}