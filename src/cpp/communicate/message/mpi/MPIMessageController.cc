//
// Created by LYL232 on 2021/2/27.
//

#include <cstring>
#include "global/Global.h"
#include "communicate/message/mpi/MPIMessageController.h"
#include "communicate/backend/mpi/MPICommunicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

MPIMessageController::MPIMessageController() :
        mutex_(PTHREAD_MUTEX_INITIALIZER), statusBuffer_() {}

Message *MPIMessageController::broadcastMessage(
        const Message &message, int root, const std::shared_ptr<Communicator> &communicator) {
    size_t len = message.length;
    pthread_mutex_lock(&mutex_);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID(
            "before broadcasting message: " << message.msg_ptr << ", root " << root)
#endif
    const auto &comm = dynamic_cast<const MPICommunicator &>(*communicator);
    // 先传输字符串的长度
    MPI_Bcast(&len, sizeof(size_t), MPI_BYTE, root, comm.mpiComm());

    auto *msgPtr = (char *) memManager_->allocateBytes(len + 1);
    if (communicator->rank() == root) {
        memcpy(msgPtr, message.msg_ptr, len);
    }

    if (len < MAX_MPI_BUFFER_SIZE) {
        MPI_Bcast(msgPtr, len, MPI_CHAR, root, comm.mpiComm());
    } else {
        size_t begin = 0, end = MAX_MPI_BUFFER_SIZE;
        while (begin < len) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController broadcasting Message with length:"
                            << len << ", at [" << begin << "][" << end << "]"
            )
#endif
            MPI_Bcast(msgPtr + begin, end - begin, MPI_CHAR, root, comm.mpiComm());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController brocasted Message with length:"
                            << len << ", at [" << begin << "][" << end << "]")
#endif
            begin = end;
            end = std::min(len, end + MAX_MPI_BUFFER_SIZE);
        }
    }
    msgPtr[len] = 0;
    auto *res = TRACK_TYPE_ALLOCATE(memManager_, new Message(root, len, msgPtr), Message);
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
    if (len < MAX_MPI_BUFFER_SIZE) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID(
                "MPIMessageController sending message length:"
                        << len << ", at all"
        )
#endif
        MPI_Send(
                message.msg_ptr, len, MPI_CHAR, receiver,
                MPIBackend::MPI_TAG_MESSAGE_MSG, comm.mpiComm()
        );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID(
                "MPIMessageController sent message length:"
                        << len << ", at all"
        )
#endif
    } else {
        size_t begin = 0, end = MAX_MPI_BUFFER_SIZE;
        while (begin < len) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController sending message length:"
                            << len << ", at [" << begin << "][" << end << "]"
            )
#endif
            MPI_Send(
                    message.msg_ptr + begin, end - begin, MPI_CHAR, receiver,
                    MPIBackend::MPI_TAG_MESSAGE_MSG, comm.mpiComm()
            );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController sent message length:"
                            << len << ", at [" << begin << "][" << end << "]"
            )
#endif
            begin = end;
            end = std::min(len, end + MAX_MPI_BUFFER_SIZE);
        }
    }

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

    auto *msgPtr = (char *) memManager_->allocateBytes(len + 1);
    if (len < MAX_MPI_BUFFER_SIZE) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID(
                "MPIMessageController receiving message length:"
                        << len << ", at all"
        )
#endif
        MPI_Recv(
                msgPtr, len, MPI_CHAR, sender,
                MPIBackend::MPI_TAG_MESSAGE_MSG,
                comm.mpiComm(), &statusBuffer_
        );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
        GLOBAL_INFO_WITH_THREAD_ID(
                "MPIMessageController received message length:"
                        << len << ", at all"
        )
#endif
    } else {
        size_t begin = 0, end = MAX_MPI_BUFFER_SIZE;
        while (begin < len) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController sending message length:"
                            << len << ", at [" << begin << "][" << end << "]"
            )
#endif
            MPI_Recv(
                    msgPtr + begin, len, MPI_CHAR, sender,
                    MPIBackend::MPI_TAG_MESSAGE_MSG,
                    comm.mpiComm(), &statusBuffer_
            );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_THREAD_ID(
                    "MPIMessageController sent message length:"
                            << len << ", at [" << begin << "][" << end << "]"
            )
#endif
            begin = end;
            end = std::min(len, end + MAX_MPI_BUFFER_SIZE);
        }
    }
    msgPtr[len] = 0;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATE_MESSAGE_LOG_MESSAGE
    GLOBAL_INFO_WITH_THREAD_ID("received message msg: " << (const char *) msgPtr << ", sender: " << sender)
#endif
    auto *res = TRACK_TYPE_ALLOCATE(memManager_, new Message(sender, len, msgPtr), Message);
    pthread_mutex_unlock(&mutex_);
    return res;
}

MPIMessageController::~MPIMessageController() {
    pthread_mutex_destroy(&mutex_);
}


}}}