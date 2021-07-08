//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H

#include "pthread.h"
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/message/MessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPIMessageController : public MessageController {
public:
    explicit MPIMessageController();

    Message *broadcastMessage(
            const Message &message, int root, const std::shared_ptr<Communicator> &communicator) override;

    void sendMessage(const Message &message, int receiver,
                     const std::shared_ptr<Communicator> &communicator) override;

    Message *listen(const Communicator &communicator) override;

    ~MPIMessageController();

private:
    pthread_mutex_t mutex_;

    MPI_Status statusBuffer_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H
