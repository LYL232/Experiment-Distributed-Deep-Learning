//
// Created by LYL232 on 2021/2/27.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H

#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/message/MessageController.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPIMessageController : public MessageController {
public:
    MPIMessageController(std::shared_ptr<MPIBackend> backend);

    void sendMessage(const Message &message, int receiver) override;

    Message* listen() override;

    ~MPIMessageController();

private:
    std::shared_ptr<MPIBackend> backend_;

    char *buffer_;

    size_t bufferSize_;

    MPI_Status statusBuffer_;

    void checkBuffer_(size_t byteSize);

    static double inflateFactor_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIMESSAGECONTROLLER_H
