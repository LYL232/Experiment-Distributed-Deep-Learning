//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H

#include <memory>
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/tensor/end2end/controller/bcc/BlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

class MPIBlockedEnd2EndCommunication : public BlockedEnd2EndCommunication {
public:

    explicit MPIBlockedEnd2EndCommunication(std::shared_ptr<MPIBackend> backend);

    MPIBlockedEnd2EndCommunication(const MPIBlockedEnd2EndCommunication &) = delete;

    MPIBlockedEnd2EndCommunication(MPIBlockedEnd2EndCommunication &&) = delete;

    StatusCode send(const TensorSendCommunicateRequest &request) const override;

    StatusCode receive(const TensorReceiveCommunicateRequest &request) const override;

    ~MPIBlockedEnd2EndCommunication() override;

private:
    mutable std::shared_ptr<MPIBackend> backend_;

    mutable MPI_Status statusBuffer_;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H
