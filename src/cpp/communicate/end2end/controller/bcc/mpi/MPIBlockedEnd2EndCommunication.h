//
// Created by LYL232 on 2021/2/16.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H

#include <memory>
#include "pthread.h"
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/end2end/controller/bcc/BlockedEnd2EndCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace bcc {

class MPIBlockedEnd2EndCommunication : public BlockedEnd2EndCommunication {
public:

    MPIBlockedEnd2EndCommunication(std::shared_ptr<MPIBackend> backend);

    MPIBlockedEnd2EndCommunication(const MPIBlockedEnd2EndCommunication &) = delete;

    MPIBlockedEnd2EndCommunication(MPIBlockedEnd2EndCommunication &&) = delete;

    virtual StatusCode sendOrReceiveRequest(const TensorEnd2EndCommunicateRequest &request) const;

    virtual ~MPIBlockedEnd2EndCommunication();

private:
    mutable char *buffer_;
    mutable size_t bufferSize_;

    mutable std::shared_ptr<MPIBackend> backend_;

    mutable MPI_Status statusBuffer_;

    mutable pthread_mutex_t mutex_;

    static double inflateFactor_;

    void checkBuffer_(size_t bytesRequire) const;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBLOCKEDEND2ENDCOMMUNICATION_H
