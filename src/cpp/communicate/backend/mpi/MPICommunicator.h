//
// Created by LYL232 on 2021/3/21.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATOR_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATOR_H

#include "mpi.h"
#include <memory>
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPICommunicator : public Communicator {
public:
    MPICommunicator(std::shared_ptr<MPI_Comm> mpiComm, int rank, int size);

    MPICommunicator(const MPICommunicator &other) = delete;

    MPICommunicator(MPICommunicator &&other) = delete;

    StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            AllreduceOperation op) const override;

    StatusCode broadcast(
            void *buffer,
            size_t elements, DataType dtype,
            int rootRank) const override;

    std::shared_ptr<Communicator> split(int color, int key) const override;

    MPI_Comm &mpiComm() const;

    ID id() const noexcept override;

    /**
     * AllreduceOperation到MPIOp的映射
     * @param op
     * @return MPI_Op
     */
    static MPI_Op AllreduceOperation2MPIOp(AllreduceOperation op) noexcept;

private:
    std::shared_ptr<MPI_Comm> mpiComm_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPICOMMUNICATOR_H
