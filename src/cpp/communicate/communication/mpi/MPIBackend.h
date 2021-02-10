//
// Created by LYL232 on 2021/2/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H

#include "mpi.h"
#include "communicate/communication/CommunicationBackend.h"

#define MPI_USED_TAG_COUNTER 0

namespace lyl232 { namespace experiment { namespace ddl {

class MPIBackend : public CommunicationBackend {
    friend class Global;

public:
    MPIBackend(int *argc = nullptr, char ***argv = nullptr);

    MPIBackend(const MPIBackend &) = delete;

    MPIBackend(MPIBackend &&) = delete;

    virtual ~MPIBackend();

    virtual StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            AllreduceOperation op) const override;

    /**
     * DataType到MPI_TYPE的映射
     * @param dtype
     * @return MPI_TYPE
     */
    static int DataType2MPIType(DataType dtype) noexcept;

    /**
     * AllreduceOperation到MPIOp的映射
     * @param op
     * @return MPI_Op
     */
    static MPI_Op AllreduceOperation2MPIOp(AllreduceOperation op) noexcept;

protected:
    virtual void finalize() override final;

private:
    static int processes_, processRank_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H
