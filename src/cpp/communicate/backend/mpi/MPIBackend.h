//
// Created by LYL232 on 2021/2/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H

#include <mutex>
#include "mpi.h"
#include "communicate/backend/CommunicationBackend.h"

namespace lyl232 { namespace experiment { namespace ddl {

class MPIBackend : public CommunicationBackend {
    friend class Global;

public:
    enum MPICommunicateTag : int {
        MPI_TAG_RTA_META,
        MPI_TAG_RTA_MSG,
        MPI_TAG_BCC_COMMUNICATE_AS_SENDER,
        MPI_TAG_BCC_COMMUNICATE_AS_RECEIVER,
    };

    MPIBackend(int *argc = nullptr, char ***argv = nullptr);

    MPIBackend(const MPIBackend &) = delete;

    MPIBackend(MPIBackend &&) = delete;

    virtual ~MPIBackend();

    virtual StatusCode allreduce(
            void *sendBuffer, void *recvBuffer,
            size_t elements, DataType dtype,
            AllreduceOperation op) const override;

    virtual StatusCode broadcast(
            void *buffer,
            size_t elements, DataType dtype,
            int rootRank) const override;

    virtual int processes() const override;

    virtual int processRank() const override;

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

private:
    static std::mutex mutex_;
    static int processes_, processRank_, refs_;
    static bool initialized_, finalized_;

    static int processesImpl_(int *argc = nullptr, char ***argv = nullptr);

    static int processRankImpl_(int *argc = nullptr, char ***argv = nullptr);

    static void initialize_(int *argc = nullptr, char ***argv = nullptr);
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H
