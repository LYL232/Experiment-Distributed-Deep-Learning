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
        MPI_TAG_MESSAGE_META,
        MPI_TAG_MESSAGE_MSG,
        // 以下的枚举一定要最后一个, 为了自定义tag区分于内置tag
        MPI_CUSTOM_TAG_BEGIN,
    };

    explicit MPIBackend(int *argc = nullptr, char ***argv = nullptr);

    MPIBackend(const MPIBackend &) = delete;

    MPIBackend(MPIBackend &&) = delete;

    ~MPIBackend() override;

    /**
     * 包含全部进程的通信域
     * @return
     */
    std::shared_ptr<Communicator> worldCommunicator() const noexcept override;


    /**
     * DataType到MPI_TYPE的映射
     * @param dtype
     * @return MPI_TYPE
     */
    static MPI_Datatype DataType2MPIType(DataType dtype) noexcept;

private:
    static std::mutex mutex_;
    static int refs_;
    static bool initialized_, finalized_;

    static std::shared_ptr<Communicator> worldGetter_(int *argc = nullptr, char ***argv = nullptr);

    static void initialize_(int *argc = nullptr, char ***argv = nullptr);
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIBACKEND_H
