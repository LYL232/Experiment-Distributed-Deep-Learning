//
// Created by LYL232 on 2021/2/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENALLREDUCECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENALLREDUCECONTROLLER_H

#include "communicate/communication/mpi/MPIBackend.h"
#include "communicate/tensor/allreduce/rta/RingTokenAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS 0

class MPIRingTokenAllreduceController : public RingTokenAllreduceController {
public:
    // 为了统计所有mpi tag的使用情况, 只能出此下策
#ifndef MPI_USED_TAG_COUNTER
#define MPI_USED_TAG_COUNTER 0
#endif
    enum MPICommunicateTag : int {
        MPI_TAG_RTA_META = MPI_USED_TAG_COUNTER,
        MPI_TAG_RTA_MSG = MPI_USED_TAG_COUNTER + 1,
    };
#define MPI_USED_TAG_COUNTER_TEMP MPI_USED_TAG_COUNTER + 2
#undef MPI_USED_TAG_COUNTER
#define MPI_USED_TAG_COUNTER MPI_USED_TAG_COUNTER_TEMP
#undef MPI_USED_TAG_COUNTER_TEMP

    MPIRingTokenAllreduceController(
            std::ostream &initializingLogStream,
            std::shared_ptr<MPIBackend> backend
    );

    MPIRingTokenAllreduceController(const MPIRingTokenAllreduceController &) = delete;

    MPIRingTokenAllreduceController(MPIRingTokenAllreduceController &&) = delete;

    ~MPIRingTokenAllreduceController();

private:
    std::shared_ptr<MPIBackend> communication_;

    mutable MPI_Status statusBuffer_;

    mutable char *sendBuffer_, *recvBuffer_,
            *allreduceSendBuffer_, *allreduceRecvBuffer_;
    mutable size_t sendBufferSize_, recvBufferSize_,
            allreduceBufferSize_, tokenMetaSize_;


    static double inflateFactor_;

    virtual void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const override;

    virtual std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const override;

    virtual StatusCode allreduceRequests(
            const std::map<std::string, TensorAllreduceRequest *> &requests,
            size_t elements, size_t byteSize
    ) const override;

    void checkSendBuffer_(size_t bytesRequire) const;

    void checkRecvBuffer_(size_t bytesRequire) const;

    void checkAllreduceBuffer_(size_t bytesRequire) const;
};

}}}}}

#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENALLREDUCECONTROLLER_H
