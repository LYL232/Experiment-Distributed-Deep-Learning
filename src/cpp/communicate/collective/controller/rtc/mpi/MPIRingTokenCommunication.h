//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H

#include <memory>
#include "communicate/communication/mpi/MPIBackend.h"
#include "communicate/collective/controller/rtc/RingTokenCommunication.h"

#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_MPI_CALLS 0

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class MPIRingTokenCommunication : public RingTokenCommunication {
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

    typedef std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> Requests;

    MPIRingTokenCommunication(std::shared_ptr<MPIBackend> backend) noexcept;

    virtual void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const override;

    virtual std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const override;

    virtual StatusCode allreduceRequests(const Requests &requests) const override;

    virtual StatusCode broadcastRequests(const Requests &requests) const override;

    virtual ~MPIRingTokenCommunication();

private:
    mutable MPI_Status statusBuffer_;

    mutable char *sendBuffer_, *recvBuffer_,
            *collectiveCommunicateSendBuffer_, *collectiveCommunicateRecvBuffer_;
    mutable size_t sendBufferSize_, recvBufferSize_,
            collectiveCommunicateBufferSize_, tokenMetaSize_;

    std::shared_ptr<MPIBackend> backend_;

    static double inflateFactor_;

    void checkSendBuffer_(size_t bytesRequire) const;

    void checkRecvBuffer_(size_t bytesRequire) const;

    void checkCollectiveCommunicateBuffer_(size_t bytesRequire) const;

    static void getRequestsInfo(const Requests &requests, size_t &elements, size_t &byteSize) noexcept;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
