//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H

#include <memory>
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/collective/controller/rtc/RingTokenCommunication.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class MPIRingTokenCommunication : public RingTokenCommunication {
public:
    typedef std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> Requests;

    MPIRingTokenCommunication(std::shared_ptr<MPIBackend> backend) noexcept;

    MPIRingTokenCommunication(const MPIRingTokenCommunication &) = delete;

    MPIRingTokenCommunication(MPIRingTokenCommunication &&) = delete;

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
