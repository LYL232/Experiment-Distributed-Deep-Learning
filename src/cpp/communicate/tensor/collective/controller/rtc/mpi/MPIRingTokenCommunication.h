//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H

#include <memory>
#include "communicate/backend/mpi/MPIBackend.h"
#include "communicate/backend/mpi/MPICommunicator.h"
#include "communicate/tensor/collective/controller/rtc/RingTokenCommunication.h"


namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class MPIRingTokenCommunication : public RingTokenCommunication {
public:
    typedef std::vector<std::shared_ptr<TensorCollectiveCommunicateRequest>> Requests;

    explicit MPIRingTokenCommunication(std::shared_ptr<Communicator> communicator) noexcept;

    MPIRingTokenCommunication(const MPIRingTokenCommunication &) = delete;

    MPIRingTokenCommunication(MPIRingTokenCommunication &&) = delete;

    void communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const override;

    std::shared_ptr<Token> communicationReceiveTokenFrom(int sender) const override;

    StatusCode allreduceRequests(const Requests &requests) const override;

    StatusCode allgatherRequests(const Requests &requests) const override;

    StatusCode broadcastRequests(const Requests &requests) const override;

    virtual ~MPIRingTokenCommunication();

private:
    mutable MPI_Status statusBuffer_;

    mutable char *sendBuffer_, *recvBuffer_,
            *collectiveCommunicateSendBuffer_, *collectiveCommunicateRecvBuffer_;
    mutable size_t sendBufferSize_, recvBufferSize_,
            collectiveSendBufferSize_, collectiveReceiveBufferSize_,
            tokenMetaSize_;
    const MPICommunicator &mpiCommunicator_;

    static double inflateFactor_;

    void checkSendBuffer_(size_t bytesRequire) const;

    void checkRecvBuffer_(size_t bytesRequire) const;

    void checkCollectiveSendBuffer_(size_t bytesRequire) const;

    void checkCollectiveReceiveBuffer_(size_t bytesRequire) const;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
