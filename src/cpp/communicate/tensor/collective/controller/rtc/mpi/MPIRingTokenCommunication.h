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
    struct CommunicatePlan {
        size_t requestBegin, elementBegin, requestEnd, elementEnd;

        CommunicatePlan(size_t requestBegin_, size_t elementBegin_, size_t requestEnd_, size_t elementEnd_)
                : requestBegin(requestBegin_), elementBegin(elementBegin_), requestEnd(requestEnd_),
                  elementEnd(elementEnd_) {}
    };

    enum CheckingBuffer {
        CHECKING_SEND_BUFFER,
        CHECKING_RECV_BUFFER,
        CHECKING_COLLECTIVE_SEND_BUFFER,
        CHECKING_COLLECTIVE_RECV_BUFFER,
        CHECKING_ALLGATHER_SEND_BUFFER,
        CHECKING_ALLGATHER_RECV_BUFFER,
    };

    mutable MPI_Status statusBuffer_;

    mutable char *sendBuffer_, *recvBuffer_,
            *collectiveCommunicateSendBuffer_, *collectiveCommunicateRecvBuffer_,
            *allgatherSendBuffer_, *allgatherRecvBuffer_;
    mutable size_t sendBufferSize_, recvBufferSize_,
            collectiveSendBufferSize_, collectiveReceiveBufferSize_,
            allgatherSendBufferSize_, allgatherRecvBufferSize_,
            tokenMetaSize_;
    const MPICommunicator &mpiCommunicator_;

    static double inflateFactor_;

    void checkCollectiveBuffer_(char **buffer, size_t bytesRequire) const;

    void checkBuffer_(CheckingBuffer checkingBuffer, size_t bytesRequire) const;

    /**
     * 根据通信请求计算出接下来的通信计划
     * @param requests 通信请求
     * @param resultPlan 返回的通信计划
     * 其每个元素的意义为<该计划包含的起始请求下标，起始请求的起始元素下标，该计划包含的结束请求下标，结束请求的结束元素（不包含）下标>
     */
    static void makeCollectiveCommunicatePlan(
            const Requests &requests, std::vector<CommunicatePlan> &resultPlan
    );

    StatusCode executeCommunicatePlan_(
            const Requests &requests,
            const std::vector<CommunicatePlan> &communicatePlan,
            char **bufferForSending, char **bufferForReceiving,
            std::function<StatusCode(size_t)> doCommunicateFunction
    ) const;

    /**
     * 将请求按照数据类型分类，存储在resultMap中
     * @param requests 请求列表
     * @param resultMap 分类后的请求列表
     */
    static void classifyRequestsByDataType(
            const Requests &requests,
            std::map<DataType, Requests> &resultMap
    );
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_MPIRINGTOKENCOMMUNICATION_H
