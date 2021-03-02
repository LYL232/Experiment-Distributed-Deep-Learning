//
// Created by LYL232 on 2021/2/13.
//

#include "global/Global.h"
#include "communicate/tensor/collective/allreduce/TensorAllreduceRequest.h"
#include "communicate/tensor/collective/broadcast/TensorBroadcastRequest.h"
#include "communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

double MPIRingTokenCommunication::inflateFactor_ = 1.5;

MPIRingTokenCommunication::MPIRingTokenCommunication(
        std::shared_ptr<MPIBackend> backend) noexcept
        : RingTokenCommunication(),
          statusBuffer_(),
          sendBuffer_(nullptr), recvBuffer_(nullptr),
          collectiveCommunicateSendBuffer_(nullptr), collectiveCommunicateRecvBuffer_(nullptr),
          sendBufferSize_(0), recvBufferSize_(0), collectiveCommunicateBufferSize_(0),
          tokenMetaSize_(sizeof(Token::Type) + sizeof(Token::RequestType) + sizeof(size_t)),
          backend_(backend) {}

void
MPIRingTokenCommunication::communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const {
    using namespace std;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("before mpi sending Token")
#endif
    size_t stringSize = token->msg().length();
    Token::Type tType = token->type();
    Token::RequestType rType = token->requestType();

    checkSendBuffer_(max(tokenMetaSize_, stringSize));
    // 发送Token:
    // 1. 先打包发送描述信息: token.type(), token.requestType(), token.msg_.length()
    // 2. 再发送token.msg_.length()的字符串
    int offset = 0;
    MPI_Pack(&tType, sizeof(Token::Type), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
    MPI_Pack(&rType, sizeof(Token::RequestType), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
    MPI_Pack(&stringSize, sizeof(size_t), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
    MPI_Send(
            sendBuffer_, offset, MPI_PACKED, receiver,
            MPIBackend::MPI_TAG_RTA_META, MPI_COMM_WORLD
    );
    MPI_Send(
            token->msg().c_str(), (int) stringSize, MPI_CHAR, receiver,
            MPIBackend::MPI_TAG_RTA_MSG, MPI_COMM_WORLD
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent Token")
#endif
}

std::shared_ptr<Token> MPIRingTokenCommunication::communicationReceiveTokenFrom(int sender) const {
    int rank = backend_->processRank(),
            processes = backend_->processes(),
            senderRank = (rank - 1 + processes) % processes,
            offset;
    size_t stringSize;
    Token::Type tType;
    Token::RequestType rType;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("receive thread waiting rank " << senderRank << " for Token")
#endif

    checkRecvBuffer_(tokenMetaSize_);
    MPI_Recv(
            recvBuffer_, tokenMetaSize_, MPI_PACKED, senderRank,
            MPIBackend::MPI_TAG_RTA_META,
            MPI_COMM_WORLD, &statusBuffer_
    );

    // todo: 检查status_, 出错则做相应处理
    offset = 0;
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &tType, sizeof(Token::Type), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &rType, sizeof(Token::RequestType), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &stringSize, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    checkRecvBuffer_(stringSize + 1);

    MPI_Recv(
            recvBuffer_, stringSize, MPI_PACKED, senderRank,
            MPIBackend::MPI_TAG_RTA_MSG,
            MPI_COMM_WORLD, &statusBuffer_
    );
    // todo: 检查status_, 出错则做相应处理

    recvBuffer_[stringSize] = 0;
    return std::make_shared<Token>(tType, rType, recvBuffer_);
}

StatusCode
MPIRingTokenCommunication::allreduceRequests(const Requests &requests) const {
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorAllreduceRequest *>(requests[0].get());
    assert(firstRequest != nullptr);
    size_t byteSize, elements, offset = 0;

    getRequestsInfo(requests, elements, byteSize);

    checkCollectiveCommunicateBuffer_(byteSize);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    for (size_t i = 0; i < requests.size(); ++i) {
        auto &request = requests[i];
        memcpy(
                collectiveCommunicateSendBuffer_ + offset,
                request->requestingTensorData(),
                request->tensorSize()
        );
        offset += request->tensorSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before allreduce")
#endif
    backend_->allreduce(
            collectiveCommunicateSendBuffer_, collectiveCommunicateRecvBuffer_,
            elements,
            firstRequest->dtype(),
            firstRequest->op()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allreduced, copying memory from collective communicate buffer to output tensors")
#endif
    offset = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &request = requests[i];
        memcpy(
                request->resultTensorData(),
                collectiveCommunicateRecvBuffer_ + offset,
                request->tensorSize()
        );
        offset += request->tensorSize();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors allreduce")
#endif
        request->done(STATUS_OK);
    }
    // todo: check status
    return STATUS_OK;
}

StatusCode
MPIRingTokenCommunication::broadcastRequests(const Requests &requests) const {
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorBroadcastRequest *>(requests[0].get());
    assert(firstRequest != nullptr);
    size_t byteSize, elements, offset = 0;

    getRequestsInfo(requests, elements, byteSize);

    checkCollectiveCommunicateBuffer_(byteSize);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    for (size_t i = 0; i < requests.size(); ++i) {
        auto &request = requests[i];
        memcpy(
                collectiveCommunicateSendBuffer_ + offset,
                request->requestingTensorData(),
                request->tensorSize()
        );
        offset += request->tensorSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before broadcast")
#endif
    backend_->broadcast(
            collectiveCommunicateSendBuffer_,
            elements,
            firstRequest->dtype(),
            firstRequest->rootRank()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("broadcasted, copying memory from collective communicate buffer to output tensors")
#endif
    offset = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &request = requests[i];
        memcpy(
                request->resultTensorData(),
                collectiveCommunicateSendBuffer_ + offset,
                request->tensorSize()
        );
        offset += request->tensorSize();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors broadcast")
#endif
        request->done(STATUS_OK);
    }
    // todo: check status
    return STATUS_OK;
}

void MPIRingTokenCommunication::checkSendBuffer_(size_t bytesRequire) const {
    if (sendBufferSize_ < bytesRequire) {
        delete[]sendBuffer_;
        sendBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        sendBuffer_ = new char[sendBufferSize_];
    }
}

void MPIRingTokenCommunication::checkRecvBuffer_(size_t bytesRequire) const {
    if (recvBufferSize_ < bytesRequire) {
        delete[]recvBuffer_;
        recvBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        recvBuffer_ = new char[recvBufferSize_];
    }
}

void MPIRingTokenCommunication::checkCollectiveCommunicateBuffer_(size_t bytesRequire) const {
    if (collectiveCommunicateBufferSize_ < bytesRequire) {
        delete[]collectiveCommunicateSendBuffer_;
        delete[]collectiveCommunicateRecvBuffer_;
        collectiveCommunicateBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        collectiveCommunicateSendBuffer_ = new char[collectiveCommunicateBufferSize_];
        collectiveCommunicateRecvBuffer_ = new char[collectiveCommunicateBufferSize_];
    }
}

void MPIRingTokenCommunication::getRequestsInfo(
        const Requests &requests, size_t &elements, size_t &byteSize) noexcept {
    elements = byteSize = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto &request = requests[i];
        elements += request->elements();
        byteSize += request->tensorSize();
    }
}

MPIRingTokenCommunication::~MPIRingTokenCommunication() {
    delete[]sendBuffer_;
    delete[]recvBuffer_;
    delete[]collectiveCommunicateSendBuffer_;
    delete[]collectiveCommunicateRecvBuffer_;
}


}}}}