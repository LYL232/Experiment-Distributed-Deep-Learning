//
// Created by LYL232 on 2021/2/12.
//

#include "MPIRingTokenAllreduceCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

double MPIRingTokenAllreduceCommunication::inflateFactor_ = 1.5;

MPIRingTokenAllreduceCommunication::MPIRingTokenAllreduceCommunication(
        std::shared_ptr<MPIBackend> backend)
        : RingTokenAllreduceCommunication(),
          statusBuffer_(),
          sendBuffer_(nullptr), recvBuffer_(nullptr),
          allreduceSendBuffer_(nullptr), allreduceRecvBuffer_(nullptr),
          sendBufferSize_(0), recvBufferSize_(0), allreduceBufferSize_(0),
          tokenMetaSize_(sizeof(Token::Type) + sizeof(size_t)),
          backend_(backend) {}

void
MPIRingTokenAllreduceCommunication::communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const {
    using namespace std;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_RANK_THREAD_ID("before mpi sending Token");
#endif
    size_t stringSize = token->msg().length();
    int offset = 0;
    Token::Type tType = token->type();
    checkSendBuffer_(max(tokenMetaSize_, stringSize));
    // 发送Token:
    // 1. 先打包发送描述信息: token.SourceRank, token.msg_.length()
    // 2. 再发送token.msg_.length()的字符串
    MPI_Pack(&tType, sizeof(Token::Type), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
    MPI_Pack(&stringSize, sizeof(size_t), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
    MPI_Send(
            sendBuffer_, offset, MPI_PACKED, receiver,
            MPI_TAG_RTA_META, MPI_COMM_WORLD
    );
    MPI_Send(
            token->msg().c_str(), (int) stringSize, MPI_CHAR, receiver,
            MPI_TAG_RTA_MSG, MPI_COMM_WORLD
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_RANK_THREAD_ID("mpi sent Token");
#endif
}

std::shared_ptr<Token> MPIRingTokenAllreduceCommunication::communicationReceiveTokenFrom(int sender) const {
    int rank = backend_->processRank(),
            processes = backend_->processes(),
            senderRank = (rank - 1 + processes) % processes,
            offset;
    size_t stringSize;
    Token::Type tType;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread waiting rank " << senderRank << " for Token");
#endif

    checkRecvBuffer_(tokenMetaSize_);
    MPI_Recv(
            recvBuffer_, tokenMetaSize_, MPI_PACKED, senderRank,
            MPI_TAG_RTA_META,
            MPI_COMM_WORLD, &statusBuffer_
    );

    // todo: 检查status_, 出错则做相应处理
    offset = 0;
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &tType, sizeof(Token::Type), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &stringSize, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    checkRecvBuffer_(stringSize + 1);

    MPI_Recv(
            recvBuffer_, stringSize, MPI_PACKED, senderRank,
            MPI_TAG_RTA_MSG,
            MPI_COMM_WORLD, &statusBuffer_
    );
    // todo: 检查status_, 出错则做相应处理

    recvBuffer_[stringSize] = 0;
    return std::make_shared<Token>(tType, recvBuffer_);
}

StatusCode
MPIRingTokenAllreduceCommunication::allreduceRequests(
        const std::map<std::string, TensorAllreduceRequest *> &requests,
        size_t elements, size_t byteSize) const {
    int offset = 0;
    checkAllreduceBuffer_(byteSize);
    const auto firstRequest = requests.begin()->second;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL
    GLOBAL_INFO_WITH_RANK_THREAD_ID("copying memory from input tensors to tensors allreduce buffer");
#endif

    for (auto i = requests.begin(); i != requests.end(); ++i) {
        auto request = i->second;
        memcpy(
                allreduceSendBuffer_ + offset,
                request->requestingTensorData(),
                request->tensorSize()
        );
        offset += request->tensorSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL
    GLOBAL_INFO_WITH_RANK_THREAD_ID("before allreduce");
#endif
    backend_->allreduce(
            allreduceSendBuffer_, allreduceRecvBuffer_,
            elements,
            firstRequest->dtype(),
            firstRequest->op()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL
    GLOBAL_INFO_WITH_RANK_THREAD_ID("allreduced, copying memory from allreduce buffer to output tensors");
#endif
    offset = 0;
    for (auto i = requests.begin(); i != requests.end(); ++i) {
        TensorAllreduceRequest *request = i->second;
        memcpy(
                request->resultTensorData(),
                allreduceRecvBuffer_ + offset,
                request->tensorSize()
        );
        offset += request->tensorSize();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_RANK_THREAD_ID("tensor:" << request->key() << " done tensors allreduce");
#endif
        request->done(STATUS_OK);
        delete request;
    }
    // todo: check status
    return STATUS_OK;
}

void MPIRingTokenAllreduceCommunication::checkSendBuffer_(size_t bytesRequire) const {
    if (sendBufferSize_ < bytesRequire) {
        delete[]sendBuffer_;
        sendBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        sendBuffer_ = new char[sendBufferSize_];
    }
}

void MPIRingTokenAllreduceCommunication::checkRecvBuffer_(size_t bytesRequire) const {
    if (recvBufferSize_ < bytesRequire) {
        delete[]recvBuffer_;
        recvBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        recvBuffer_ = new char[recvBufferSize_];
    }
}

void MPIRingTokenAllreduceCommunication::checkAllreduceBuffer_(size_t bytesRequire) const {
    if (allreduceBufferSize_ < bytesRequire) {
        delete[]allreduceSendBuffer_;
        delete[]allreduceRecvBuffer_;
        allreduceBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        allreduceSendBuffer_ = new char[allreduceBufferSize_];
        allreduceRecvBuffer_ = new char[allreduceBufferSize_];
    }
}

MPIRingTokenAllreduceCommunication::~MPIRingTokenAllreduceCommunication() {
    delete[]sendBuffer_;
    delete[]recvBuffer_;
    delete[]allreduceSendBuffer_;
    delete[]allreduceRecvBuffer_;
}

}}}}}