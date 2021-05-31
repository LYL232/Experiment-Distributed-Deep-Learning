//
// Created by LYL232 on 2021/2/13.
//

#include "global/Global.h"
#include "communicate/tensor/collective/request/TensorAllreduceRequest.h"
#include "communicate/tensor/collective/request/TensorAllgatherRequest.h"
#include "communicate/tensor/collective/request/TensorBroadcastRequest.h"
#include "communicate/tensor/collective/controller/rtc/mpi/MPIRingTokenCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

double MPIRingTokenCommunication::inflateFactor_ = 1.5;

MPIRingTokenCommunication::MPIRingTokenCommunication(
        std::shared_ptr<Communicator> communicator) noexcept
        : RingTokenCommunication(std::move(communicator)),
          statusBuffer_(),
          sendBuffer_(nullptr), recvBuffer_(nullptr),
          collectiveCommunicateSendBuffer_(nullptr), collectiveCommunicateRecvBuffer_(nullptr),
          sendBufferSize_(0), recvBufferSize_(0),
          collectiveSendBufferSize_(0), collectiveReceiveBufferSize_(0),
          tokenMetaSize_(sizeof(Token::Type) + sizeof(Token::RequestType) + sizeof(size_t)),
          mpiCommunicator_(dynamic_cast<const MPICommunicator &>(*communicator_)) {}

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
    MPI_Pack(&tType, sizeof(Token::Type), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset,
             mpiCommunicator_.mpiComm());
    MPI_Pack(&rType, sizeof(Token::RequestType), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset,
             mpiCommunicator_.mpiComm());
    MPI_Pack(&stringSize, sizeof(size_t), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset,
             mpiCommunicator_.mpiComm());
    MPI_Send(
            sendBuffer_, offset, MPI_PACKED, receiver,
            MPIBackend::MPI_TAG_RTA_META, mpiCommunicator_.mpiComm()
    );
    MPI_Send(
            token->msg().c_str(), (int) stringSize, MPI_CHAR, receiver,
            MPIBackend::MPI_TAG_RTA_MSG, mpiCommunicator_.mpiComm()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_THREAD_ID("mpi sent Token")
#endif
}

std::shared_ptr<Token> MPIRingTokenCommunication::communicationReceiveTokenFrom(int sender) const {
    int rank = communicator_->rank(),
            processes = communicator_->size(),
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
            mpiCommunicator_.mpiComm(), &statusBuffer_
    );

    // todo: 检查status_, 出错则做相应处理
    offset = 0;
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &tType, sizeof(Token::Type), MPI_BYTE,
               mpiCommunicator_.mpiComm());
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &rType, sizeof(Token::RequestType), MPI_BYTE,
               mpiCommunicator_.mpiComm());
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &stringSize, sizeof(size_t), MPI_BYTE,
               mpiCommunicator_.mpiComm());

    checkRecvBuffer_(stringSize + 1);

    MPI_Recv(
            recvBuffer_, stringSize, MPI_PACKED, senderRank,
            MPIBackend::MPI_TAG_RTA_MSG,
            mpiCommunicator_.mpiComm(), &statusBuffer_
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
    size_t byteSize = 0, elements = 0, offset = 0;

    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        elements += tensor->elements();
        byteSize += tensor->byteSize();
    }

    checkCollectiveSendBuffer_(byteSize);
    checkCollectiveReceiveBuffer_(byteSize);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        memcpy(
                collectiveCommunicateSendBuffer_ + offset,
                tensor->data(), tensor->byteSize()
        );
        offset += tensor->byteSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before allreduce")
#endif
    communicator_->allreduce(
            collectiveCommunicateSendBuffer_, collectiveCommunicateRecvBuffer_,
            elements,
            firstRequest->requestingTensor()->dtype(),
            firstRequest->op()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allreduced, copying memory from collective communicate buffer to output tensors")
#endif
    offset = 0;
    for (const auto &request : requests) {
        const auto &tensor = request->resultTensor();
        memcpy(
                tensor->data(),
                collectiveCommunicateRecvBuffer_ + offset,
                tensor->byteSize()
        );
        offset += tensor->byteSize();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors allreduce")
#endif
        request->done(STATUS_OK);
    }
    // todo: check status
    return STATUS_OK;
}

StatusCode
MPIRingTokenCommunication::allgatherRequests(const Requests &requests) const {
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorAllgatherRequest *>(requests[0].get());
    assert(firstRequest != nullptr);

    const size_t dtypeSize = dataTypeSize(firstRequest->requestingTensor()->dtype());

    // 每个请求的张量除了第一维的形状
    std::vector<CommonTensorShape> requestSingleSliceShapes;
    requestSingleSliceShapes.resize(requests.size());

    for (size_t j = 0; j < requests.size(); ++j) {
        const auto &shape = requests[j]->requestingTensor()->shape();
        auto &singleSliceShape = requestSingleSliceShapes[j];
        for (size_t i = 1; i < shape.dims(); ++i) {
            singleSliceShape.addDim(shape.dimSize(i));
        }
    }

    // 每个进程收集其他进程的张量的第一个维度信息
    // 每个进程都发送requests.size()个size_t, 表示每个要allgather的张量的第一个维度,
    // 因为IndexedSlices的张量除了第一个维度, 其他维度都一致

    // todo: 或许直接可以在RingToken的传递中直接传递这个信息, 不用在这里再allgather一次

    checkCollectiveSendBuffer_(sizeof(size_t) * requests.size());
    checkCollectiveReceiveBuffer_(sizeof(size_t) * requests.size() * communicator_->size());

    auto *sizeBuffer = (size_t *) collectiveCommunicateSendBuffer_;

    for (size_t i = 0; i < requests.size(); ++i) {
        sizeBuffer[i] = requests[i]->requestingTensor()->shape().dimSize(0);
    }

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    {
        std::string firstDimString;
        for (size_t i = 0; i < requests.size(); ++i) {
            firstDimString += std::to_string(sizeBuffer[i]);
            firstDimString += ", ";
        }
        GLOBAL_INFO_WITH_THREAD_ID("before allgather first dimension of every process: ["
                                           << firstDimString << "]")
    }
#endif

    communicator_->allgather(
            collectiveCommunicateSendBuffer_, requests.size(),
            collectiveCommunicateRecvBuffer_, requests.size(),
            DataType::DT_UINT64
    );

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allgathered")
#endif

    // 每个rank的每个请求的张量的第一个维度 [rank][requestIndex]
    std::vector<std::vector<size_t>> firstDims;
    firstDims.resize(communicator_->size());

    auto *firstDimsBuffer = (size_t *) collectiveCommunicateRecvBuffer_;

    // recvDataOffset[i][j] = 接收数据buffer中第i个rank的第j个请求的tensor数据的字节下标
    std::vector<std::vector<size_t>> recvDataOffset;
    size_t offsetCounter = 0;

    recvDataOffset.resize(communicator_->size());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    std::string firstDimLogString;
    {
        std::string temp = "buffer: [";
        for (size_t i = 0; i < communicator_->size(); ++i) {
            for (size_t j = 0; j < requests.size(); ++j) {
                temp += std::to_string(firstDimsBuffer[i * requests.size() + j]) + ", ";
            }
        }
        GLOBAL_INFO_WITH_THREAD_ID(temp << "]\n")

    }
#endif

    size_t totRecvElements = 0;
    std::vector<size_t> displs, recvcounts;
    displs.resize(communicator_->size());
    recvcounts.resize(communicator_->size());

    for (int i = 0; i < communicator_->size(); ++i) {
        auto &offset = recvDataOffset[i];
        offset.resize(requests.size(), 0);

        auto &dims = firstDims[i];
        dims.resize(requests.size(), 0);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
        std::string dimString = "[";
#endif
        size_t elements = 0;
        for (size_t j = 0; j < requests.size(); ++j) {
            dims[j] = firstDimsBuffer[i * requests.size() + j];
            size_t requestElements = dims[j] * requestSingleSliceShapes[j].numElements();
            offset[j] = offsetCounter;
            elements += requestElements;
            offsetCounter += dtypeSize * requestElements;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
            dimString += std::to_string(firstDims[i][j]) + ", ";
#endif
        }

        displs[i] = totRecvElements;
        totRecvElements += elements;
        recvcounts[i] = elements;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
        firstDimLogString += dimString + "]\n";
#endif
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allgathered first dimension of every process: \n" << firstDimLogString)
#endif

    size_t sendByteSize = 0, sendElements = 0;

    // 获取每个需要传输的数据量
    for (size_t j = 0; j < requests.size(); ++j) {
        const auto &request = requests[j];

        sendElements += request->requestingTensor()->elements();
        sendByteSize += request->requestingTensor()->byteSize();

        // 除了第一维度的张量的形状
        auto &singleSliceShape = requestSingleSliceShapes[j];

        size_t totalFirstDimSize = 0;
        for (int i = 0; i < communicator_->size(); ++i) {
            totalFirstDimSize += firstDims[i][j];
        }

        CommonTensorShape outputShape;
        outputShape.addDim(totalFirstDimSize);
        outputShape.appendShape(singleSliceShape);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
        GLOBAL_INFO_WITH_THREAD_ID("allocating memory of output tensor of request: " << request->key())
#endif

        auto statusCode = request->context()->allocateOutput(outputShape, request->resultTensor());
        if (statusCode != STATUS_OK) {
            // todo: status check
            return statusCode;
        }
    }

    checkCollectiveSendBuffer_(sendByteSize);
    checkCollectiveReceiveBuffer_(totRecvElements * dtypeSize);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    offsetCounter = 0;
    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        memcpy(
                collectiveCommunicateSendBuffer_ + offsetCounter,
                tensor->data(), tensor->byteSize()
        );
        offsetCounter += tensor->byteSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before allgathering tensor data")
#endif
    communicator_->allgather(
            collectiveCommunicateSendBuffer_, sendElements,
            collectiveCommunicateRecvBuffer_,
            recvcounts, displs,
            firstRequest->requestingTensor()->dtype()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allgathered, copying memory from collective communicate buffer to output tensors")
#endif

    for (size_t j = 0; j < requests.size(); ++j) {
        const auto &request = requests[j];
        auto *dataBytesPtr = (char *) request->resultTensor()->data();
        offsetCounter = 0;

        for (int i = 0; i < communicator_->size(); ++i) {
            size_t bytes = firstDims[i][j] * requestSingleSliceShapes[j].numElements() * dtypeSize;
            memcpy(
                    dataBytesPtr + offsetCounter,
                    collectiveCommunicateRecvBuffer_ + recvDataOffset[i][j],
                    bytes
            );
            offsetCounter += bytes;
        }
        request->done(STATUS_OK);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors allgather")
#endif
    }

    // todo: check status
    return STATUS_OK;
}


StatusCode
MPIRingTokenCommunication::broadcastRequests(const Requests &requests) const {
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorBroadcastRequest *>(requests[0].get());
    assert(firstRequest != nullptr);
    size_t byteSize = 0, elements = 0, offset = 0;

    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        elements += tensor->elements();
        byteSize += tensor->byteSize();
    }

    checkCollectiveSendBuffer_(byteSize);
    checkCollectiveReceiveBuffer_(byteSize);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        memcpy(
                collectiveCommunicateSendBuffer_ + offset,
                tensor->data(), tensor->byteSize()
        );
        offset += tensor->byteSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before broadcast")
#endif
    communicator_->broadcast(
            collectiveCommunicateSendBuffer_,
            elements,
            firstRequest->requestingTensor()->dtype(),
            firstRequest->rootRank()
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("broadcasted, copying memory from collective communicate buffer to output tensors")
#endif
    offset = 0;
    for (const auto &request : requests) {
        const auto &tensor = request->resultTensor();
        memcpy(
                tensor->data(),
                collectiveCommunicateSendBuffer_ + offset,
                tensor->byteSize()
        );
        offset += tensor->byteSize();

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

void MPIRingTokenCommunication::checkCollectiveSendBuffer_(size_t bytesRequire) const {
    if (collectiveSendBufferSize_ < bytesRequire) {
        delete[]collectiveCommunicateSendBuffer_;
        collectiveSendBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        collectiveCommunicateSendBuffer_ = new char[collectiveSendBufferSize_];
    }
}

void MPIRingTokenCommunication::checkCollectiveReceiveBuffer_(size_t bytesRequire) const {
    if (collectiveReceiveBufferSize_ < bytesRequire) {
        delete[]collectiveCommunicateRecvBuffer_;
        collectiveReceiveBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        collectiveCommunicateRecvBuffer_ = new char[collectiveReceiveBufferSize_];
    }
}

MPIRingTokenCommunication::~MPIRingTokenCommunication() {
    delete[]sendBuffer_;
    delete[]recvBuffer_;
    delete[]collectiveCommunicateSendBuffer_;
    delete[]collectiveCommunicateRecvBuffer_;
}


}}}}