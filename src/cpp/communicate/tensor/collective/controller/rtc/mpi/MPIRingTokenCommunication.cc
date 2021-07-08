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

    checkBuffer_(CHECKING_SEND_BUFFER, max(tokenMetaSize_, stringSize));
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

    checkBuffer_(CHECKING_RECV_BUFFER, tokenMetaSize_);
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

    checkBuffer_(CHECKING_RECV_BUFFER, stringSize + 1);

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
    using namespace std;
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorAllreduceRequest *>(requests[0].get());
    assert(firstRequest != nullptr);
    map<DataType, Requests> dtypeRequests;
    classifyRequestsByDataType(requests, dtypeRequests);

    for (auto iter = dtypeRequests.begin(); iter != dtypeRequests.end(); ++iter) {
        auto dtype = iter->first;
        const auto &dtypeReq = iter->second;
        vector<CommunicatePlan> communicatePlans;

        makeCollectiveCommunicatePlan(dtypeReq, communicatePlans);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
        {
            stringstream ss;
            size_t dtypeSize = dataTypeSize(dtype);
            ss << "handling requests of DataType: " << dtype << ": [\n";
            for (const auto &each : dtypeReq) {
                ss << "{key: " << each->key() << ", dtype size: " << dtypeSize <<
                   ", elements: " << each->requestingTensor()->elements() << "},\n";
            }
            ss << "]\nplan: [";
            for (const auto &plan: communicatePlans) {
                ss << "(" << plan.requestBegin << ", " << plan.elementBegin
                   << ", " << plan.requestEnd << ", " << plan.elementEnd << "),\n";
            }
            ss << "]";
            GLOBAL_INFO_WITH_THREAD_ID(ss.str())
        }
#endif

        // 执行计划
        executeCommunicatePlan_(
                dtypeReq, communicatePlans,
                &collectiveCommunicateSendBuffer_,
                &collectiveCommunicateRecvBuffer_,
                [this, dtype, firstRequest](size_t elements) {
                    communicator_->allreduce(
                            collectiveCommunicateSendBuffer_,
                            collectiveCommunicateRecvBuffer_,
                            elements,
                            dtype,
                            firstRequest->op()
                    );
                    return STATUS_OK;
                });
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

    checkBuffer_(CHECKING_ALLGATHER_SEND_BUFFER, sizeof(size_t) * requests.size());
    checkBuffer_(CHECKING_ALLGATHER_RECV_BUFFER, sizeof(size_t) * requests.size() * communicator_->size());

    auto *sizeBuffer = (size_t *) allgatherSendBuffer_;

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
            allgatherSendBuffer_, requests.size(),
            allgatherRecvBuffer_, requests.size(),
            DataType::DT_UINT64
    );

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("allgathered")
#endif

    // 每个rank的每个请求的张量的第一个维度 [rank][requestIndex]
    std::vector<std::vector<size_t>> firstDims;
    firstDims.resize(communicator_->size());

    auto *firstDimsBuffer = (size_t *) allgatherRecvBuffer_;

    // recvDataOffset[i][j] = 接收数据buffer中第i个rank的第j个请求的tensor数据的字节下标
    std::vector<std::vector<size_t>> recvDataOffset;
    size_t offsetCounter = 0;

    recvDataOffset.resize(communicator_->size());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    std::string firstDimLogString;
    {
        std::string temp = "buffer: [";
        for (size_t i = 0; i < (size_t) communicator_->size(); ++i) {
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

    checkBuffer_(CHECKING_ALLGATHER_SEND_BUFFER, sendByteSize);
    checkBuffer_(CHECKING_ALLGATHER_RECV_BUFFER, totRecvElements * dtypeSize);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("copying memory from input tensors to collective communicate buffer")
#endif

    offsetCounter = 0;
    for (const auto &request : requests) {
        const auto &tensor = request->requestingTensor();
        memcpy(
                allgatherSendBuffer_ + offsetCounter,
                tensor->data(), tensor->byteSize()
        );
        offsetCounter += tensor->byteSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("before allgathering tensor data")
#endif
    communicator_->allgather(
            allgatherSendBuffer_, sendElements,
            allgatherRecvBuffer_,
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
                    allgatherRecvBuffer_ + recvDataOffset[i][j],
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
    using namespace std;
    assert(!requests.empty());
    auto *firstRequest = dynamic_cast<TensorBroadcastRequest *>(requests[0].get());
    assert(firstRequest != nullptr);

    map<DataType, Requests> dtypeRequests;
    classifyRequestsByDataType(requests, dtypeRequests);

    for (auto iter = dtypeRequests.begin(); iter != dtypeRequests.end(); ++iter) {
        auto dtype = iter->first;
        const auto &dtypeReq = iter->second;
        vector<CommunicatePlan> communicatePlans;

        makeCollectiveCommunicatePlan(dtypeReq, communicatePlans);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
        {
            stringstream ss;
            size_t dtypeSize = dataTypeSize(dtype);
            ss << "handling requests of DataType: " << dtype << ": [\n";
            for (const auto &each : dtypeReq) {
                ss << "{key: " << each->key() << ", dtype size: " << dtypeSize <<
                   ", elements: " << each->requestingTensor()->elements() << "},\n";
            }
            ss << "]\nplan: [";
            for (const auto &plan: communicatePlans) {
                ss << "(" << plan.requestBegin << ", " << plan.elementBegin
                   << ", " << plan.requestEnd << ", " << plan.elementEnd << "),\n";
            }
            ss << "]";
            GLOBAL_INFO_WITH_THREAD_ID(ss.str())
        }
#endif

        // 执行计划
        executeCommunicatePlan_(
                dtypeReq, communicatePlans,
                &collectiveCommunicateSendBuffer_,
                &collectiveCommunicateSendBuffer_,
                [this, dtype, firstRequest](size_t elements) {
                    communicator_->broadcast(
                            collectiveCommunicateSendBuffer_,
                            elements, dtype,
                            firstRequest->rootRank()
                    );
                    return STATUS_OK;
                });
    }

    // todo: check status
    return STATUS_OK;
}

void MPIRingTokenCommunication::checkCollectiveBuffer_(char **buffer, size_t bytesRequire) const {
    assert(buffer == &collectiveCommunicateSendBuffer_ || buffer == &collectiveCommunicateRecvBuffer_);
    if (buffer == &collectiveCommunicateSendBuffer_) {
        checkBuffer_(CHECKING_COLLECTIVE_SEND_BUFFER, bytesRequire);
    } else {
        checkBuffer_(CHECKING_COLLECTIVE_RECV_BUFFER, bytesRequire);
    }
}

void MPIRingTokenCommunication::checkBuffer_(CheckingBuffer checkingBuffer, size_t bytesRequire) const {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
    GLOBAL_INFO_WITH_THREAD_ID("MPIRingTokenCommunication::checkBuffer_("
                                       << checkingBuffer << ", " << bytesRequire << ")")
#endif
    char **checkingBufferPtr;
    size_t *checkingBufferSize;
    bool limit;
    switch (checkingBuffer) {
        case CHECKING_SEND_BUFFER:
            checkingBufferPtr = &sendBuffer_;
            checkingBufferSize = &sendBufferSize_;
            limit = true;
            break;
        case CHECKING_RECV_BUFFER:
            checkingBufferPtr = &recvBuffer_;
            checkingBufferSize = &recvBufferSize_;
            limit = true;
            break;
        case CHECKING_COLLECTIVE_SEND_BUFFER:
            checkingBufferPtr = &collectiveCommunicateSendBuffer_;
            checkingBufferSize = &collectiveSendBufferSize_;
            limit = true;
            break;
        case CHECKING_COLLECTIVE_RECV_BUFFER:
            checkingBufferPtr = &collectiveCommunicateRecvBuffer_;
            checkingBufferSize = &collectiveReceiveBufferSize_;
            limit = true;
            break;
        case CHECKING_ALLGATHER_SEND_BUFFER:
            checkingBufferPtr = &allgatherSendBuffer_;
            checkingBufferSize = &allgatherSendBufferSize_;
            // allgather的缓冲数组不能被限制，因为无法分批发送
            limit = false;
            break;
        case CHECKING_ALLGATHER_RECV_BUFFER:
            checkingBufferPtr = &allgatherRecvBuffer_;
            checkingBufferSize = &allgatherRecvBufferSize_;
            // allgather的缓冲数组不能被限制，因为无法分批发送
            limit = false;
            break;
        default:
            return;
    }
    if (*checkingBufferSize < bytesRequire &&
        (
                !limit ||
                *checkingBufferSize < MAX_MPI_BUFFER_SIZE
        )) {
        memManager_->deallocateBytes(*checkingBufferPtr);
        *checkingBufferSize = (size_t) ((double) bytesRequire * inflateFactor_);
        if (limit) {
            *checkingBufferSize = std::min(*checkingBufferSize, MAX_MPI_BUFFER_SIZE);
        }
        *checkingBufferPtr = (char *) memManager_->allocateBytes(*checkingBufferSize);
    }
}

MPIRingTokenCommunication::~MPIRingTokenCommunication() {
    memManager_->deallocateBytes(sendBuffer_);
    memManager_->deallocateBytes(recvBuffer_);
    memManager_->deallocateBytes(collectiveCommunicateSendBuffer_);
    memManager_->deallocateBytes(collectiveCommunicateRecvBuffer_);
}

void MPIRingTokenCommunication::makeCollectiveCommunicatePlan(
        const Requests &requests, std::vector<CommunicatePlan> &resultPlan) {
    using namespace std;
    size_t byteSize = 0, allRequestsByteSize = 0,
            requestPlannedBegin = 0, requestPlannedBeginElement = 0,
            requestPlannedEnd = 0, requestPlannedEndByteEnd = 0, requestPlannedEndElement = 0;
    vector<size_t> requestSizeBegin;

    for (const auto &request: requests) {
        requestSizeBegin.emplace_back(allRequestsByteSize);
        allRequestsByteSize += request->requestingTensor()->byteSize();
    }

    while (byteSize < allRequestsByteSize) {
        byteSize = byteSize + MAX_MPI_BUFFER_SIZE;
        if (byteSize >= allRequestsByteSize) {
            requestPlannedEnd = requests.size() - 1;
            requestPlannedEndElement = requests[requestPlannedEnd]->requestingTensor()->elements();
        } else {
            // 找到这次进行组通讯的最后一个请求并确定
            for (size_t i = requestPlannedBegin; i < requests.size(); ++i) {
                if (i == requests.size() - 1 ||
                    (requestSizeBegin[i] < byteSize && byteSize <= requestSizeBegin[i + 1])) {
                    requestPlannedEnd = i;
                    requestPlannedEndByteEnd = byteSize - requestSizeBegin[i];
                    break;
                }
            }
            requestPlannedEndElement = requestPlannedEndByteEnd
                                       / dataTypeSize(
                    requests[requestPlannedEnd]->requestingTensor()->dtype()
            );
        }
        resultPlan.emplace_back(
                requestPlannedBegin, requestPlannedBeginElement,
                requestPlannedEnd, requestPlannedEndElement
        );
        // 如果正好requestPlannedEndByteEnd是requestPlannedEnd请求的最后一个元素，那么
        // requestPlannedBegin+1 requestPlannedBeginElement=0
        if (requestPlannedEndElement == requests[requestPlannedEnd]->requestingTensor()->elements()) {
            requestPlannedBegin += 1;
            requestPlannedBeginElement = 0;
        } else {
            requestPlannedBegin = requestPlannedEnd;
            requestPlannedBeginElement = requestPlannedEndElement;
        }
        // 修正已经计划好的字节数
        byteSize = requestSizeBegin[requestPlannedEnd]
                   + dataTypeSize(requests[requestPlannedEnd]->requestingTensor()->dtype())
                     * requestPlannedEndElement;
    }
}

StatusCode MPIRingTokenCommunication::executeCommunicatePlan_(
        const Requests &requests, const std::vector<CommunicatePlan> &communicatePlan,
        char **bufferForSending, char **bufferForReceiving,
        std::function<StatusCode(size_t)> doCommunicateFunction) const {
    using namespace std;

    for (const auto &plan: communicatePlan) {
        if (plan.requestBegin == plan.requestEnd) {
            const auto &request = requests[plan.requestBegin];
            const auto &requestingTensor = request->requestingTensor(),
                    &resultTensor = request->resultTensor();
            size_t dtypeSize = dataTypeSize(requestingTensor->dtype()),
                    byteBegin = plan.elementBegin * dtypeSize,
                    byteEnd = plan.elementEnd * dtypeSize,
                    byteSize = byteEnd - byteBegin;

            assert(byteSize <= MAX_MPI_BUFFER_SIZE);

            checkCollectiveBuffer_(bufferForSending, byteSize);
            checkCollectiveBuffer_(bufferForReceiving, byteSize);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
            GLOBAL_INFO_WITH_THREAD_ID(
                    "memcpy from request[" << plan.requestBegin << ", " << plan.elementBegin << ":"
                                           << plan.elementEnd << "] to send buffer["
                                           << 0 << ":" << byteSize << "]")
#endif
            memcpy(
                    *bufferForSending,
                    (char *) requestingTensor->data() + byteBegin,
                    byteSize
            );
            // todo: status check
            doCommunicateFunction(plan.elementEnd - plan.elementBegin);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
            GLOBAL_INFO_WITH_THREAD_ID(
                    "communicted memcpy from receive buffer[" << 0 << ":" << byteSize << "] to request["
                                                              << plan.requestBegin << ", "
                                                              << plan.elementBegin << ":" << plan.elementEnd << "]")
#endif
            memcpy(
                    (char *) resultTensor->data() + byteBegin,
                    *bufferForReceiving,
                    byteSize
            );
            if (plan.elementEnd == requestingTensor->elements()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
                GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors communicate")
#endif
                request->done(STATUS_OK);
            }
        } else {
            size_t byteSize = 0;
            // 先获取需要通信的字节数
            for (size_t i = plan.requestBegin; i <= plan.requestEnd; ++i) {
                const auto &tensor = requests[i]->requestingTensor();
                if (i == plan.requestBegin) {
                    byteSize += (tensor->elements() - plan.elementBegin) * dataTypeSize(tensor->dtype());
                } else if (i == plan.requestEnd) {
                    byteSize += plan.elementEnd * dataTypeSize(tensor->dtype());
                } else {
                    byteSize += tensor->byteSize();
                }
            }
            assert(byteSize <= MAX_MPI_BUFFER_SIZE);
            checkCollectiveBuffer_(bufferForSending, byteSize);
            checkCollectiveBuffer_(bufferForReceiving, byteSize);

            // 复制内存到发送缓冲中
            size_t offset = 0, elements = 0;
            for (size_t i = plan.requestBegin; i <= plan.requestEnd; ++i) {
                const auto &tensor = requests[i]->requestingTensor();
                if (i == plan.requestBegin) {
                    size_t dtypeSize = dataTypeSize(tensor->dtype());
                    byteSize = (tensor->elements() - plan.elementBegin) * dtypeSize;
                    elements += tensor->elements() - plan.elementBegin;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from request[" << i << "][" << plan.elementBegin << ":] to send buffer["
                                                   << offset << ":" << offset + byteSize << "]")
#endif
                    memcpy(
                            *bufferForSending + offset,
                            (char *) tensor->data() + plan.elementBegin * dtypeSize,
                            byteSize
                    );
                } else if (i == plan.requestEnd) {
                    size_t dtypeSize = dataTypeSize(tensor->dtype());
                    byteSize = plan.elementEnd * dtypeSize;
                    elements += plan.elementEnd;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from request[" << i << "][:" << plan.elementEnd << "] to send buffer["
                                                   << offset << ":" << offset + byteSize << "]")
#endif
                    memcpy(
                            *bufferForSending + offset,
                            tensor->data(), byteSize
                    );
                } else {
                    byteSize = tensor->byteSize();
                    elements += tensor->elements();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from request[" << i << ", 0:" << tensor->elements() << "] to send buffer["
                                                   << offset << ":" << offset + byteSize << "]")
#endif
                    memcpy(
                            *bufferForSending + offset,
                            tensor->data(), byteSize
                    );
                }
                offset += byteSize;
            }

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
            GLOBAL_INFO_WITH_THREAD_ID("before communicating")
#endif
            doCommunicateFunction(elements);
            // 复制接收缓冲中的内容到张量中
            offset = 0;
            elements = 0;
            for (size_t i = plan.requestBegin; i <= plan.requestEnd; ++i) {
                const auto &request = requests[i];
                const auto &tensor = request->resultTensor();
                if (i == plan.requestBegin) {
                    size_t dtypeSize = dataTypeSize(tensor->dtype());
                    byteSize = (tensor->elements() - plan.elementBegin) * dtypeSize;
                    elements += tensor->elements() - plan.elementBegin;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from receive buffer[" << offset << ":" << offset + byteSize << "] to request["
                                                          << i << ", " << plan.elementBegin << ":]")
#endif
                    memcpy(
                            (char *) tensor->data() + plan.elementBegin * dtypeSize,
                            *bufferForReceiving + offset,
                            byteSize
                    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
                    GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors communicate")
#endif
                    request->done(STATUS_OK);
                } else if (i == plan.requestEnd) {
                    size_t dtypeSize = dataTypeSize(tensor->dtype());
                    byteSize = plan.elementEnd * dtypeSize;
                    elements += plan.elementEnd;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from receive buffer[" << offset << ":" << offset + byteSize << "] to request["
                                                          << i << ", :" << plan.elementEnd << "]")
#endif
                    memcpy(
                            tensor->data(), *bufferForReceiving + offset,
                            byteSize
                    );
                    if (plan.elementEnd == tensor->elements()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
                        GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors communicate")
#endif
                        request->done(STATUS_OK);
                    }
                } else {
                    byteSize = tensor->byteSize();
                    elements += tensor->elements();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_DETAIL
                    GLOBAL_INFO_WITH_THREAD_ID(
                            "memcpy from receive buffer[" << offset << ":" << offset + byteSize << "] to request["
                                                          << i << ", 0:" << tensor->elements() << "]")
#endif
                    memcpy(
                            tensor->data(), *bufferForReceiving + offset,
                            byteSize
                    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
                    GLOBAL_INFO_WITH_THREAD_ID("tensor:" << request->key() << " done tensors communicate")
#endif
                    request->done(STATUS_OK);
                }
                offset += byteSize;
            }
        }
    }
    // todo: status check
    return STATUS_OK;
}

void MPIRingTokenCommunication::classifyRequestsByDataType(
        const Requests &requests, std::map<DataType, Requests> &resultMap) {
    assert(resultMap.size() == 0);
    for (const auto &each : requests) {
        auto dtype = each->requestingTensor()->dtype();
        auto iter = resultMap.find(dtype);
        if (iter == resultMap.end()) {
            auto dtypeReq = Requests();
            dtypeReq.emplace_back(each);
            resultMap[dtype] = dtypeReq;
        } else {
            iter->second.emplace_back(each);
        }
    }
}


}}}}