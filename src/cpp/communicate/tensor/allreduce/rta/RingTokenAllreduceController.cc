//
// Created by LYL232 on 2021/2/8.
//

#include <assert.h>
#include <algorithm>
#include "communicate/communication/MPICommunication.h"
#include "communicate/tensor/allreduce/rta/RingTokenAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce { namespace rta {

const std::string &Token::desc() const {
    using namespace std;
    if (desc_.length() == 0) {
        desc_.append("{type: ");
        switch (type_) {
            case TOKEN_TYPE_READY:
                desc_.append("TOKEN_TYPE_READY");
                break;
            case TOKEN_TYPE_SYNC:
                desc_.append("TOKEN_TYPE_SYNC");
                break;
            case TOKEN_TYPE_ALLREDUCE:
                desc_.append("TOKEN_TYPE_ALLREDUCE");
                break;
            case TOKEN_TYPE_SHUT_DOWN:
                desc_.append("TOKEN_TYPE_SHUT_DOWN");
                break;
        }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_TOKEN_DESC_SHOW_MSG
        desc_.append(", msg: ").append(msg_);
#endif
        desc_.append("}");
    }
    return desc_;
}

double RingTokenAllreduceController::inflateFactor_ = 1.5;
std::string RingTokenAllreduceController::tokenNameSplitDelimiter_("\n");

RingTokenAllreduceController::~RingTokenAllreduceController() {
    using namespace std;

    GLOBAL_INFO_WITH_RANK("controller destructing");

    fillTokenSendBufferAndNotify(make_shared<Token>(Token::shutDownToken()));

    sendThread_.join();
    recvThread_.join();

    delete[]sendBuffer_;
    delete[]recvBuffer_;
    delete[]allreduceSendBuffer_;
    delete[]allreduceRecvBuffer_;
    pthread_mutex_destroy(&outMutex_);
    pthread_rwlock_destroy(&stageLock_);
    pthread_rwlock_destroy(&registerTensorLock_);
    pthread_cond_destroy(&outputTokenCond_);
}

void RingTokenAllreduceController::checkSendBuffer_(size_t bytesRequire) {
    if (sendBufferSize_ < bytesRequire) {
        delete[]sendBuffer_;
        sendBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        sendBuffer_ = new char[sendBufferSize_];
    }
}

void RingTokenAllreduceController::checkRecvBuffer_(size_t bytesRequire) {
    if (recvBufferSize_ < bytesRequire) {
        delete[]recvBuffer_;
        recvBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        recvBuffer_ = new char[recvBufferSize_];
    }
}

void RingTokenAllreduceController::checkAllreduceBuffer_(size_t bytesRequire) {
    if (allreduceBufferSize_ < bytesRequire) {
        delete[]allreduceSendBuffer_;
        delete[]allreduceRecvBuffer_;
        allreduceBufferSize_ = (size_t) ((double) bytesRequire * inflateFactor_);
        allreduceSendBuffer_ = new char[allreduceBufferSize_];
        allreduceRecvBuffer_ = new char[allreduceBufferSize_];
    }
}

void RingTokenAllreduceController::sendMain_() {
    using namespace std;
    auto &global = Global::get();
    size_t stringSize;
    int worldRank = global.MPIWorldRank(),
            worldSize = global.MPIWorldSize(),
            receiverRank = (worldRank + 1) % worldSize,
            offset;

    Token::Type tType;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("send thread started");
#endif

    while (global.MPIWorldSize() > 1) {
        pthread_mutex_lock(&outMutex_);
        if (inStage_(RTAC_INIT)) {
            if (worldRank == 0) {
                fromStageToStage_(RTAC_INIT, RTAC_WAITING_TENSORS);
            } else {
                fromStageToStage_(RTAC_INIT, RTAC_WAITING_READY_TOKEN);
            }
        }
        while (outputtingTokenQueue_.empty()) {
            pthread_cond_wait(&outputTokenCond_, &outMutex_);
        }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
        GLOBAL_INFO_WITH_RANK_THREAD_ID("send thread woke up");
#endif
        queue<shared_ptr<Token>> sendingTokens;
        while (!outputtingTokenQueue_.empty()) {
            sendingTokens.emplace(outputtingTokenQueue_.front());
            outputtingTokenQueue_.pop();
        }
        pthread_mutex_unlock(&outMutex_);

        bool shutdown = false;

        while (!sendingTokens.empty()) {
            auto token = sendingTokens.front();
            sendingTokens.pop();
            stringSize = token->msg().length();
            tType = token->type();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_RANK_THREAD_ID("before mpi sending Token");
#endif
            checkSendBuffer_(max(tokenMetaSize_, stringSize));
            // 发送Token:
            // 1. 先打包发送描述信息: token.SourceRank, token.msg_.length()
            // 2. 再发送token.msg_.length()的字符串
            offset = 0;
            MPI_Pack(&tType, sizeof(Token::Type), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
            MPI_Pack(&stringSize, sizeof(size_t), MPI_BYTE, sendBuffer_, tokenMetaSize_, &offset, MPI_COMM_WORLD);
            MPI_Send(
                    sendBuffer_, offset, MPI_PACKED, receiverRank,
                    MPICommunication::MPI_TAG_RTA_COMMUNICATE_META, MPI_COMM_WORLD
            );
            MPI_Send(
                    token->msg().c_str(), (int) stringSize, MPI_CHAR, receiverRank,
                    MPICommunication::MPI_TAG_RTA_COMMUNICATE_MSG, MPI_COMM_WORLD
            );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
            GLOBAL_INFO_WITH_RANK_THREAD_ID("mpi sent Token");
#endif
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID("sent Token " << token->desc());
#endif

            shutdown = token->isShutDown();
        }
        if (shutdown) {
            forceToStage_(RTAC_SHUT_DOWN);
            break;
        }
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("send thread exit");
#endif
}

void RingTokenAllreduceController::recvMain_() {
    using namespace std;
    int worldRank = Global::get().MPIWorldRank();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread started");
#endif

    while (Global::get().MPIWorldSize() > 1 && !inStage_(RTAC_SHUT_DOWN)) {

        auto token = receiveTokenFromSender();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
        GLOBAL_INFO_WITH_RANK_THREAD_ID("received Token " << token->desc());
#endif

        if (token->isShutDown()) {
            forceToStage_(RTAC_SHUT_DOWN);
            break;
        }
        if (worldRank == 0) {
            handleReceivingTokenAsTokenGenerator(token);
        } else {
            handleReceivingTokenAsTokenReceiver(token);
        }
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread exit");
#endif
}

std::shared_ptr<Token> RingTokenAllreduceController::receiveTokenFromSender() {
    auto &global = Global::get();
    int worldRank = global.MPIWorldRank(),
            worldSize = global.MPIWorldSize(),
            senderRank = (worldRank - 1 + worldSize) % worldSize,
            offset;
    size_t stringSize;
    Token::Type tType;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread waiting rank " << senderRank << " for Token");
#endif

    checkRecvBuffer_(tokenMetaSize_);
    MPI_Recv(
            recvBuffer_, tokenMetaSize_, MPI_PACKED, senderRank,
            MPICommunication::MPI_TAG_RTA_COMMUNICATE_META,
            MPI_COMM_WORLD, &statusBuffer_
    );

    // todo: 检查status_, 出错则做相应处理
    offset = 0;
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &tType, sizeof(Token::Type), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(recvBuffer_, tokenMetaSize_, &offset, &stringSize, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    checkRecvBuffer_(stringSize + 1);

    MPI_Recv(
            recvBuffer_, stringSize, MPI_PACKED, senderRank,
            MPICommunication::MPI_TAG_RTA_COMMUNICATE_MSG,
            MPI_COMM_WORLD, &statusBuffer_
    );
    // todo: 检查status_, 出错则做相应处理

    recvBuffer_[stringSize] = 0;
    return std::make_shared<Token>(tType, recvBuffer_);
}


void RingTokenAllreduceController::handleReceivingTokenAsTokenGenerator(std::shared_ptr<Token> token) {
    using namespace std;

    switch (token->type()) {
        case Token::TOKEN_TYPE_READY: {
            assert(inStage_(RTAC_WAITING_READY_TOKEN));

            string allReadyNames;
            pthread_rwlock_rdlock(&registerTensorLock_);
            for (auto iter = registeredTensors_.begin(); iter != registeredTensors_.end(); ++iter) {
                allReadyNames.append(iter->first).append(tokenNameSplitDelimiter_);
            }
            pthread_rwlock_unlock(&registerTensorLock_);

            fromStageToStage_(RTAC_WAITING_READY_TOKEN, RTAC_WAITING_SYNC_TOKEN);

            token = make_shared<Token>(Token::TOKEN_TYPE_SYNC, move(allReadyNames));
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTAC_RECV_SYNC_TOKEN");
#endif
            fillTokenSendBufferAndNotify(token);
            break;
        }
        case Token::TOKEN_TYPE_SYNC: {
            assert(inStage_(RTAC_WAITING_SYNC_TOKEN));

            token = make_shared<Token>(Token::TOKEN_TYPE_ALLREDUCE, token->movingMsg());
            fromStageToStage_(RTAC_WAITING_SYNC_TOKEN, RTAC_WAITING_ALLREDUCE_TOKEN);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTAC_RECV_ALLREDUCE_TOKEN");
#endif
            fillTokenSendBufferAndNotify(token);
            break;
        }
        case Token::TOKEN_TYPE_ALLREDUCE: {
            assert(inStage_(RTAC_WAITING_ALLREDUCE_TOKEN));

            fromStageToStage_(RTAC_WAITING_ALLREDUCE_TOKEN, RTAC_ALLREDUCING);
            allreduceByNames(getNamesFromToken(*token));
            pthread_rwlock_rdlock(&registerTensorLock_);
            if (!registeredTensors_.empty()) {
                token = make_shared<Token>(
                        Token::TOKEN_TYPE_READY,
                        registeredTensors_.begin()->first);
                fromStageToStage_(RTAC_ALLREDUCING, RTAC_WAITING_READY_TOKEN);
                pthread_rwlock_unlock(&registerTensorLock_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTAC_RECV_READY_TOKEN");
#endif
                fillTokenSendBufferAndNotify(token);
            } else {
                fromStageToStage_(RTAC_ALLREDUCING, RTAC_WAITING_TENSORS);
                pthread_rwlock_unlock(&registerTensorLock_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "wait for registering: to stage RTAC_WAITING_TENSORS");
#endif
            }
            break;
        }
        case Token::TOKEN_TYPE_SHUT_DOWN: {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID("shut down token received");
            return;
#endif
        }
    }
}

void RingTokenAllreduceController::handleReceivingTokenAsTokenReceiver(std::shared_ptr<Token> token) {
    using namespace std;

    switch (token->type()) {
        case Token::TOKEN_TYPE_READY: {
            assert(inStage_(RTAC_WAITING_READY_TOKEN));

            pthread_rwlock_rdlock(&registerTensorLock_);
            if (registeredTensors_.find(token->msg()) != registeredTensors_.end()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTAC_RECV_SYNC_TOKEN");
#endif
                fillTokenSendBufferAndNotify(token);
                fromStageToStage_(RTAC_WAITING_READY_TOKEN, RTAC_WAITING_SYNC_TOKEN);
            } else {
                waitingReadyTokenName_ = token->msg();
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTAC_WAITING_TENSORS");
#endif
                fromStageToStage_(RTAC_WAITING_READY_TOKEN, RTAC_WAITING_TENSORS);
            }
            pthread_rwlock_unlock(&registerTensorLock_);
            break;
        }
        case Token::TOKEN_TYPE_SYNC: {
            assert(inStage_(RTAC_WAITING_SYNC_TOKEN));

            auto names = getNamesFromToken(*token);
            set<string> notReady;

            pthread_rwlock_rdlock(&registerTensorLock_);
            for (auto iter = names.begin(); iter != names.end(); ++iter) {
                if (registeredTensors_.find(*iter) == registeredTensors_.end()) {
                    notReady.emplace(*iter);
                }
            }
            pthread_rwlock_unlock(&registerTensorLock_);

            if (!notReady.empty()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                string notReadyNames("{\n");
#endif
                for (auto iter = notReady.begin(); iter != notReady.end(); ++iter) {
                    names.erase(*iter);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                    notReadyNames.append("\t").append(*iter).append("\n");
#endif
                }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                notReadyNames.append("}\n");
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "has not ready Tensors: "
                                << notReadyNames << " filter them");
#endif
                string allReadyNames;
                for (auto iter = names.begin(); iter != names.end(); ++iter) {
                    allReadyNames.append(*iter).append(tokenNameSplitDelimiter_);
                }
                token = make_shared<Token>(token->type(), move(allReadyNames));
            }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTAC_RECV_SYNC_TOKEN";);
#endif
            fillTokenSendBufferAndNotify(token);
            fromStageToStage_(RTAC_WAITING_SYNC_TOKEN, RTAC_WAITING_ALLREDUCE_TOKEN);
            break;
        }
        case Token::TOKEN_TYPE_ALLREDUCE: {
            assert(inStage_(RTAC_WAITING_ALLREDUCE_TOKEN));

            fromStageToStage_(RTAC_WAITING_ALLREDUCE_TOKEN, RTAC_ALLREDUCING);
            auto names = getNamesFromToken(*token);
            fillTokenSendBufferAndNotify(token);
            allreduceByNames(names);
            fromStageToStage_(RTAC_ALLREDUCING, RTAC_WAITING_READY_TOKEN);
            break;
        }
        case Token::TOKEN_TYPE_SHUT_DOWN: {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
            GLOBAL_INFO_WITH_RANK_THREAD_ID("shut down token received");
#endif
            return;
        }
    }
}

void RingTokenAllreduceController::fillTokenSendBufferAndNotify(std::shared_ptr<Token> token) {
    pthread_mutex_lock(&outMutex_);
    outputtingTokenQueue_.emplace(token);
    pthread_cond_signal(&outputTokenCond_);
    pthread_mutex_unlock(&outMutex_);
}

StatusCode RingTokenAllreduceController::handleTenorAllreduceRequest(
        const std::string &name,
        std::shared_ptr<tensorflow::Tensor> sendTensor,
        std::shared_ptr<tensorflow::Tensor> recvTensor,
        std::function<void(StatusCode)> done) {
    using namespace std;
    auto &global = Global::get();
    assert(!inStage_(RTAC_INIT));
    pthread_rwlock_wrlock(&registerTensorLock_);
    assert(registeredTensors_.find(name) == registeredTensors_.end());
    auto *entry = new TensorEntry(name, sendTensor, recvTensor, done);
    registeredTensors_.emplace(name, entry);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION
    GLOBAL_INFO_WITH_RANK_THREAD_ID("op request to allreuce tensor: " << name);
#endif
    if (global.MPIWorldRank() == 0) {
        if (inStage_(RTAC_WAITING_TENSORS)) {
            fromStageToStage_(RTAC_WAITING_TENSORS, RTAC_WAITING_READY_TOKEN);
            fillTokenSendBufferAndNotify(make_shared<Token>(Token::TOKEN_TYPE_READY, name));
        }
    } else {
        if (name == waitingReadyTokenName_) {
            fromStageToStage_(RTAC_WAITING_TENSORS, RTAC_WAITING_SYNC_TOKEN);
            fillTokenSendBufferAndNotify(make_shared<Token>(Token::TOKEN_TYPE_READY, name));
            waitingReadyTokenName_ = "";
        }
    }
    pthread_rwlock_unlock(&registerTensorLock_);
    return STATUS_OK;
}

void RingTokenAllreduceController::allreduceByNames(const std::set<std::string> &names) {
    using namespace std;

    size_t allreduceSize = 0, elements = 0, offset;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
    string allreducingTensorsDesc = "(\n";
#endif
    pthread_rwlock_wrlock(&registerTensorLock_);
    map<string, TensorEntry *> entries;
    for (auto i = names.begin(); i != names.end(); ++i) {
        const auto &name = *i;
        auto iter = registeredTensors_.find(name);

        if (iter == registeredTensors_.end()) {
            string r = "registered Tensors: {\n";
            for (auto _iter = registeredTensors_.begin(); _iter != registeredTensors_.end(); ++_iter) {
                r.append("\t").append(_iter->first).append("\n");
            }
            string n = "{\n";
            for (auto _iter = names.begin(); _iter != names.end(); ++_iter) {
                n.append("\t").append(*_iter).append("\n");
            }
            n.append("}");
            r.append("}, but does not contain all needing Tensor: ").append(n);
            GLOBAL_ERROR_WITH_RANK(r);
        }
        assert(iter != registeredTensors_.end());
        TensorEntry *entry = iter->second;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
        allreducingTensorsDesc.append("\t").append(name).append(", \n");
#endif
        entries.emplace(name, entry);
        allreduceSize += entry->tensorSize();
        elements += entry->elements();
        registeredTensors_.erase(iter);
    }
    pthread_rwlock_unlock(&registerTensorLock_);


#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
    allreducingTensorsDesc.append(")");
    GLOBAL_INFO_WITH_RANK_THREAD_ID(
            "allreducing Tensors: " << allreducingTensorsDesc
                                    << " allreducing size: " << allreduceSize);
#endif
    checkAllreduceBuffer_(allreduceSize);
    const auto firstEntry = entries.begin()->second;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL
    GLOBAL_INFO_WITH_RANK_THREAD_ID("copying memory from input tensors to tensorsallreduce buffer");
#endif

    offset = 0;
    for (auto i = entries.begin(); i != entries.end(); ++i) {
        TensorEntry *entry = i->second;
        memcpy(
                allreduceSendBuffer_ + offset,
                entry->sendTensorData(),
                entry->tensorSize()
        );
        offset += entry->tensorSize();
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_MPI_CALLS
    GLOBAL_INFO_WITH_RANK_THREAD_ID("before mpi tensorsallreduce");
#endif
    MPI_Allreduce(
            allreduceSendBuffer_, allreduceRecvBuffer_,
            (int) elements,
            MPICommunication::DataType2MPIType(firstEntry->dtype()),
            MPI_SUM,
            MPI_COMM_WORLD
    );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE_DETAIL
    GLOBAL_INFO_WITH_RANK_THREAD_ID("mpi allreduced, copying memory from tensorsallreduce buffer to output tensors");
#endif
    offset = 0;
    for (auto i = entries.begin(); i != entries.end(); ++i) {
        TensorEntry *entry = i->second;
        memcpy(
                entry->recvTensorData(),
                allreduceRecvBuffer_ + offset,
                entry->tensorSize()
        );
        offset += entry->tensorSize();

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION
        GLOBAL_INFO_WITH_RANK_THREAD_ID("tensor:" << entry->name() << " done tensorsallreduce");
#endif
        entry->done(STATUS_OK);
        delete entry;
    }
}

std::set<std::string> RingTokenAllreduceController::getNamesFromToken(const Token &token) {
    using namespace std;
    set<string> res;
    const string &str = token.msg();
    if ("" == str) {
        return res;
    }
    std::string strs = str;

    size_t pos = strs.find(tokenNameSplitDelimiter_);
    size_t size = strs.size();

    while (pos != string::npos) {
        res.emplace(strs.substr(0, pos));
        strs = strs.substr(pos + 1, size);
        pos = strs.find(tokenNameSplitDelimiter_);
    }

    return res;
}

void RingTokenAllreduceController::fromStageToStage_(
        RingTokenAllreduceController::Stage from,
        RingTokenAllreduceController::Stage to) {
    pthread_rwlock_wrlock(&stageLock_);
    if (currentStage_ == from) {
        currentStage_ = to;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_STAGE_CHANGE
        GLOBAL_INFO_WITH_RANK_THREAD_ID(
                "from stage: " << stageName_(from)
                               << " to stage: " << stageName_(to));
#endif
    }
    pthread_rwlock_unlock(&stageLock_);
}

bool RingTokenAllreduceController::inStage_(Stage stage) {
    pthread_rwlock_rdlock(&stageLock_);
    bool res = currentStage_ == stage;
    pthread_rwlock_unlock(&stageLock_);
    return res;
}

void RingTokenAllreduceController::forceToStage_(RingTokenAllreduceController::Stage to) {
    pthread_rwlock_wrlock(&stageLock_);
    currentStage_ = to;
    pthread_rwlock_unlock(&stageLock_);
}

std::string RingTokenAllreduceController::stageName_(RingTokenAllreduceController::Stage stage) {
    switch (stage) {
        case RTAC_INIT:
            return "RTAC_INIT";
        case RTAC_WAITING_TENSORS:
            return "RTAC_WAITING_TENSORS";
        case RTAC_WAITING_READY_TOKEN:
            return "RTAC_WAITING_READY_TOKEN";
        case RTAC_WAITING_SYNC_TOKEN:
            return "RTAC_WAITING_SYNC_TOKEN";
        case RTAC_WAITING_ALLREDUCE_TOKEN:
            return "RTAC_WAITING_ALLREDUCE_TOKEN";
        case RTAC_ALLREDUCING:
            return "RTAC_ALLREDUCING";
        case RTAC_SHUT_DOWN:
            return "RTAC_SHUT_DOWN";
    }
    return "RTAC_UNKNOWN";
}
}}}}}