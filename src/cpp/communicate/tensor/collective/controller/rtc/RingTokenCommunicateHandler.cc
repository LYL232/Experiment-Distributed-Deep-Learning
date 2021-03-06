//
// Created by LYL232 on 2021/3/23.
//

#include <global/Global.h>
#include "communicate/tensor/collective/controller/rtc/RingTokenCommunicateHandler.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

const std::string RingTokenCommunicateHandler::tokenSplitDelimiter_("\n");
const std::string RingTokenCommunicateHandler::tokenKeySplitDelimiter_("::");

RingTokenCommunicateHandler::RingTokenCommunicateHandler(
        std::shared_ptr<Communicator> communicator,
        std::shared_ptr<RingTokenCommunication> communicationImplement
) :
        outMutex_(PTHREAD_MUTEX_INITIALIZER),
        registerLock_(PTHREAD_RWLOCK_INITIALIZER),
        stageLock_(PTHREAD_RWLOCK_INITIALIZER),
        outputTokenCond_(PTHREAD_COND_INITIALIZER),
        currentStage_(RTCC_INIT),
        sendThread_(nullptr), recvThread_(nullptr),
        outputtingTokenQueue_(),
        waitingReadyTokenId_(),
        registeredRequest_(), communicationImplement_(std::move(communicationImplement)),
        communicator_(std::move(communicator)) {
    sendThread_ = new std::thread(&RingTokenCommunicateHandler::sendMain_, this);  // no mem track
    while (!initialized()) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    recvThread_ = new std::thread(&RingTokenCommunicateHandler::recvMain_, this);  // no mem track
}

RingTokenCommunicateHandler::~RingTokenCommunicateHandler() {
    fillTokenSendBufferAndNotify_(std::make_shared<Token>(
            Token::TOKEN_TYPE_SHUT_DOWN,
            Token::TOKEN_REQUEST_SHUTDOWN,
            "shut down"
    ));
    sendThread_->join();
    recvThread_->join();
    delete sendThread_;  // no mem track
    delete recvThread_;  // no mem track
    pthread_mutex_destroy(&outMutex_);
    pthread_rwlock_destroy(&stageLock_);
    pthread_rwlock_destroy(&registerLock_);
    pthread_cond_destroy(&outputTokenCond_);
}

void RingTokenCommunicateHandler::sendMain_() {
    using namespace std;
    int rank = communicator_->rank(),
            processes = communicator_->size(),
            receiverRank = (rank + 1) % processes;
    if (communicator_->size() <= 1) {
        pthread_mutex_lock(&outMutex_);
        fromStageToStage_(RTCC_INIT, RTCC_WAITING_TENSORS);
        pthread_mutex_unlock(&outMutex_);
        return;
    }
    while (!inStage_(RTCC_SHUT_DOWN)) {
        pthread_mutex_lock(&outMutex_);
        if (inStage_(RTCC_INIT)) {
            if (rank == 0) {
                fromStageToStage_(RTCC_INIT, RTCC_WAITING_TENSORS);
            } else {
                fromStageToStage_(RTCC_INIT, RTCC_WAITING_READY_TOKEN);
            }
        }
        while (outputtingTokenQueue_.empty()) {
            pthread_cond_wait(&outputTokenCond_, &outMutex_);
        }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_THREAD_MANNER
        GLOBAL_INFO_WITH_THREAD_ID("RingTokenCommunicateHandler send thread woke up")
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

            communicationImplement_->communicationSendTokenTo(receiverRank, token);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID("sent Token " << token->desc())
#endif

            shutdown = token->isShutDown();
        }
        if (shutdown) {
            forceToStage_(RTCC_SHUT_DOWN);
            break;
        }
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_THREAD_ID("RingTokenCommunicateHandler send thread exit")
#endif
}

void RingTokenCommunicateHandler::recvMain_() {
    using namespace std;
    int rank = communicator_->rank(),
            processes = communicator_->size(),
            senderRank = (rank - 1 + processes) % processes;
    while (communicator_->size() > 1 && !inStage_(RTCC_SHUT_DOWN)) {
        auto token = communicationImplement_->communicationReceiveTokenFrom(senderRank);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
        GLOBAL_INFO_WITH_THREAD_ID("received Token " << token->desc())
#endif

        if (token->isShutDown()) {
            forceToStage_(RTCC_SHUT_DOWN);
            break;
        }
        if (rank == 0) {
            handleReceivingTokenAsTokenGenerator_(token);
        } else {
            handleReceivingTokenAsTokenReceiver_(token);
        }
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_THREAD_ID("RingTokenCommunicateHandler receive thread exit")
#endif
}

void RingTokenCommunicateHandler::handleReceivingTokenAsTokenGenerator_(std::shared_ptr<Token> token) {
    using namespace std;

    switch (token->type()) {
        case Token::TOKEN_TYPE_READY: {
            assert(inStage_(RTCC_WAITING_READY_TOKEN));

            string readyKeys;
            pthread_rwlock_rdlock(&registerLock_);
            for (auto &iter : registeredRequest_) {
                if (iter.first.first == token->requestTypeName()) {
                    readyKeys.append(iter.first.first).append(tokenKeySplitDelimiter_).append(iter.first.second)
                            .append(tokenSplitDelimiter_);
                }
            }
            pthread_rwlock_unlock(&registerLock_);

            fromStageToStage_(RTCC_WAITING_READY_TOKEN, RTCC_WAITING_SYNC_TOKEN);

            token = make_shared<Token>(
                    Token::TOKEN_TYPE_SYNC,
                    token->requestType(),
                    move(readyKeys)
            );
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTCC_RECV_SYNC_TOKEN")
#endif
            fillTokenSendBufferAndNotify_(token);
            break;
        }
        case Token::TOKEN_TYPE_SYNC: {
            assert(inStage_(RTCC_WAITING_SYNC_TOKEN));

            token = make_shared<Token>(
                    Token::TOKEN_TYPE_COMMUNICATE,
                    token->requestType(),
                    token->movingMsg()
            );
            fromStageToStage_(RTCC_WAITING_SYNC_TOKEN, RTCC_WAITING_COMMUNICATE_TOKEN);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTCC_RECV_COMMUNICATE_TOKEN")
#endif
            fillTokenSendBufferAndNotify_(token);
            break;
        }
        case Token::TOKEN_TYPE_COMMUNICATE: {
            assert(inStage_(RTCC_WAITING_COMMUNICATE_TOKEN));

            fromStageToStage_(RTCC_WAITING_COMMUNICATE_TOKEN, RTCC_COMMUNICATING);
            communicateById_(getIdSetFromToken_(*token));
            pthread_rwlock_rdlock(&registerLock_);
            if (!registeredRequest_.empty()) {
                token = make_shared<Token>(
                        Token::TOKEN_TYPE_READY,
                        token->requestType(),
                        registeredRequest_.begin()->first.second
                );
                fromStageToStage_(RTCC_COMMUNICATING, RTCC_WAITING_READY_TOKEN);
                pthread_rwlock_unlock(&registerLock_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                GLOBAL_INFO_WITH_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTCC_RECV_READY_TOKEN")
#endif
                fillTokenSendBufferAndNotify_(token);
            } else {
                fromStageToStage_(RTCC_COMMUNICATING, RTCC_WAITING_TENSORS);
                pthread_rwlock_unlock(&registerLock_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                GLOBAL_INFO_WITH_THREAD_ID(
                        "wait for registering: to stage RTCC_WAITING_TENSORS")
#endif
            }
            break;
        }
        case Token::TOKEN_TYPE_SHUT_DOWN: {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID("shut down token received")
            return;
#endif
        }
    }
}

void RingTokenCommunicateHandler::handleReceivingTokenAsTokenReceiver_(std::shared_ptr<Token> token) {
    using namespace std;

    switch (token->type()) {
        case Token::TOKEN_TYPE_READY: {
            assert(inStage_(RTCC_WAITING_READY_TOKEN));

            pthread_rwlock_rdlock(&registerLock_);
            if (registeredRequest_.find(
                    make_pair(token->requestTypeName(), token->msg())
            ) != registeredRequest_.end()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                GLOBAL_INFO_WITH_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTCC_RECV_SYNC_TOKEN")
#endif
                fillTokenSendBufferAndNotify_(token);
                fromStageToStage_(RTCC_WAITING_READY_TOKEN, RTCC_WAITING_SYNC_TOKEN);
            } else {
                waitingReadyTokenId_ = make_pair(token->requestTypeName(), token->msg());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                GLOBAL_INFO_WITH_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTCC_WAITING_TENSORS")
#endif
                fromStageToStage_(RTCC_WAITING_READY_TOKEN, RTCC_WAITING_TENSORS);
            }
            pthread_rwlock_unlock(&registerLock_);
            break;
        }
        case Token::TOKEN_TYPE_SYNC: {
            assert(inStage_(RTCC_WAITING_SYNC_TOKEN));

            auto idSet = getIdSetFromToken_(*token);
            set<RequestIdentifier> notReady;

            pthread_rwlock_rdlock(&registerLock_);
            for (const auto &iter : idSet) {
                if (registeredRequest_.find(iter) == registeredRequest_.end()) {
                    notReady.emplace(iter);
                }
            }
            pthread_rwlock_unlock(&registerLock_);

            if (!notReady.empty()) {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                string notReadyNames("{\n");
#endif
                for (const auto &iter : notReady) {
                    idSet.erase(iter);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                    notReadyNames.append("\t").append(iter.first)
                            .append(":").append(iter.second).append("\n");
#endif
                }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
                notReadyNames.append("}\n");
                GLOBAL_INFO_WITH_THREAD_ID(
                        "has not ready Tensors: "
                                << notReadyNames << " filter them")
#endif
                string allReadyKeys;
                for (const auto &iter : idSet) {
                    allReadyKeys.append(iter.first).append(tokenKeySplitDelimiter_)
                            .append(iter.second).append(tokenSplitDelimiter_);
                }
                token = make_shared<Token>(
                        token->type(),
                        token->requestType(),
                        move(allReadyKeys)
                );
            }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID(
                    "forward token: " << token->desc()
                                      << " to stage RTCC_RECV_SYNC_TOKEN";)
#endif
            fillTokenSendBufferAndNotify_(token);
            fromStageToStage_(RTCC_WAITING_SYNC_TOKEN, RTCC_WAITING_COMMUNICATE_TOKEN);
            break;
        }
        case Token::TOKEN_TYPE_COMMUNICATE: {
            assert(inStage_(RTCC_WAITING_COMMUNICATE_TOKEN));

            fromStageToStage_(RTCC_WAITING_COMMUNICATE_TOKEN, RTCC_COMMUNICATING);
            fillTokenSendBufferAndNotify_(token);
            communicateById_(getIdSetFromToken_(*token));
            fromStageToStage_(RTCC_COMMUNICATING, RTCC_WAITING_READY_TOKEN);
            break;
        }
        case Token::TOKEN_TYPE_SHUT_DOWN: {
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN
            GLOBAL_INFO_WITH_THREAD_ID("shut down token received")
#endif
            return;
        }
    }
}

void RingTokenCommunicateHandler::fillTokenSendBufferAndNotify_(const std::shared_ptr<Token> &token) {
    pthread_mutex_lock(&outMutex_);
    outputtingTokenQueue_.emplace(token);
    pthread_cond_signal(&outputTokenCond_);
    pthread_mutex_unlock(&outMutex_);
}

StatusCode RingTokenCommunicateHandler::handleRequest(
        const std::shared_ptr<TensorCollectiveCommunicateRequest> &request
) {
    using namespace std;
    assert(*request->communicator() == *communicator_);
    assert(!inStage_(RTCC_INIT));
    pthread_rwlock_wrlock(&registerLock_);
    RequestIdentifier id = make_pair(request->requestTypeName(), request->key());
    assert(registeredRequest_.find(id) == registeredRequest_.end());
    registeredRequest_.emplace(id, request);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION
    GLOBAL_INFO_WITH_THREAD_ID(
            "op request to " << request->requestTypeName() << " tensor: " << id.second)
#endif
    if (communicator_->rank() == 0) {
        if (inStage_(RTCC_WAITING_TENSORS)) {
            fromStageToStage_(RTCC_WAITING_TENSORS, RTCC_WAITING_READY_TOKEN);
            fillTokenSendBufferAndNotify_(make_shared<Token>(
                    Token::TOKEN_TYPE_READY,
                    Token::requestType(id.first),
                    id.second
            ));
        }
    } else {
        if (id == waitingReadyTokenId_) {
            fromStageToStage_(RTCC_WAITING_TENSORS, RTCC_WAITING_SYNC_TOKEN);
            fillTokenSendBufferAndNotify_(make_shared<Token>(
                    Token::TOKEN_TYPE_READY,
                    Token::requestType(id.first),
                    id.second
            ));
            waitingReadyTokenId_ = make_pair("", "");
        }
    }
    pthread_rwlock_unlock(&registerLock_);
    return STATUS_OK;
}

StatusCode RingTokenCommunicateHandler::communicateById_(
        const std::set<RequestIdentifier> &idSet) {
    using namespace std;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_COMMUNICATE
    string communicatingTensorsDesc = "(\n";
#endif
    pthread_rwlock_wrlock(&registerLock_);
    vector<shared_ptr<TensorCollectiveCommunicateRequest>> requests;
    for (auto i = idSet.begin(); i != idSet.end(); ++i) {
        const auto &id = *i;
        auto iter = registeredRequest_.find(id);

        if (iter == registeredRequest_.end()) {
            string r = "registered Tensors: {\n";
            for (auto &_iter : registeredRequest_) {
                r.append("\t").append(_iter.first.first).append(":")
                        .append(_iter.first.second).append("\n");
            }
            string n = "{\n";
            for (const auto &_iter : idSet) {
                n.append("\t").append(_iter.first).append(":")
                        .append(_iter.second).append("\n");
            }
            n.append("}");
            r.append("}, but does not contain all needing Tensor: ").append(n);
            GLOBAL_ERROR_WITH_THREAD_ID(r)
        }
        assert(iter != registeredRequest_.end());
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_COMMUNICATE
        communicatingTensorsDesc.append("\t").append(id.first)
                .append(tokenKeySplitDelimiter_)
                .append(id.second).append(", \n");
#endif
        requests.emplace_back(iter->second);
        registeredRequest_.erase(iter);
    }
    pthread_rwlock_unlock(&registerLock_);

    assert(!requests.empty());

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_COMMUNICATE
    communicatingTensorsDesc.append(")");
    GLOBAL_INFO_WITH_THREAD_ID("communicating Tensors: " << communicatingTensorsDesc)
#endif
    return requests.front()->collectiveCommunicate(requests);
}

std::set<RingTokenCommunicateHandler::RequestIdentifier>
RingTokenCommunicateHandler::getIdSetFromToken_(const Token &token) {
    using namespace std;
    set<RequestIdentifier> res;
    const string &str = token.msg();
    if (str.empty()) {
        return res;
    }
    string strs = str;

    size_t pos = strs.find(tokenSplitDelimiter_);
    size_t size = strs.size();

    while (pos != string::npos) {
        string tokenStr = strs.substr(0, pos);
        strs = strs.substr(pos + tokenSplitDelimiter_.length(), size);
        pos = tokenStr.find(tokenKeySplitDelimiter_);
        assert(pos != string::npos);
        res.emplace(
                tokenStr.substr(0, pos),
                tokenStr.substr(pos + tokenKeySplitDelimiter_.length(), tokenStr.length())
        );
        pos = strs.find(tokenSplitDelimiter_);
    }
    return res;
}

void RingTokenCommunicateHandler::fromStageToStage_(Stage from, Stage to) {
    pthread_rwlock_wrlock(&stageLock_);
    if (currentStage_ == from) {
        currentStage_ = to;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_STAGE_CHANGE
        GLOBAL_INFO_WITH_THREAD_ID(
                "from stage: " << stageName_(from)
                               << " to stage: " << stageName_(to))
#endif
    }
    pthread_rwlock_unlock(&stageLock_);
}

bool RingTokenCommunicateHandler::inStage_(Stage stage) {
    pthread_rwlock_rdlock(&stageLock_);
    bool res = currentStage_ == stage;
    pthread_rwlock_unlock(&stageLock_);
    return res;
}

StatusCode RingTokenCommunicateHandler::allreduce(const Requests &requests) {
    return communicationImplement_->allreduceRequests(requests);
}

StatusCode RingTokenCommunicateHandler::allgather(const Requests &requests) {
    return communicationImplement_->allgatherRequests(requests);
}

StatusCode RingTokenCommunicateHandler::broadcast(const Requests &requests) {
    return communicationImplement_->broadcastRequests(requests);
}

void RingTokenCommunicateHandler::forceToStage_(Stage to) {
    pthread_rwlock_wrlock(&stageLock_);
    currentStage_ = to;
    pthread_rwlock_unlock(&stageLock_);
}

std::string RingTokenCommunicateHandler::stageName_(Stage stage) noexcept {
    switch (stage) {
        case RTCC_INIT:
            return "RTCC_INIT";
        case RTCC_WAITING_TENSORS:
            return "RTCC_WAITING_TENSORS";
        case RTCC_WAITING_READY_TOKEN:
            return "RTCC_WAITING_READY_TOKEN";
        case RTCC_WAITING_SYNC_TOKEN:
            return "RTCC_WAITING_SYNC_TOKEN";
        case RTCC_WAITING_COMMUNICATE_TOKEN:
            return "RTCC_WAITING_COMMUNICATE_TOKEN";
        case RTCC_COMMUNICATING:
            return "RTCC_COMMUNICATING";
        case RTCC_SHUT_DOWN:
            return "RTCC_SHUT_DOWN";
    }
    return "RTCC_UNKNOWN";
}

}}}}