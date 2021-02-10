//
// Created by LYL232 on 2021/2/8.
//

#include <assert.h>
#include <algorithm>
#include "communicate/tensor/allreduce/rta/RingTokenAllreduceController.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace tensorsallreduce {
namespace rta {

std::string RingTokenAllreduceController::tokenNameSplitDelimiter_("\n");

RingTokenAllreduceController::RingTokenAllreduceController(
        std::ostream &initializingLogStream,
        std::shared_ptr<CommunicationBackend> communication
) :
        outMutex_(PTHREAD_MUTEX_INITIALIZER),
        registerLock_(PTHREAD_RWLOCK_INITIALIZER),
        stageLock_(PTHREAD_RWLOCK_INITIALIZER),
        outputTokenCond_(PTHREAD_COND_INITIALIZER),
        currentStage_(RTAC_INIT),
        sendThread_(&RingTokenAllreduceController::sendMain_, this, communication),
        recvThread_(&RingTokenAllreduceController::recvMain_, this, communication),
        outputtingTokenQueue_(),
        waitingReadyTokenName_(),
        registeredRequest_() {
    std::string controllerLogStr("RingTokenAllreduceController:");
    controllerLogStr.append(std::to_string((size_t) this)).append(" ");
    initializingLogStream << STREAM_WITH_RANK(
            "new " << controllerLogStr << ", waiting for initialization" << std::endl,
            communication->processRank());
    while (!initialized()) {
        initializingLogStream << STREAM_WITH_RANK(
                controllerLogStr << "initialization check failed" << std::endl,
                communication->processRank());
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    initializingLogStream << STREAM_WITH_RANK(
            controllerLogStr << "initialized" << std::endl, communication->processRank()
    );

}

RingTokenAllreduceController::~RingTokenAllreduceController() {
    pthread_mutex_destroy(&outMutex_);
    pthread_rwlock_destroy(&stageLock_);
    pthread_rwlock_destroy(&registerLock_);
    pthread_cond_destroy(&outputTokenCond_);
}

void RingTokenAllreduceController::sendMain_(
        std::shared_ptr<CommunicationBackend> communication
) {
    using namespace std;
    int rank = communication->processRank(),
            processes = communication->processes(),
            receiverRank = (rank + 1) % processes;

    while (communication->processes() > 1 && !inStage_(RTAC_SHUT_DOWN)) {
        pthread_mutex_lock(&outMutex_);
        if (inStage_(RTAC_INIT)) {
            if (rank == 0) {
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

            this->communicationSendTokenTo(receiverRank, token);
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

void RingTokenAllreduceController::recvMain_(std::shared_ptr<CommunicationBackend> communication) {
    using namespace std;
    int rank = communication->processRank(),
            processes = communication->processes(),
            senderRank = (rank - 1 + processes) % processes;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread started");
#endif

    while (Global::get().processes() > 1 && !inStage_(RTAC_SHUT_DOWN)) {

        auto token = this->communicationReceiveTokenFrom(senderRank);

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
        GLOBAL_INFO_WITH_RANK_THREAD_ID("received Token " << token->desc());
#endif

        if (token->isShutDown()) {
            forceToStage_(RTAC_SHUT_DOWN);
            break;
        }
        if (rank == 0) {
            handleReceivingTokenAsTokenGenerator(token);
        } else {
            handleReceivingTokenAsTokenReceiver(token);
        }
    }
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_THREAD_MANNER
    GLOBAL_INFO_WITH_RANK_THREAD_ID("receive thread exit");
#endif
}


void RingTokenAllreduceController::handleReceivingTokenAsTokenGenerator(std::shared_ptr<Token> token) {
    using namespace std;

    switch (token->type()) {
        case Token::TOKEN_TYPE_READY: {
            assert(inStage_(RTAC_WAITING_READY_TOKEN));

            string allReadyNames;
            pthread_rwlock_rdlock(&registerLock_);
            for (auto iter = registeredRequest_.begin(); iter != registeredRequest_.end(); ++iter) {
                allReadyNames.append(iter->first).append(tokenNameSplitDelimiter_);
            }
            pthread_rwlock_unlock(&registerLock_);

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
            pthread_rwlock_rdlock(&registerLock_);
            if (!registeredRequest_.empty()) {
                token = make_shared<Token>(
                        Token::TOKEN_TYPE_READY,
                        registeredRequest_.begin()->first);
                fromStageToStage_(RTAC_ALLREDUCING, RTAC_WAITING_READY_TOKEN);
                pthread_rwlock_unlock(&registerLock_);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TOKEN
                GLOBAL_INFO_WITH_RANK_THREAD_ID(
                        "forward token: " << token->desc()
                                          << " to stage RTAC_RECV_READY_TOKEN");
#endif
                fillTokenSendBufferAndNotify(token);
            } else {
                fromStageToStage_(RTAC_ALLREDUCING, RTAC_WAITING_TENSORS);
                pthread_rwlock_unlock(&registerLock_);
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

            pthread_rwlock_rdlock(&registerLock_);
            if (registeredRequest_.find(token->msg()) != registeredRequest_.end()) {
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
            pthread_rwlock_unlock(&registerLock_);
            break;
        }
        case Token::TOKEN_TYPE_SYNC: {
            assert(inStage_(RTAC_WAITING_SYNC_TOKEN));

            auto names = getNamesFromToken(*token);
            set<string> notReady;

            pthread_rwlock_rdlock(&registerLock_);
            for (auto iter = names.begin(); iter != names.end(); ++iter) {
                if (registeredRequest_.find(*iter) == registeredRequest_.end()) {
                    notReady.emplace(*iter);
                }
            }
            pthread_rwlock_unlock(&registerLock_);

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
        const std::string &key,
        std::shared_ptr<tensorflow::Tensor> sendTensor,
        std::shared_ptr<tensorflow::Tensor> recvTensor,
        std::function<void(StatusCode)> done,
        Operation op) {
    using namespace std;
    auto &global = Global::get();
    assert(!inStage_(RTAC_INIT));
    pthread_rwlock_wrlock(&registerLock_);
    assert(registeredRequest_.find(key) == registeredRequest_.end());
    auto *request = new TensorAllreduceRequest(key, sendTensor, recvTensor, done, op);
    registeredRequest_.emplace(key, request);
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_TF_OP_INTERACTION
    GLOBAL_INFO_WITH_RANK_THREAD_ID("op request to allreuce tensor: " << key);
#endif
    if (global.processRank() == 0) {
        if (inStage_(RTAC_WAITING_TENSORS)) {
            fromStageToStage_(RTAC_WAITING_TENSORS, RTAC_WAITING_READY_TOKEN);
            fillTokenSendBufferAndNotify(make_shared<Token>(Token::TOKEN_TYPE_READY, key));
        }
    } else {
        if (key == waitingReadyTokenName_) {
            fromStageToStage_(RTAC_WAITING_TENSORS, RTAC_WAITING_SYNC_TOKEN);
            fillTokenSendBufferAndNotify(make_shared<Token>(Token::TOKEN_TYPE_READY, key));
            waitingReadyTokenName_ = "";
        }
    }
    pthread_rwlock_unlock(&registerLock_);
    return STATUS_OK;
}

StatusCode RingTokenAllreduceController::allreduceByNames(const std::set<std::string> &names) {
    using namespace std;

    size_t allreduceSize = 0, elements = 0;

#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
    string allreducingTensorsDesc = "(\n";
#endif
    pthread_rwlock_wrlock(&registerLock_);
    map<string, TensorAllreduceRequest *> requests;
    for (auto i = names.begin(); i != names.end(); ++i) {
        const auto &key = *i;
        auto iter = registeredRequest_.find(key);

        if (iter == registeredRequest_.end()) {
            string r = "registered Tensors: {\n";
            for (auto _iter = registeredRequest_.begin(); _iter != registeredRequest_.end(); ++_iter) {
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
        assert(iter != registeredRequest_.end());
        TensorAllreduceRequest *request = iter->second;
#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
        allreducingTensorsDesc.append("\t").append(key).append(", \n");
#endif
        requests.emplace(key, request);
        allreduceSize += request->tensorSize();
        elements += request->elements();
        registeredRequest_.erase(iter);
    }
    pthread_rwlock_unlock(&registerLock_);


#if LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_ALLREDUCE_LOG_ALLREDUCE
    allreducingTensorsDesc.append(")");
    GLOBAL_INFO_WITH_RANK_THREAD_ID(
            "allreducing Tensors: " << allreducingTensorsDesc
                                    << " allreducing size: " << allreduceSize);
#endif
    return this->allreduceRequests(requests, elements, allreduceSize);
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

std::shared_ptr<Token> RingTokenAllreduceController::communicationReceiveTokenFrom(int sender) const {
    CALLING_ABSTRACT_INTERFACE_ERROR("RingTokenAllreduceController::communicationReceiveTokenFrom(int sender)");
}

void RingTokenAllreduceController::communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenAllreduceController::"
            "communicationSendTokenTo(int receiver, const std::shared_ptr<Token> &token)\n"
            "hint: please make sure the destructor of the implement class of"
            " RingTokenAllreduceController would call protected method:\n"
            " RingTokenAllreduceController::notifyAndWaitThreadToShutDown()"
            " to shut down sender receiver thread correctly"
    );
}

StatusCode RingTokenAllreduceController::allreduceRequests(
        const std::map<std::string, TensorAllreduceRequest *> &requests,
        size_t elements, size_t byteSize
) const {
    CALLING_ABSTRACT_INTERFACE_ERROR(
            "RingTokenAllreduceController::"
            "allreduceRequests(const std::map<std::string,"
            " TensorAllreduceRequest *> &requests,"
            " size_t elements, size_t byteSize)"
    );
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

void RingTokenAllreduceController::notifyAndWaitThreadToShutDown() {
    fillTokenSendBufferAndNotify(std::make_shared<Token>(Token::shutDownToken()));
    sendThread_.join();
    recvThread_.join();
}

}
}}}}