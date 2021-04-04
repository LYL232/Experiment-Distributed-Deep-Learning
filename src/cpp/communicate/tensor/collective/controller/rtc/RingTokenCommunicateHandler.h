//
// Created by LYL232 on 2021/3/23.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATEHANDLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATEHANDLER_H

#include <thread>
#include <pthread.h>
#include <queue>
#include <map>
#include <set>
#include "communicate/backend/CommunicationBackend.h"
#include "communicate/tensor/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/tensor/collective/controller/rtc/RingTokenCommunication.h"

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class RingTokenCommunicateHandler {
public:
    using Requests = TensorsCollectiveCommunicateController::Requests;

    typedef std::pair<std::string, std::string> RequestIdentifier;

    RingTokenCommunicateHandler(
            std::shared_ptr<Communicator> communicator,
            std::shared_ptr<RingTokenCommunication> communicationImplement
    );

    RingTokenCommunicateHandler(const RingTokenCommunicateHandler &) = delete;

    RingTokenCommunicateHandler(RingTokenCommunicateHandler &&) = delete;

    bool initialized() const noexcept { return currentStage_ != RTCC_INIT; }

    ~RingTokenCommunicateHandler();

    StatusCode handleRequest(const std::shared_ptr<TensorCollectiveCommunicateRequest> &request);

    StatusCode allreduce(const Requests &requests);

    StatusCode broadcast(const Requests &requests);

private:
    enum Stage : int {
        RTCC_INIT = 0,
        RTCC_WAITING_TENSORS = 1,
        RTCC_WAITING_READY_TOKEN = 2,
        RTCC_WAITING_SYNC_TOKEN = 3,
        RTCC_WAITING_COMMUNICATE_TOKEN = 4,
        RTCC_COMMUNICATING = 5,
        RTCC_SHUT_DOWN,
    };

    pthread_mutex_t outMutex_;
    pthread_rwlock_t registerLock_, stageLock_;
    pthread_cond_t outputTokenCond_;
    Stage currentStage_;

    std::thread *sendThread_, *recvThread_;

    std::queue<std::shared_ptr<Token>> outputtingTokenQueue_;
    RequestIdentifier waitingReadyTokenId_;

    std::map<RequestIdentifier, std::shared_ptr<TensorCollectiveCommunicateRequest>> registeredRequest_;

    std::shared_ptr<RingTokenCommunication> communicationImplement_;

    std::shared_ptr<Communicator> communicator_;

    const static std::string tokenSplitDelimiter_;
    const static std::string tokenKeySplitDelimiter_;

    void fromStageToStage_(Stage from, Stage to);

    void forceToStage_(Stage to);

    bool inStage_(Stage stage);

    /**
     * 后台发送线程主函数
     */
    void sendMain_();

    void fillTokenSendBufferAndNotify_(const std::shared_ptr<Token> &token);

    /**
     * 后台接收线程主函数
     */
    void recvMain_();

    void handleReceivingTokenAsTokenGenerator_(std::shared_ptr<Token> token);

    void handleReceivingTokenAsTokenReceiver_(std::shared_ptr<Token> token);

    StatusCode communicateById_(const std::set<RequestIdentifier> &idSet);

    static std::set<RequestIdentifier> getIdSetFromToken_(const Token &token);

    static std::string stageName_(Stage stage) noexcept;
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATEHANDLER_H
