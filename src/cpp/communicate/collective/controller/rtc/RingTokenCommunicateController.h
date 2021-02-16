//
// Created by LYL232 on 2021/2/13.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H

#include <thread>
#include <pthread.h>
#include <queue>
#include <map>
#include <set>
#include "communicate/backend/CommunicationBackend.h"
#include "communicate/collective/controller/TensorsCollectiveCommunicateController.h"
#include "communicate/collective/controller/rtc/RingTokenCommunication.h"

#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_THREAD_MANNER 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TOKEN 0
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_TF_OP_INTERACTION 1
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_COMMUNICATE 1
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RING_TOKEN_COMMUNICATE_LOG_STAGE_CHANGE 0

namespace lyl232 { namespace experiment { namespace ddl { namespace rtc {

class RingTokenCommunicateController : public TensorsCollectiveCommunicateController {
public:
    typedef std::pair<std::string, std::string> RequestIdentifier;

    RingTokenCommunicateController(
            std::shared_ptr<CommunicationBackend> backend,
            std::shared_ptr<RingTokenCommunication> communicationImplement
    );

    RingTokenCommunicateController(const RingTokenCommunicateController &) = delete;

    RingTokenCommunicateController(RingTokenCommunicateController &&) = delete;


    bool initialized() const noexcept { return currentStage_ != RTCC_INIT; }

    virtual ~RingTokenCommunicateController();

    virtual StatusCode handleRequest(std::shared_ptr<TensorCollectiveCommunicateRequest>) override;

    virtual StatusCode allreduce(const Requests &requests) override;

    virtual StatusCode broadcast(const Requests &requests) override;

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

    std::thread sendThread_, recvThread_;

    std::queue<std::shared_ptr<Token>> outputtingTokenQueue_;
    RequestIdentifier waitingReadyTokenId_;

    std::map<RequestIdentifier, std::shared_ptr<TensorCollectiveCommunicateRequest>> registeredRequest_;

    std::shared_ptr<RingTokenCommunication> communicationImplement_;

    static std::string tokenSplitDelimiter_;
    static std::string tokenKeySplitDelimiter_;

    void fromStageToStage_(Stage from, Stage to);

    void forceToStage_(Stage to);

    bool inStage_(Stage stage);

    /**
     * 后台发送线程主函数
     */
    void sendMain_();

    void fillTokenSendBufferAndNotify_(std::shared_ptr<Token> token);

    /**
     * 后台接收线程主函数
     */
    void recvMain_();

    void handleReceivingTokenAsTokenGenerator_(std::shared_ptr<Token> token);

    void handleReceivingTokenAsTokenReceiver_(std::shared_ptr<Token> token);

    StatusCode communicateById_(const std::set<RequestIdentifier> &idSet);

    static std::set<RequestIdentifier> getIdSetFromToken_(const Token &token);

    static std::string stageName_(Stage stage);
};

}}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_RINGTOKENCOMMUNICATECONTROLLER_H
