//
// Created by LYL232 on 2021/2/11.
//

#ifndef LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H
#define LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H

#include "def.h"
#include "communicate/backend/Communicator.h"

namespace lyl232 { namespace experiment { namespace ddl {

class CommunicationBackend {
    friend class Global;
public:
    CommunicationBackend() = default;

    CommunicationBackend(const CommunicationBackend &) = delete;

    CommunicationBackend(CommunicationBackend &&) = delete;

    /**
     * 包含全部进程的通信域
     * @return
     */
    virtual std::shared_ptr<Communicator> worldCommunicator() const;

    virtual ~CommunicationBackend() = default;
protected:
    // 初始化时是空指针, 需要子类实现时赋值(reset)
    static std::shared_ptr<Communicator> world_;
};

}}}


#endif //LYL232_EXPERIMENT_DISTRIBUTED_DEEP_LEARNING_COMMUNICATIONBACKEND_H
