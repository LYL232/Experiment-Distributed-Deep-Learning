//
// Created by LYL232 on 2021/2/5.
//

#include "global/Global.h"
#include "global/initialize.h"

namespace lyl232 { namespace experiment { namespace ddl {

const std::shared_ptr<GlobalLog> Global::log(globalLogGetter());

Global Global::instance_(
        communicationBackendGetter(),
        collectiveCommunicateControllerGetter(),
        end2EndCommunicateControllerGetter(),
        messageControllerGetter()
);

Global::Global(
        std::shared_ptr<CommunicationBackend> communicationBackend,
        std::shared_ptr<TensorsCollectiveCommunicateController> collectiveController,
        std::shared_ptr<TensorEnd2EndCommunicateController> end2EndController,
        std::shared_ptr<MessageController> messageController
) : communicationBackend_(communicationBackend),
    collectiveCommunicateController_(collectiveController),
    end2EndCommunicateController_(end2EndController),
    messageController_(messageController) {}


int Global::processes() const noexcept {
    return communicationBackend_->processes();
}

int Global::processRank() const noexcept {
    return communicationBackend_->processRank();
}

TensorsCollectiveCommunicateController &Global::collectiveCommunicateController() const noexcept {
    return *collectiveCommunicateController_;
}

TensorEnd2EndCommunicateController &Global::end2EndCommunicateController() const noexcept {
    return *end2EndCommunicateController_;
}

MessageController &Global::messageController() const noexcept {
    return *messageController_;
}

CommunicationBackend &Global::communicationBackend() const noexcept {
    return *communicationBackend_;
}

Global &Global::get() noexcept {
    return instance_;
}

Global::~Global() {}

}}}