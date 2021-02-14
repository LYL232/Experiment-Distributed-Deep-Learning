//
// Created by LYL232 on 2021/2/5.
//

#include "global/Global.h"
#include "global/initialize.h"

namespace lyl232 { namespace experiment { namespace ddl {

const std::shared_ptr<GlobalLog> Global::log(globalLogGetter());

Global Global::instance_(
        communicationBackendGetter(),
        collectiveCommunicateControllerGetter()
);

Global::Global(
        std::shared_ptr<CommunicationBackend> communicationBackend,
        std::shared_ptr<TensorsCollectiveCommunicateController> collectiveCommunicateController
) : communicationBackend_(communicationBackend),
    collectiveCommunicateController_(collectiveCommunicateController) {}


int Global::processes() const noexcept {
    return communicationBackend_->processes();
}

int Global::processRank() const noexcept {
    return communicationBackend_->processRank();
}

TensorsCollectiveCommunicateController &Global::collectiveCommunicateController() const noexcept {
    return *collectiveCommunicateController_;
}

CommunicationBackend &Global::communicationBackend() const noexcept {
    return *communicationBackend_;
}

Global &Global::get() noexcept {
    return instance_;
}

Global::~Global() {}

}}}