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
) noexcept: communicationBackend_(std::move(communicationBackend)),
            collectiveCommunicateController_(std::move(collectiveController)),
            end2EndCommunicateController_(std::move(end2EndController)),
            messageController_(std::move(messageController)),
            communicatorMap_(new std::map<Communicator::ID, std::shared_ptr<Communicator>>()) {
    const auto &world = communicationBackend_->worldCommunicator();
    communicatorMap_->emplace(world->id(), world);
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

std::shared_ptr<Communicator> Global::worldCommunicator() const noexcept {
    return communicationBackend().worldCommunicator();
}

const std::shared_ptr<Communicator> &Global::getCommunicator(Communicator::ID id)
const noexcept {
    return (*communicatorMap_)[id];
}

void Global::detachCommunicator(Communicator::ID id) const noexcept {
    communicatorMap_->erase(id);
}

Global::~Global() {
    delete communicatorMap_;
}

}}}